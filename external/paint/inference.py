"""
HTTP client for the InteractGarment inference server (src/tools/run_server.py).

The GUI never loads the model in-process. It posts user-drawn 2D guides to a
Flask server running on a GPU box and gets back point clouds. Host/port are
configurable via env vars or constructor args:

    INTERACT_GARMENT_HOST   default: 127.0.0.1
    INTERACT_GARMENT_PORT   default: 12345

Public surface (used by main.py, garment.py, and any panel callbacks):

    Inference()                          - constructs a client
    Inference.health()                   - True iff server reachable
    Inference.generate(prompt, n)        - unconditional Nx3 sampling
    Inference.predict_with_silhouette(prompt, guide_xy, **kw)
    Inference.predict_with_silhouette_stream(prompt, guide_xy, on_event, **kw)
    Inference.predict_with_pattern(prompt, guide_uv, **kw)
    Inference.run_inference(vertices)    - backwards-compat shim used by
                                           main.py's File->Inference action;
                                           routes through silhouette since
                                           projection panel guides are XY.

    predict_points(input_vertices, callback)
                                         - drop-in replacement for the
                                           garment.py stub.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import requests

from definitions import Vertex, prediction_to_vertices


DEFAULT_HOST = os.environ.get("INTERACT_GARMENT_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("INTERACT_GARMENT_PORT", "12345"))
DEFAULT_TIMEOUT = 600  # seconds; inverse-design calls can take minutes


class InferenceError(RuntimeError):
    """Raised when the server returns a non-2xx response or an error payload."""


class Inference:
    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        self.base_url = f"http://{host}:{port}"
        self.timeout = timeout
        print(f"Inference client -> {self.base_url}")

    # ---- low-level ---------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        url = f"{self.base_url}{path}"
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
        except requests.exceptions.RequestException as e:
            raise InferenceError(f"POST {url} failed: {e}") from e
        if resp.status_code != 200:
            try:
                body = resp.json()
            except ValueError:
                body = {"error": resp.text}
            raise InferenceError(
                f"POST {url} returned {resp.status_code}: {body.get('error', body)}"
            )
        body = resp.json()
        if "error" in body:
            raise InferenceError(f"POST {url}: {body['error']}")
        return body

    def health(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200 and resp.json().get("status") == "ok"
        except requests.exceptions.RequestException:
            return False

    # ---- typed endpoints ---------------------------------------------------

    def generate(self, prompt: str = "", num_points: int = 2048) -> np.ndarray:
        """Unconditional /generate. Returns (N, 3) float array."""
        body = self._post("/generate", {"prompt": prompt, "num_points": int(num_points)})
        return np.asarray(body["points"], dtype=np.float32)

    def predict_with_silhouette(
        self,
        prompt: str = "",
        guide_xy: Sequence[Sequence[float]] = (),
        num_points: int = 2048,
        lr: float = 0.1,
        early_stop_t: float = 0.7,
        n_opt_steps: int = 2,
        num_timesteps: int = 100,
        mode: str = "completion",
    ) -> np.ndarray:
        """/predict_with_silouette. guide is XY world-space. Returns (N, 6)
        with columns [u, v, x, y, z, flag]."""
        payload = _opt_payload(
            prompt, guide_xy, num_points, lr, early_stop_t, n_opt_steps,
            num_timesteps, mode,
        )
        return np.asarray(self._post("/predict_with_silouette", payload)["points"],
                          dtype=np.float32)

    def predict_with_silhouette_stream(
        self,
        prompt: str = "",
        guide_xy: Sequence[Sequence[float]] = (),
        num_points: int = 2048,
        lr: float = 0.1,
        early_stop_t: float = 0.7,
        n_opt_steps: int = 2,
        num_timesteps: int = 100,
        mode: str = "completion",
        on_event: Optional[Callable[[dict], None]] = None,
        K: Optional[Sequence[Sequence[float]]] = None,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        radius: float = 150.0,
        look_at: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """
        Streaming variant. Calls on_event(dict) for each {progress|result|error}
        message as it arrives. Returns the final (N, 3) point cloud, or raises.
        Server transport is application/x-ndjson.
        """
        payload = _opt_payload(
            prompt, guide_xy, num_points, lr, early_stop_t, n_opt_steps,
            num_timesteps, mode,
        )
        payload.update({
            "K": (np.asarray(K).tolist() if K is not None else [
                [1.0, 0.0, 250.0], [0.0, 1.0, 250.0], [0.0, 0.0, 1.0]
            ]),
            "azimuth": float(azimuth),
            "elevation": float(elevation),
            "radius": float(radius),
            "look_at": list(look_at),
        })
        url = f"{self.base_url}/predict_with_silouette_stream"
        final_points: Optional[np.ndarray] = None
        with requests.post(url, json=payload, timeout=self.timeout, stream=True) as resp:
            if resp.status_code != 200:
                raise InferenceError(
                    f"POST {url} returned {resp.status_code}: {resp.text}"
                )
            for raw in resp.iter_lines():
                if not raw:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                # tolerate either ndjson or SSE-style "data: <json>"
                line = raw[len("data: "):] if raw.startswith("data: ") else raw
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if on_event is not None:
                    try:
                        on_event(msg)
                    except Exception:
                        pass
                if msg.get("type") == "error":
                    raise InferenceError(f"server error: {msg.get('error')}")
                if msg.get("type") == "result":
                    final_points = np.asarray(msg["points"], dtype=np.float32)
        if final_points is None:
            raise InferenceError("stream ended without a 'result' message")
        return final_points

    def predict_with_pattern(
        self,
        prompt: str = "",
        guide_uv: Sequence[Sequence[float]] = (),
        num_points: int = 2048,
        lr: float = 0.1,
        early_stop_t: float = 0.7,
        n_opt_steps: int = 2,
        num_timesteps: int = 100,
        mode: str = "completion",
    ) -> np.ndarray:
        """/predict_with_pattern. guide is UV pattern-space. Returns (N, 6)
        with columns [x, y, z, u, v, c]."""
        payload = _opt_payload(
            prompt, guide_uv, num_points, lr, early_stop_t, n_opt_steps,
            num_timesteps, mode,
        )
        return np.asarray(self._post("/predict_with_pattern", payload)["points"],
                          dtype=np.float32)

    # ---- backwards-compat with main.py -------------------------------------

    def run_inference(self, vertices: Iterable[Vertex]) -> List[Vertex]:
        """
        main.py wires File -> Inference to this method, passing
        projectionPanel._vertices (drawn in XY world space). Route through the
        silhouette endpoint and return a list of Vertex objects.

        The server returns (N, 6) [u, v, x, y, z, flag]; if a 3-channel
        [x, y, z] response is ever seen, pad u/v/flag with zeros.
        """
        guide_xy = [(v.x, v.y) for v in vertices]
        pred = self.predict_with_silhouette(prompt="", guide_xy=guide_xy)
        if pred.ndim != 2 or pred.shape[1] not in (3, 6):
            raise InferenceError(f"unexpected prediction shape {pred.shape}")
        if pred.shape[1] == 3:
            full = np.zeros((pred.shape[0], 6), dtype=np.float32)
            full[:, 2:5] = pred
            pred = full
        return prediction_to_vertices(pred)


# ---- module-level helpers --------------------------------------------------

def _opt_payload(
    prompt: str,
    guide_2d: Sequence[Sequence[float]],
    num_points: int,
    lr: float,
    early_stop_t: float,
    n_opt_steps: int,
    num_timesteps: int,
    mode: str,
) -> dict:
    """Build the JSON body shared by the three optimization endpoints. The
    server validates types strictly (lr/early_stop_t must be Python floats,
    not ints; num_points/n_opt_steps/num_timesteps must be ints), so cast."""
    return {
        "prompt": str(prompt or ""),
        "guide": [list(map(float, p)) for p in guide_2d],
        "num_points": int(num_points),
        "lr": float(lr),
        "early_stop_t": float(early_stop_t),
        "n_opt_steps": int(n_opt_steps),
        "num_timesteps": int(num_timesteps),
        "mode": str(mode),
    }


_default_client: Optional[Inference] = None


def _get_default_client() -> Inference:
    global _default_client
    if _default_client is None:
        _default_client = Inference()
    return _default_client


def predict_points(input_vertices: Iterable[Vertex],
                   callback: Callable[[List[Vertex]], None]) -> None:
    """Drop-in replacement for garment.predict_points(). Synchronous: blocks
    until the server responds, then calls callback with the result vertices."""
    out = _get_default_client().run_inference(input_vertices)
    callback(out)
