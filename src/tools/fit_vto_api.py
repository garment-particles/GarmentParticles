#!/usr/bin/env python3
"""OpenAI-compatible replacements for FIT's Gemini and Nano Banana calls.

The default configuration targets NVIDIA's OpenAI-compatible gateway and uses
the exact model routes requested for this project:

* ``openai/openai/gpt-image-2`` for image generation and editing
* ``azure/openai/gpt-5.6-luna`` for text and vision responses

Credentials are never copied into this repository.  The client reads the
second-to-last non-empty, non-comment credential from ``--key-file`` by
default, leaving the final IHub management credential separate from the
``sk-`` inference credential.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Optional, Sequence

import requests


DEFAULT_KEY_FILE = Path(
    "/scratch/m000133/george/foveated_diffusion_3d/key.txt"
)
DEFAULT_IMAGE_BASE_URL = "https://inference-api.nvidia.com/v1"
DEFAULT_TEXT_BASE_URL = "https://inference-api.nvidia.com/openai/v1"
DEFAULT_IMAGE_MODEL = "openai/openai/gpt-image-2"
DEFAULT_TEXT_MODEL = "azure/openai/gpt-5.6-luna"
DEFAULT_CREDENTIAL_INDEX_FROM_END = 2

HEAD_SHOES_PROMPT = """Change the head to make it look photorealistic. Add realistic {hair_style} hair, but the hair should always be behind the shoulder and never at the front. Add {shoe_type} if feet are visible. Make sure that everything else stays identical, including the human pose, garment shape, size, design, and position."""

GARMENT_TRY_OFF_PROMPT = (
    "Create an in-shop product image of the top garment only against a plain "
    "white background. Preserve its color, texture, graphics, closures, seams, "
    "sleeves, neckline, and proportions exactly. Do not include a person, body "
    "parts, hanger, mannequin, accessories, or a lower garment."
)

PAIRED_PERSON_PROMPT = """Generate a new image where the upper garment is changed, and keep everything else exactly the same, including the bottom garment, face, human pose, body shape, camera, lighting, background, and position."""

PHOTOREALISTIC_GARMENT_PROMPT = """Use case: identity-preserving garment render conversion.
Asset type: photorealistic full-body fashion catalog photograph.

Convert the supplied synthetic garment/mannequin render into a realistic studio photograph of one adult fashion model wearing exactly the same outfit. Source description: {garment_description}

Replace only the synthetic mannequin/body rendering with a plausible real adult person and make the visible garment materials physically realistic. If the source contains a floating or headless garment, place it naturally on one complete adult body in the same front-facing pose and framing.

Preserve exactly the garment silhouette and proportions; neckline, hood, or collar; sleeves and cuffs; waistband; skirt shape, length, and slit; panel, seam, and closure locations; colors; and relative fit. Do not add, remove, shorten, lengthen, or redesign any garment part.

Add natural skin, face, eyes, hands, legs, and realistic understated shoes wherever visible. Keep hair behind the shoulders so it does not cover garment details. Show the complete outfit head-to-toe, centered, against a neutral light-gray fashion studio background with soft photographic lighting, realistic skin pores, fabric grain, wrinkles, and shadows. No text, logo, watermark, extra people, props, jewelry, bags, or other accessories."""

GARMENT_RETEXTURE_PROMPT = """Use case: precise garment surface retexturing.
Asset type: photorealistic full-body fashion catalog photograph.

Change only the visible surface appearance of every garment piece to this new design: {texture_description}

Actively replace the source garment's existing colors, material, weave, pattern, print, surface finish, and reflectance. The old garment texture and palette should not remain unless explicitly requested by the new design. Render the new materials physically realistically, including appropriate fine-scale grain, weave, nap, embroidery, highlights, wrinkles, and shadows.

Preserve exactly the garment geometry and construction: silhouette, volume, fit, neckline or hood state, sleeve shape and length, cuffs, waistband, skirt shape and length, slit, hems, panel boundaries, seam locations, and closures. Do not add or remove garment pieces, pockets, straps, sleeves, collars, hoods, slits, or structural seams.

Keep everything outside the garments unchanged: the same person and identity, face, hair, skin, body shape, pose, hands, legs, shoes, camera, framing, studio background, and lighting. Keep the full outfit visible head-to-toe. No brands, text, logos, watermark, extra people, props, jewelry, bags, or accessories."""

PAPER_CAPTION_PROMPT = """Describe the garment in the image in exactly two sentences. The first sentence must describe the top garment, and the second sentence must describe the bottom garment. Treat the input as an illustration of garment type, style, and size: ignore its existing texture and propose a plausible new texture, logo, and design. Add pockets, zippers, buttons, and other garment details only when appropriate. Keep the complete answer under 50 words and output only the two sentences."""

RELEASED_CODE_CAPTION_PROMPT = """Describe this image in exactly three sentences separated by a period and one space. Sentence 1 must describe the person and studio setting. Sentence 2 must describe the upper garment, including its type, silhouette, sleeves, neckline, material, texture, color, and appropriate construction details. Sentence 3 must describe the lower garment. Keep the complete answer under 80 words and output only those three sentences."""

MEASUREMENT_PROMPT = """Estimate the height, bust, hips, and waist of the human in centimeters. Also estimate the bust, length from shoulder or shoulder strap to bottom hem, and sleeve length of the top garment. Return exactly this format with one precise number per field and no ranges or extra text: Human Height: xx cm, Human Bust: xx cm, Human Hips: xx cm, Human Waist: xx cm, Garment Length: xx cm, Garment Sleeve Length: xx cm, Garment Bust: xx cm."""

CHEST_QA_PROMPT = (
    "Does the upper garment cover the person's chest? Return only 'pass' if it "
    "does and only 'fail' if it does not."
)
GROIN_QA_PROMPT = """Does the image contain a bottom garment (skirt, pants, underwear, boxers, leggings, or shorts) that covers the person's groin area? Return only 'pass' if it does and only 'fail' if it does not."""


class GatewayError(RuntimeError):
    """An OpenAI-compatible gateway request failed."""


def read_credential(path: Path, index_from_end: int = 1) -> str:
    """Return a non-empty, non-comment credential counted from file end.

    ``NAME=value`` entries are accepted for compatibility with dotenv-style
    files.  The returned value must never be logged by callers.
    """

    if index_from_end < 1:
        raise GatewayError("Credential index from end must be at least 1")

    entries = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not entries:
        raise GatewayError(f"No credential found in {path}")

    if len(entries) < index_from_end:
        raise GatewayError(
            f"Credential {index_from_end} from the end does not exist in {path}"
        )

    value = entries[-index_from_end]
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", value):
        value = value.split("=", 1)[1]
    value = value.strip().strip("\"'")
    if not value:
        raise GatewayError(
            f"Credential {index_from_end} from the end in {path} is empty"
        )
    return value


def read_last_credential(path: Path) -> str:
    """Return the final non-empty, non-comment credential in *path*."""

    return read_credential(path, index_from_end=1)


def endpoint_url(base_url: str, endpoint: str) -> str:
    """Join a gateway base URL and endpoint without duplicating ``/v1``."""

    base = base_url.rstrip("/")
    suffix = endpoint.strip("/")
    if base.endswith("/" + suffix):
        return base
    return base + "/" + suffix


def image_data_url(path: Path) -> str:
    media_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def extract_response_text(payload: dict[str, Any]) -> str:
    """Extract assistant text from Responses or Chat Completions JSON."""

    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    pieces = []
    output = payload.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content", [])
            if isinstance(content, str):
                pieces.append(content)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in {"output_text", "text"}:
                    text = part.get("text")
                    if isinstance(text, str):
                        pieces.append(text)
    if pieces:
        return "\n".join(piece.strip() for piece in pieces if piece.strip()).strip()

    choices = payload.get("choices", [])
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message", {})
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"].strip()

    raise GatewayError("Gateway response did not contain any assistant text")


def _decode_base64_image(value: str) -> bytes:
    if value.startswith("data:"):
        try:
            value = value.split(",", 1)[1]
        except IndexError as exc:
            raise GatewayError("Malformed image data URL in gateway response") from exc
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise GatewayError("Gateway returned invalid base64 image data") from exc


def image_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Return standard or common gateway-specific image result objects."""

    data = payload.get("data")
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data
    images = payload.get("images")
    if isinstance(images, list):
        normalized = []
        for item in images:
            normalized.append(item if isinstance(item, dict) else {"b64_json": item})
        return normalized
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list) and all(
        isinstance(item, dict) for item in artifacts
    ):
        return artifacts
    raise GatewayError("Gateway response did not contain an image result list")


class GatewayClient:
    def __init__(
        self,
        key_file: Path = DEFAULT_KEY_FILE,
        image_base_url: str = DEFAULT_IMAGE_BASE_URL,
        text_base_url: str = DEFAULT_TEXT_BASE_URL,
        image_model: str = DEFAULT_IMAGE_MODEL,
        text_model: str = DEFAULT_TEXT_MODEL,
        credential_index_from_end: int = DEFAULT_CREDENTIAL_INDEX_FROM_END,
        timeout: float = 600.0,
        max_attempts: int = 4,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.key_file = key_file
        self.image_base_url = image_base_url
        self.text_base_url = text_base_url
        self.image_model = image_model
        self.text_model = text_model
        self.credential_index_from_end = credential_index_from_end
        self.timeout = timeout
        self.max_attempts = max(1, max_attempts)
        self.session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        credential = read_credential(
            self.key_file, index_from_end=self.credential_index_from_end
        )
        return {"Authorization": f"Bearer {credential}"}

    @staticmethod
    def _error_body(response: requests.Response) -> str:
        try:
            payload = response.json()
            error = payload.get("error", payload)
            if isinstance(error, dict):
                return str(error.get("message") or error.get("code") or error)[:1000]
            return str(error)[:1000]
        except (ValueError, TypeError):
            return response.text[:1000]

    @staticmethod
    def _retry_delay(response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.5, float(retry_after))
            except ValueError:
                pass
        return min(60.0, 2.0 ** (attempt - 1))

    def _request(
        self,
        endpoint: str,
        *,
        base_url: Optional[str] = None,
        json_body: Optional[dict[str, Any]] = None,
        data: Optional[dict[str, str]] = None,
        files: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        url = endpoint_url(base_url or self.text_base_url, endpoint)
        response: Optional[requests.Response] = None
        last_exception: Optional[Exception] = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.session.post(
                    url,
                    headers=self._headers(),
                    json=json_body,
                    data=data,
                    files=files,
                    timeout=(30.0, self.timeout),
                )
            except requests.RequestException as exc:
                last_exception = exc
                if attempt < self.max_attempts:
                    time.sleep(min(60.0, 2.0 ** (attempt - 1)))
                    continue
                raise GatewayError(
                    f"Gateway request failed after {attempt} attempts: {exc}"
                ) from exc

            if response.ok:
                try:
                    value = response.json()
                except ValueError as exc:
                    raise GatewayError("Gateway returned a non-JSON response") from exc
                if not isinstance(value, dict):
                    raise GatewayError("Gateway response is not a JSON object")
                return value

            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt == self.max_attempts:
                raise GatewayError(
                    f"Gateway HTTP {response.status_code}: {self._error_body(response)}"
                )
            time.sleep(self._retry_delay(response, attempt))

        if last_exception is not None:
            raise GatewayError(f"Gateway request failed: {last_exception}")
        raise GatewayError("Gateway request failed without a response")

    def list_models(self, *, api: str = "image") -> list[str]:
        """List model IDs visible to the current gateway credential."""

        base_url = self.image_base_url if api == "image" else self.text_base_url
        url = endpoint_url(base_url, "models")
        try:
            response = self.session.get(
                url,
                headers=self._headers(),
                timeout=(30.0, min(self.timeout, 120.0)),
            )
        except requests.RequestException as exc:
            raise GatewayError(f"Could not list gateway models: {exc}") from exc
        if not response.ok:
            raise GatewayError(
                f"Gateway HTTP {response.status_code}: {self._error_body(response)}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GatewayError("Gateway returned a non-JSON model list") from exc
        data = payload.get("data", []) if isinstance(payload, dict) else []
        if not isinstance(data, list):
            raise GatewayError("Gateway model list has an unexpected shape")
        return sorted(
            str(item["id"])
            for item in data
            if isinstance(item, dict) and item.get("id")
        )

    def generate_images(
        self,
        prompt: str,
        *,
        n: int = 1,
        size: str = "1024x1024",
        quality: Optional[str] = None,
    ) -> dict[str, Any]:
        # Keep this payload exactly aligned with the user-provided contract.
        body = {
            "model": self.image_model,
            "prompt": prompt,
            "n": n,
            "size": size,
        }
        if quality is not None:
            body["quality"] = quality
        return self._request(
            "images/generations", base_url=self.image_base_url, json_body=body
        )

    def edit_image(
        self,
        image_path: Path,
        prompt: str,
        *,
        n: int = 1,
        size: str = "1024x1024",
        quality: Optional[str] = None,
    ) -> dict[str, Any]:
        media_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        return self.edit_image_bytes(
            image_path.read_bytes(),
            prompt,
            filename=image_path.name,
            media_type=media_type,
            n=n,
            size=size,
            quality=quality,
        )

    def edit_image_bytes(
        self,
        image: bytes,
        prompt: str,
        *,
        filename: str = "input.png",
        media_type: str = "image/png",
        n: int = 1,
        size: str = "1024x1024",
        quality: Optional[str] = None,
    ) -> dict[str, Any]:
        """Edit an in-memory image without creating a source file.

        Immutable bytes also make HTTP retries safe: every attempt receives
        the complete image instead of an already-consumed file handle.
        """

        files = {"image": (filename, image, media_type)}
        data = {
            "model": self.image_model,
            "prompt": prompt,
            "n": str(n),
            "size": size,
        }
        if quality is not None:
            data["quality"] = quality
        return self._request(
            "images/edits", base_url=self.image_base_url, data=data, files=files
        )

    def respond(
        self,
        prompt: str,
        *,
        image_paths: Sequence[Path] = (),
        max_output_tokens: int = 1024,
    ) -> tuple[str, dict[str, Any]]:
        if image_paths:
            content: Any = [
                {"type": "input_image", "image_url": image_data_url(path)}
                for path in image_paths
            ]
            content.append({"type": "input_text", "text": prompt})
        else:
            # Keep text-only calls exactly aligned with the supplied contract.
            content = prompt
        body = {
            "model": self.text_model,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_output_tokens,
        }
        payload = self._request(
            "responses", base_url=self.text_base_url, json_body=body
        )
        return extract_response_text(payload), payload

    def _download_image(self, url: str) -> bytes:
        response = self.session.get(url, timeout=(30.0, self.timeout))
        if response.status_code in {401, 403}:
            response = self.session.get(
                url, headers=self._headers(), timeout=(30.0, self.timeout)
            )
        if not response.ok:
            raise GatewayError(
                f"Image download HTTP {response.status_code}: "
                f"{self._error_body(response)}"
            )
        return response.content

    def image_bytes(self, item: dict[str, Any]) -> bytes:
        for key in ("b64_json", "base64", "image"):
            value = item.get(key)
            if isinstance(value, str) and value:
                if value.startswith(("http://", "https://")):
                    return self._download_image(value)
                return _decode_base64_image(value)
        url = item.get("url") or item.get("image_url")
        if isinstance(url, str) and url:
            if url.startswith("data:"):
                return _decode_base64_image(url)
            return self._download_image(url)
        raise GatewayError("Image result contained neither base64 data nor a URL")

    def save_images(self, payload: dict[str, Any], output_path: Path) -> list[Path]:
        items = image_items(payload)
        if not items:
            raise GatewayError("Gateway returned an empty image result list")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        paths = []
        for index, item in enumerate(items, 1):
            if len(items) == 1:
                path = output_path
            else:
                path = output_path.with_name(
                    f"{output_path.stem}_{index}{output_path.suffix or '.png'}"
                )
            path.write_bytes(self.image_bytes(item))
            paths.append(path)
        return paths


def _add_image_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument(
        "--quality",
        choices=("low", "medium", "high", "auto"),
        help="Explicit GPT Image output quality; omit to preserve gateway defaults.",
    )
    parser.add_argument(
        "--usage-output",
        type=Path,
        help="Optionally save the response usage object as JSON.",
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument(
        "--credential-index-from-end",
        type=int,
        default=DEFAULT_CREDENTIAL_INDEX_FROM_END,
        help="Credential position counted from the end of the key file (default: 2).",
    )
    parser.add_argument(
        "--image-base-url",
        default=os.environ.get("NVIDIA_IMAGE_BASE_URL", DEFAULT_IMAGE_BASE_URL),
    )
    parser.add_argument(
        "--text-base-url",
        default=os.environ.get("NVIDIA_TEXT_BASE_URL", DEFAULT_TEXT_BASE_URL),
    )
    parser.add_argument("--image-model", default=DEFAULT_IMAGE_MODEL)
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-attempts", type=int, default=4)

    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="List visible gateway model IDs")
    models.add_argument("--contains", help="Only print IDs containing this text")
    models.add_argument("--api", choices=("image", "text"), default="image")

    generate = commands.add_parser("generate", help="Text-to-image generation")
    generate.add_argument("--prompt", required=True)
    _add_image_output_args(generate)

    edit = commands.add_parser("edit", help="Edit one image using GPT Image 2")
    edit.add_argument("--image", type=Path, required=True)
    edit.add_argument("--prompt", required=True)
    _add_image_output_args(edit)

    respond = commands.add_parser("respond", help="Luna text or vision response")
    respond.add_argument("--prompt", required=True)
    respond.add_argument("--image", type=Path, action="append", default=[])
    respond.add_argument("--max-output-tokens", type=int, default=1024)
    respond.add_argument("--output", type=Path)

    head = commands.add_parser(
        "fit-head-shoes", help="FIT head/hair/shoe edit (Nano Banana replacement)"
    )
    head.add_argument("--image", type=Path, required=True)
    head.add_argument("--hair-style", default="natural")
    head.add_argument("--shoe-type", default="realistic casual shoes")
    _add_image_output_args(head)

    try_off = commands.add_parser(
        "fit-try-off", help="FIT lay-flat garment edit (Nano Banana replacement)"
    )
    try_off.add_argument("--image", type=Path, required=True)
    _add_image_output_args(try_off)

    paired = commands.add_parser(
        "fit-paired-person",
        help="FIT paired-person edit (Nano Banana replacement)",
    )
    paired.add_argument("--image", type=Path, required=True)
    paired.add_argument(
        "--garment-description",
        required=True,
        help="Description of the replacement upper garment.",
    )
    _add_image_output_args(paired)

    realistic = commands.add_parser(
        "fit-photorealistic",
        help="Convert a synthetic garment render into a realistic worn photograph",
    )
    realistic.add_argument("--image", type=Path, required=True)
    realistic.add_argument(
        "--garment-description",
        required=True,
        help="Visual description used to lock the source garment's design.",
    )
    realistic.add_argument(
        "--prompt-output",
        type=Path,
        help="Optionally save the exact image-edit prompt for reproducibility.",
    )
    _add_image_output_args(realistic)

    retexture = commands.add_parser(
        "fit-retexture",
        help="Replace garment surface appearance while preserving its geometry",
    )
    retexture.add_argument("--image", type=Path, required=True)
    retexture.add_argument(
        "--texture-description",
        required=True,
        help="Target material, palette, pattern, and surface treatment.",
    )
    retexture.add_argument(
        "--prompt-output",
        type=Path,
        help="Optionally save the exact image-edit prompt for reproducibility.",
    )
    _add_image_output_args(retexture)

    caption = commands.add_parser(
        "fit-caption", help="FIT garment caption (Gemini replacement)"
    )
    caption.add_argument("--image", type=Path, required=True)
    caption.add_argument(
        "--format", choices=("paper", "released-code"), default="released-code"
    )
    caption.add_argument("--max-output-tokens", type=int, default=512)
    caption.add_argument("--output", type=Path)

    measurements = commands.add_parser(
        "fit-measurements", help="FIT coarse measurements (Gemini replacement)"
    )
    measurements.add_argument("--image", type=Path, required=True)
    measurements.add_argument("--max-output-tokens", type=int, default=512)
    measurements.add_argument("--output", type=Path)

    qa = commands.add_parser("fit-qa", help="FIT chest/groin QA (Gemini replacement)")
    qa.add_argument("--image", type=Path, required=True)
    qa.add_argument("--output", type=Path)

    return parser.parse_args(argv)


def _write_or_print(text: str, output: Optional[Path]) -> None:
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"Wrote {output}")


def _print_image_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        print(f"Wrote {path}")


def _report_image_usage(
    payload: dict[str, Any], output: Optional[Path]
) -> None:
    """Print image token usage and optionally persist it without image data."""

    usage = payload.get("usage")
    if not isinstance(usage, dict):
        print("Usage: not returned by gateway")
        return
    serialized = json.dumps(usage, indent=2, sort_keys=True)
    print("Usage:\n" + serialized)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Wrote {output}")


def _save_and_report_images(
    client: GatewayClient, payload: dict[str, Any], args: argparse.Namespace
) -> None:
    _print_image_paths(client.save_images(payload, args.output))
    _report_image_usage(payload, args.usage_output)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    client = GatewayClient(
        key_file=args.key_file,
        image_base_url=args.image_base_url,
        text_base_url=args.text_base_url,
        image_model=args.image_model,
        text_model=args.text_model,
        credential_index_from_end=args.credential_index_from_end,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
    )

    try:
        if args.command == "models":
            model_ids = client.list_models(api=args.api)
            if args.contains:
                needle = args.contains.casefold()
                model_ids = [
                    model_id
                    for model_id in model_ids
                    if needle in model_id.casefold()
                ]
            print("\n".join(model_ids))
        elif args.command == "generate":
            payload = client.generate_images(
                args.prompt, n=args.n, size=args.size, quality=args.quality
            )
            _save_and_report_images(client, payload, args)
        elif args.command == "edit":
            payload = client.edit_image(
                args.image,
                args.prompt,
                n=args.n,
                size=args.size,
                quality=args.quality,
            )
            _save_and_report_images(client, payload, args)
        elif args.command == "respond":
            text, _ = client.respond(
                args.prompt,
                image_paths=args.image,
                max_output_tokens=args.max_output_tokens,
            )
            _write_or_print(text, args.output)
        elif args.command == "fit-head-shoes":
            prompt = HEAD_SHOES_PROMPT.format(
                hair_style=args.hair_style, shoe_type=args.shoe_type
            )
            payload = client.edit_image(
                args.image,
                prompt,
                n=args.n,
                size=args.size,
                quality=args.quality,
            )
            _save_and_report_images(client, payload, args)
        elif args.command == "fit-try-off":
            payload = client.edit_image(
                args.image,
                GARMENT_TRY_OFF_PROMPT,
                n=args.n,
                size=args.size,
                quality=args.quality,
            )
            _save_and_report_images(client, payload, args)
        elif args.command == "fit-paired-person":
            prompt = (
                PAIRED_PERSON_PROMPT
                + " Replace the upper garment with this garment: "
                + args.garment_description
            )
            payload = client.edit_image(
                args.image,
                prompt,
                n=args.n,
                size=args.size,
                quality=args.quality,
            )
            _save_and_report_images(client, payload, args)
        elif args.command == "fit-photorealistic":
            prompt = PHOTOREALISTIC_GARMENT_PROMPT.format(
                garment_description=args.garment_description.strip()
            )
            if args.prompt_output is not None:
                _write_or_print(prompt, args.prompt_output)
            payload = client.edit_image(
                args.image,
                prompt,
                n=args.n,
                size=args.size,
                quality=args.quality,
            )
            _save_and_report_images(client, payload, args)
        elif args.command == "fit-retexture":
            prompt = GARMENT_RETEXTURE_PROMPT.format(
                texture_description=args.texture_description.strip()
            )
            if args.prompt_output is not None:
                _write_or_print(prompt, args.prompt_output)
            payload = client.edit_image(
                args.image,
                prompt,
                n=args.n,
                size=args.size,
                quality=args.quality,
            )
            _save_and_report_images(client, payload, args)
        elif args.command == "fit-caption":
            prompt = (
                PAPER_CAPTION_PROMPT
                if args.format == "paper"
                else RELEASED_CODE_CAPTION_PROMPT
            )
            text, _ = client.respond(
                prompt,
                image_paths=[args.image],
                max_output_tokens=args.max_output_tokens,
            )
            _write_or_print(text, args.output)
        elif args.command == "fit-measurements":
            text, _ = client.respond(
                MEASUREMENT_PROMPT,
                image_paths=[args.image],
                max_output_tokens=args.max_output_tokens,
            )
            _write_or_print(text, args.output)
        elif args.command == "fit-qa":
            chest, _ = client.respond(
                CHEST_QA_PROMPT, image_paths=[args.image], max_output_tokens=32
            )
            groin, _ = client.respond(
                GROIN_QA_PROMPT, image_paths=[args.image], max_output_tokens=32
            )
            result = json.dumps(
                {"chest_coverage": chest.strip(), "groin_coverage": groin.strip()},
                indent=2,
            )
            _write_or_print(result, args.output)
        else:
            raise AssertionError(f"Unhandled command: {args.command}")
    except (GatewayError, FileNotFoundError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
