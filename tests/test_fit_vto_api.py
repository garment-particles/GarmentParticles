from __future__ import annotations

import base64
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.fit_vto_api import (  # noqa: E402
    DEFAULT_IMAGE_BASE_URL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_TEXT_BASE_URL,
    DEFAULT_TEXT_MODEL,
    GARMENT_RETEXTURE_PROMPT,
    GatewayClient,
    PHOTOREALISTIC_GARMENT_PROMPT,
    endpoint_url,
    extract_response_text,
    read_credential,
    read_last_credential,
)


class FakeResponse:
    def __init__(self, payload, status_code=200, content=b""):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.headers = {}
        self.text = ""
        self.content = content

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.posts = []
        self.gets = []

    def post(self, url, **kwargs):
        self.posts.append((url, kwargs))
        return self.responses.pop(0)

    def get(self, url, **kwargs):
        self.gets.append((url, kwargs))
        return self.responses.pop(0)


class FitVTOAPITest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_key_file(self) -> Path:
        path = self.tmp_path / "key.txt"
        path.write_text(
            "# inference then management\n"
            "INFERENCE=secret-value\n\n"
            "MANAGEMENT=management-value\n",
            encoding="utf-8",
        )
        return path

    def test_read_last_credential(self):
        self.assertEqual(
            read_last_credential(self.make_key_file()), "management-value"
        )

    def test_read_second_to_last_credential(self):
        self.assertEqual(
            read_credential(self.make_key_file(), index_from_end=2),
            "secret-value",
        )

    def test_endpoint_url_does_not_duplicate_endpoint(self):
        base = "https://example.test/openai/v1"
        self.assertEqual(endpoint_url(base, "responses"), base + "/responses")
        self.assertEqual(
            endpoint_url(base + "/responses", "responses"), base + "/responses"
        )

    def test_generate_uses_exact_requested_payload(self):
        image = base64.b64encode(b"image-bytes").decode("ascii")
        session = FakeSession([FakeResponse({"data": [{"b64_json": image}]})])
        client = GatewayClient(key_file=self.make_key_file(), session=session)

        payload = client.generate_images("sunset", n=1, size="1024x1024")

        self.assertEqual(payload["data"][0]["b64_json"], image)
        url, request = session.posts[0]
        self.assertEqual(url, DEFAULT_IMAGE_BASE_URL + "/images/generations")
        self.assertEqual(
            request["json"],
            {
                "model": DEFAULT_IMAGE_MODEL,
                "prompt": "sunset",
                "n": 1,
                "size": "1024x1024",
            },
        )
        self.assertEqual(
            request["headers"]["Authorization"], "Bearer secret-value"
        )

    def test_generate_can_request_explicit_quality(self):
        image = base64.b64encode(b"image-bytes").decode("ascii")
        session = FakeSession([FakeResponse({"data": [{"b64_json": image}]})])
        client = GatewayClient(key_file=self.make_key_file(), session=session)

        client.generate_images("sunset", quality="medium")

        self.assertEqual(session.posts[0][1]["json"]["quality"], "medium")

    def test_list_models_returns_sorted_ids(self):
        session = FakeSession(
            [FakeResponse({"data": [{"id": "model-z"}, {"id": "model-a"}]})]
        )
        client = GatewayClient(key_file=self.make_key_file(), session=session)

        self.assertEqual(client.list_models(), ["model-a", "model-z"])
        url, request = session.gets[0]
        self.assertEqual(url, DEFAULT_IMAGE_BASE_URL + "/models")
        self.assertEqual(
            request["headers"]["Authorization"], "Bearer secret-value"
        )

    def test_text_response_uses_supplied_luna_contract(self):
        session = FakeSession(
            [
                FakeResponse(
                    {
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": "Hello!"}
                                ],
                            }
                        ]
                    }
                )
            ]
        )
        client = GatewayClient(key_file=self.make_key_file(), session=session)

        text, _ = client.respond("Hello! Can you help me?", max_output_tokens=1024)

        self.assertEqual(text, "Hello!")
        _, request = session.posts[0]
        self.assertEqual(
            session.posts[0][0], DEFAULT_TEXT_BASE_URL + "/responses"
        )
        self.assertEqual(
            request["json"],
            {
                "model": DEFAULT_TEXT_MODEL,
                "input": [
                    {"role": "user", "content": "Hello! Can you help me?"}
                ],
                "max_output_tokens": 1024,
            },
        )

    def test_vision_response_adds_image_without_changing_envelope(self):
        source = self.tmp_path / "input.png"
        source.write_bytes(b"not-a-real-png")
        session = FakeSession([FakeResponse({"output_text": "caption"})])
        client = GatewayClient(key_file=self.make_key_file(), session=session)

        text, _ = client.respond("Describe it", image_paths=[source])

        self.assertEqual(text, "caption")
        content = session.posts[0][1]["json"]["input"][0]["content"]
        self.assertEqual(content[0]["type"], "input_image")
        self.assertTrue(content[0]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(
            content[1], {"type": "input_text", "text": "Describe it"}
        )

    def test_edit_uses_multipart_image_endpoint(self):
        source = self.tmp_path / "render.png"
        source.write_bytes(b"render-bytes")
        image = base64.b64encode(b"edited").decode("ascii")
        session = FakeSession([FakeResponse({"data": [{"b64_json": image}]})])
        client = GatewayClient(key_file=self.make_key_file(), session=session)

        client.edit_image(source, "Preserve geometry", size="1024x1024")

        url, request = session.posts[0]
        self.assertEqual(url, DEFAULT_IMAGE_BASE_URL + "/images/edits")
        self.assertIsNone(request["json"])
        self.assertEqual(
            request["data"],
            {
                "model": DEFAULT_IMAGE_MODEL,
                "prompt": "Preserve geometry",
                "n": "1",
                "size": "1024x1024",
            },
        )
        self.assertEqual(
            request["files"]["image"],
            ("render.png", b"render-bytes", "image/png"),
        )

    def test_edit_can_request_explicit_quality(self):
        source = self.tmp_path / "render.png"
        source.write_bytes(b"render-bytes")
        image = base64.b64encode(b"edited").decode("ascii")
        session = FakeSession([FakeResponse({"data": [{"b64_json": image}]})])
        client = GatewayClient(key_file=self.make_key_file(), session=session)

        client.edit_image(source, "Preserve geometry", quality="medium")

        self.assertEqual(session.posts[0][1]["data"]["quality"], "medium")

    def test_edit_bytes_does_not_require_a_source_file(self):
        image = base64.b64encode(b"edited").decode("ascii")
        session = FakeSession([FakeResponse({"data": [{"b64_json": image}]})])
        client = GatewayClient(key_file=self.make_key_file(), session=session)

        client.edit_image_bytes(
            b"render-bytes",
            "Preserve geometry",
            filename="archive-render.png",
            quality="medium",
        )

        request = session.posts[0][1]
        self.assertEqual(
            request["files"]["image"],
            ("archive-render.png", b"render-bytes", "image/png"),
        )
        self.assertEqual(request["data"]["quality"], "medium")

    def test_photorealistic_prompt_locks_garment_identity(self):
        description = "a blue fitted bodice and ankle-length skirt"
        prompt = PHOTOREALISTIC_GARMENT_PROMPT.format(
            garment_description=description
        )

        self.assertIn(description, prompt)
        self.assertIn("Preserve exactly the garment silhouette", prompt)
        self.assertIn("Do not add, remove, shorten, lengthen, or redesign", prompt)

    def test_retexture_prompt_changes_surface_but_locks_geometry(self):
        description = "deep green silk velvet with gold botanical embroidery"
        prompt = GARMENT_RETEXTURE_PROMPT.format(
            texture_description=description
        )

        self.assertIn(description, prompt)
        self.assertIn("Actively replace", prompt)
        self.assertIn("Preserve exactly the garment geometry", prompt)
        self.assertIn("Keep everything outside the garments unchanged", prompt)

    def test_save_images_decodes_base64(self):
        expected = b"generated-image"
        payload = {
            "data": [
                {"b64_json": base64.b64encode(expected).decode("ascii")}
            ]
        }
        client = GatewayClient(key_file=self.tmp_path / "unused")
        output = self.tmp_path / "result.png"

        self.assertEqual(client.save_images(payload, output), [output])
        self.assertEqual(output.read_bytes(), expected)

    def test_extract_response_text_supports_chat_fallback(self):
        payload = {"choices": [{"message": {"content": "fallback"}}]}
        self.assertEqual(extract_response_text(payload), "fallback")


if __name__ == "__main__":
    unittest.main()
