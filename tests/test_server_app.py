import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class UploadProcessTests(unittest.TestCase):
    def test_upload_image_saves_file_and_returns_scene_graph(self):
        import server

        scene_graph = {
            "schema_version": "test",
            "source": {"image_path": "unused"},
            "pitch": {"length_m": 105, "width_m": 68},
            "players": [],
            "ball": None,
            "carrier_id": None,
        }

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(server, "WORK_DIR", Path(tmp)), \
                 patch.object(server, "UPLOAD_DIR", Path(tmp) / "input" / "uploads"), \
                 patch.object(server, "OUTPUT_DIR", Path(tmp) / "output_real"), \
                 patch.object(server, "process_upload_image", return_value=scene_graph) as processor:
                app = server.create_app()
                response = app.test_client().post(
                    "/api/process",
                    data={"image": (io.BytesIO(b"fake image bytes"), "frame.jpg")},
                    content_type="multipart/form-data",
                )

                self.assertEqual(response.status_code, 200)
                payload = response.get_json()
                self.assertEqual(payload["scene_graph"], scene_graph)
                saved_path = Path(payload["image_path"])
                self.assertTrue(saved_path.exists())
                self.assertEqual(saved_path.read_bytes(), b"fake image bytes")
                processor.assert_called_once_with(saved_path)


if __name__ == "__main__":
    unittest.main()
