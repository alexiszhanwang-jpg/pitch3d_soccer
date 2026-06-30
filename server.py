from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
from werkzeug.utils import secure_filename
from flask import Flask, jsonify, request, send_from_directory

from process_real_image import process_real_image

WORK_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = WORK_DIR / "input" / "uploads"
OUTPUT_DIR = WORK_DIR / "output_real"
OBJECT_MODEL = WORK_DIR / "runs" / "detect" / "runs" / "detect" / "football_person_ball_v1_mps" / "weights" / "best.pt"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


def _allowed_image(filename: str) -> bool:
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix in ALLOWED_EXTENSIONS


def _unique_upload_path(filename: str) -> Path:
    safe_name = secure_filename(filename) or "upload.jpg"
    stem = Path(safe_name).stem or "upload"
    suffix = Path(safe_name).suffix.lower() or ".jpg"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    candidate = UPLOAD_DIR / f"{timestamp}-{stem}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = UPLOAD_DIR / f"{timestamp}-{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def process_upload_image(image_path: Path) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process_real_image(
        str(image_path),
        output_dir=str(OUTPUT_DIR),
        auto_detect=True,
        object_model_path=str(OBJECT_MODEL),
        use_color_fallback=True,
    )
    scene_graph_path = OUTPUT_DIR / "scene_graph.json"
    if not scene_graph_path.exists():
        raise RuntimeError("处理完成但没有生成 scene_graph.json")
    import json
    return json.loads(scene_graph_path.read_text(encoding="utf-8"))


def create_app() -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024

    @app.get("/")
    def index():
        return send_from_directory(WORK_DIR / "web_renderer", "index.html")

    @app.get("/web_renderer/")
    def web_renderer_index():
        return send_from_directory(WORK_DIR / "web_renderer", "index.html")

    @app.get("/web_renderer/<path:filename>")
    def web_renderer_file(filename: str):
        return send_from_directory(WORK_DIR / "web_renderer", filename)

    @app.get("/output_real/<path:filename>")
    def output_file(filename: str):
        return send_from_directory(OUTPUT_DIR, filename)

    @app.get("/input/<path:filename>")
    def input_file(filename: str):
        return send_from_directory(WORK_DIR / "input", filename)

    @app.post("/api/process")
    def api_process():
        upload = request.files.get("image")
        if upload is None or not upload.filename:
            return jsonify({"error": "请上传图片文件"}), 400
        if not _allowed_image(upload.filename):
            return jsonify({"error": "只支持 jpg/jpeg/png/webp 图片"}), 400

        image_path = _unique_upload_path(upload.filename)
        upload.save(image_path)

        try:
            scene_graph = process_upload_image(image_path)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500

        return jsonify({
            "image_path": str(image_path),
            "scene_graph_path": str(OUTPUT_DIR / "scene_graph.json"),
            "scene_graph": scene_graph,
        })

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8787"))
    create_app().run(host="127.0.0.1", port=port, debug=False)
