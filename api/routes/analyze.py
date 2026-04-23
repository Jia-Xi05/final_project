from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from flask import Blueprint, jsonify, request, current_app, send_from_directory

from api.services.pipeline_service import run_yolo_pipeline
from config.settings import INPUT_DIR, OUTPUT_DIR

analyze_bp = Blueprint("analyze", __name__, url_prefix="/api")


@analyze_bp.route("/analyze/yolo", methods=["POST"])
def analyze_yolo():
    try:
        if "image" not in request.files:
            return jsonify({"status": "error", "message": "No image file uploaded."}), 400

        file = request.files["image"]
        if not file or not file.filename:
            return jsonify({"status": "error", "message": "Empty upload."}), 400

        suffix = Path(file.filename).suffix.lower() or ".jpg"
        filename = f"upload_{uuid4().hex[:8]}{suffix}"
        save_path = INPUT_DIR / filename
        file.save(save_path)

        result = run_yolo_pipeline(save_path)
        result["annotated_image_url"] = f"/api/outputs/{result['annotated_filename']}"
        result["original_image_url"] = f"/api/inputs/{filename}"
        return jsonify(result)
    except Exception as exc:
        current_app.logger.exception("YOLO analyze failed")
        return jsonify({"status": "error", "message": str(exc)}), 500


@analyze_bp.route("/outputs/<path:filename>", methods=["GET"])
def get_output_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


@analyze_bp.route("/inputs/<path:filename>", methods=["GET"])
def get_input_file(filename: str):
    return send_from_directory(INPUT_DIR, filename)
