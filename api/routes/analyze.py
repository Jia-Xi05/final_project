from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from flask import Blueprint, current_app, jsonify, request, send_from_directory

from api.services.pipeline_service import run_full_pipeline, run_scam_pipeline, run_yolo_pipeline
from config.settings import INPUT_DIR, OUTPUT_DIR

analyze_bp = Blueprint("analyze", __name__, url_prefix="/api")


@analyze_bp.route("/analyze/full", methods=["POST"])
def analyze_full():
    return _handle_analysis(run_full_pipeline, "Full pipeline analyze failed")


@analyze_bp.route("/analyze/yolo", methods=["POST"])
def analyze_yolo():
    return _handle_analysis(run_yolo_pipeline, "YOLO analyze failed")


@analyze_bp.route("/analyze/scam", methods=["POST"])
def analyze_scam():
    return _handle_analysis(run_scam_pipeline, "Scam analyze failed")


@analyze_bp.route("/outputs/<path:filename>", methods=["GET"])
def get_output_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename)


@analyze_bp.route("/inputs/<path:filename>", methods=["GET"])
def get_input_file(filename: str):
    return send_from_directory(INPUT_DIR, filename)


def _handle_analysis(pipeline_fn, log_message: str):
    try:
        save_path, filename = _save_uploaded_image()
        result = pipeline_fn(save_path)
        result["annotated_image_url"] = f"/api/outputs/{result['annotated_filename']}"
        result["original_image_url"] = f"/api/inputs/{filename}"
        return jsonify(result)
    except Exception as exc:
        current_app.logger.exception(log_message)
        return jsonify({"status": "error", "message": str(exc)}), 500


def _save_uploaded_image() -> tuple[Path, str]:
    if "image" not in request.files:
        raise ValueError("No image file uploaded.")

    file = request.files["image"]
    if not file or not file.filename:
        raise ValueError("Empty upload.")

    suffix = Path(file.filename).suffix.lower() or ".jpg"
    filename = f"upload_{uuid4().hex[:8]}{suffix}"
    save_path = INPUT_DIR / filename
    file.save(save_path)
    return save_path, filename
