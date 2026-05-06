from __future__ import annotations

import subprocess
import sys
import zipfile
import os
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import requests

from config.settings import (
    OUTPUT_DIR,
    TRUFOR_AUTO_DOWNLOAD_WEIGHTS,
    TRUFOR_ENABLE,
    TRUFOR_EXP_NAME,
    TRUFOR_GPU_ID,
    TRUFOR_MODEL_FILE,
    TRUFOR_MODEL_DIR,
    TRUFOR_OUT_DIR,
    TRUFOR_TIMEOUT_SEC,
    TRUFOR_WEIGHTS_ZIP_PATH,
    TRUFOR_WEIGHTS_ZIP_URL,
    TRUFOR_WORK_DIR,
)
from utils.image_utils import draw_text, save_image

DEFAULT_TIMEOUT = 90
MAX_ZIP_BYTES = 600_000_000


class ModuleBTruFor:
    """Module B: deepfake/tampering detection with TruFor."""

    def run(self, image_path: str | Path, route_result: Dict[str, Any]) -> Dict[str, Any]:
        image_path = Path(image_path)
        should_run = bool(route_result.get("routing_flags", {}).get("run_deepfake_branch"))
        if not should_run:
            return self._skipped("Module B skipped because deepfake branch is not selected.")
        if not TRUFOR_ENABLE:
            return self._unavailable("TRUFOR_ENABLE is disabled.")
        if not TRUFOR_WORK_DIR.exists():
            return self._unavailable(
                f"TruFor code not found. Expected: {TRUFOR_WORK_DIR}. "
                "Clone TruFor into models/trufor/TruFor first."
            )

        ensure_model = self._ensure_trufor_model()
        if ensure_model is not None:
            return ensure_model

        TRUFOR_OUT_DIR.mkdir(parents=True, exist_ok=True)
        output_subdir = TRUFOR_OUT_DIR / image_path.stem
        output_subdir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "test.py",
            "-g",
            str(TRUFOR_GPU_ID),
            "-in",
            image_path.resolve().as_posix(),
            "-out",
            str(output_subdir.resolve()),
            "-exp",
            TRUFOR_EXP_NAME,
            "TEST.MODEL_FILE",
            str(TRUFOR_MODEL_FILE),
        ]
        try:
            env = os.environ.copy()
            # PyTorch 2.6+ changed torch.load default (weights_only=True), which
            # breaks legacy TruFor checkpoints saved as full pickles.
            env.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
            completed = subprocess.run(
                cmd,
                cwd=str(TRUFOR_WORK_DIR),
                capture_output=True,
                text=True,
                timeout=TRUFOR_TIMEOUT_SEC,
                check=False,
                env=env,
            )
        except Exception as exc:
            return self._unavailable(f"TruFor process launch failed: {exc}")

        if completed.returncode != 0:
            error_text = (completed.stderr or completed.stdout or "").strip()
            return self._unavailable(f"TruFor inference failed: {error_text or completed.returncode}")

        npz_path = self._find_npz_output(output_subdir, image_path)
        if npz_path is None:
            debug_text = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
            traceback_hint = _extract_traceback(debug_text)
            if traceback_hint:
                return self._unavailable(f"TruFor inference failed inside test loop: {traceback_hint}")
            return self._unavailable(
                f"TruFor output npz not found in {output_subdir}. "
                "The process exited without file output; check TruFor runtime dependencies."
            )

        try:
            payload = np.load(npz_path)
            score = float(payload["score"])
            an_map = np.asarray(payload["map"], dtype=np.float32)
            conf_map = np.asarray(payload["conf"], dtype=np.float32)
        except Exception as exc:
            return self._unavailable(f"Failed to parse TruFor npz: {exc}")

        map_file = OUTPUT_DIR / f"trufor_map_{image_path.stem}.jpg"
        conf_file = OUTPUT_DIR / f"trufor_conf_{image_path.stem}.jpg"
        mask_file = OUTPUT_DIR / f"trufor_mask_{image_path.stem}.jpg"
        save_image(map_file, _to_heatmap(an_map))
        save_image(conf_file, _to_heatmap(conf_map))
        save_image(mask_file, _to_mask(an_map))

        verdict = "tampering_suspected" if score >= 0.5 else "likely_clean"
        return {
            "module_id": "B",
            "module_name": "TruFor Deepfake Branch",
            "status": "success",
            "available": True,
            "summary": f"TruFor score={score:.3f}, verdict={verdict}.",
            "trufor_score": score,
            "trufor_verdict": verdict,
            "npz_path": str(npz_path),
            "map_filename": map_file.name,
            "conf_filename": conf_file.name,
            "mask_filename": mask_file.name,
        }

    def annotate(self, image: np.ndarray, analysis: Dict[str, Any]) -> np.ndarray:
        if analysis.get("status") != "success" or analysis.get("trufor_score") is None:
            return image
        return draw_text(
            image,
            text=f"Module B TruFor score: {float(analysis['trufor_score']):.2f}",
            position=(18, 90),
            color=(32, 32, 220),
            font_scale=0.75,
            thickness=2,
        )

    def _ensure_trufor_model(self) -> Dict[str, Any] | None:
        if TRUFOR_MODEL_FILE.exists():
            return None
        if not TRUFOR_AUTO_DOWNLOAD_WEIGHTS:
            return self._unavailable(
                f"TruFor model missing: {TRUFOR_MODEL_FILE}. "
                "Download weights from TruFor README inference section."
            )

        try:
            self._download_and_extract_weights()
        except Exception as exc:
            return self._unavailable(
                f"Auto-download TruFor weights failed: {exc}. "
                "Please download from TruFor README inference section manually."
            )

        if not TRUFOR_MODEL_FILE.exists():
            return self._unavailable(
                f"Weights downloaded but model not found at {TRUFOR_MODEL_FILE}. "
                "Check TruFor folder structure."
            )
        return None

    def _download_and_extract_weights(self) -> None:
        TRUFOR_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        TRUFOR_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(TRUFOR_WEIGHTS_ZIP_URL, timeout=DEFAULT_TIMEOUT, stream=True)
        response.raise_for_status()

        downloaded = 0
        with TRUFOR_WEIGHTS_ZIP_PATH.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > MAX_ZIP_BYTES:
                    raise RuntimeError("TruFor weight zip too large.")
                f.write(chunk)

        with zipfile.ZipFile(TRUFOR_WEIGHTS_ZIP_PATH, "r") as zf:
            zf.extractall(TRUFOR_MODEL_DIR)

        extracted_candidate = TRUFOR_MODEL_DIR / "weights" / "trufor.pth.tar"
        if extracted_candidate.exists() and not TRUFOR_MODEL_FILE.exists():
            extracted_candidate.replace(TRUFOR_MODEL_FILE)

    @staticmethod
    def _find_npz_output(output_subdir: Path, image_path: Path) -> Path | None:
        candidates = [
            output_subdir / f"{image_path.name}.npz",
            output_subdir / f"{image_path.stem}.jpg.npz",
            output_subdir / f"{image_path.stem}.png.npz",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        npz_files = sorted(output_subdir.glob("*.npz"))
        return npz_files[0] if npz_files else None

    @staticmethod
    def _skipped(message: str) -> Dict[str, Any]:
        return {
            "module_id": "B",
            "module_name": "TruFor Deepfake Branch",
            "status": "skipped",
            "available": False,
            "summary": message,
            "trufor_score": None,
        }

    @staticmethod
    def _unavailable(message: str) -> Dict[str, Any]:
        return {
            "module_id": "B",
            "module_name": "TruFor Deepfake Branch",
            "status": "unavailable",
            "available": False,
            "summary": f"Module B unavailable: {message}",
            "error": message,
            "trufor_score": None,
        }


def _to_heatmap(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = arr - arr.min()
    max_val = float(arr.max()) if arr.size else 0.0
    if max_val > 0:
        arr = arr / max_val
    gray = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def _to_mask(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    threshold = np.quantile(arr, 0.85) if arr.size else 0.5
    mask = (arr >= threshold).astype(np.uint8) * 255
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


def _extract_traceback(text: str) -> str:
    if not text:
        return ""
    marker = "Traceback (most recent call last):"
    idx = text.find(marker)
    if idx < 0:
        return ""
    snippet = text[idx:].strip()
    return snippet[:2500]
