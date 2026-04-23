from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List

from config.settings import IMGBB_API_KEY, SCAM_OCR_LANG, SERPAPI_API_KEY
from utils.image_utils import crop_bbox, draw_text, save_image

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

try:
    from serpapi import GoogleSearch
except ImportError:  # pragma: no cover - optional dependency
    GoogleSearch = None

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - optional dependency
    PaddleOCR = None


class ModuleCOcrSerpApiRoi:
    """Module C: OCR, ROI extraction, and evidence collection."""

    def __init__(self) -> None:
        self.ocr_init_error: str | None = None
        self.ocr_engine = self._init_ocr_engine()

    def run(
        self,
        image_path: str | Path,
        image,
        route_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        image_path = Path(image_path)
        roi_result = self._extract_rois(image_path, image, route_result)
        ocr_result = self._run_ocr(image_path)
        serpapi_result = self._search_sources(image_path) if route_result["routing_flags"]["run_module_c"] else {
            "available": False,
            "exact_match_count": 0,
            "top_sources": [],
            "error": "Module C route not selected.",
        }

        suspicious_keywords = self._find_suspicious_keywords(ocr_result["text"])
        evidence = list(roi_result["evidence"])
        if suspicious_keywords:
            evidence.append(f"OCR suspicious keywords: {', '.join(suspicious_keywords[:5])}.")
        if serpapi_result["available"]:
            evidence.append(
                f"SerpAPI exact matches: {serpapi_result['exact_match_count']} from {', '.join(serpapi_result['top_sources'][:3]) or 'unknown sources'}."
            )
        if not evidence:
            evidence.append("Module C did not collect strong ROI or OCR evidence from the current image.")

        summary = (
            f"Module C processed {len(roi_result['roi_items'])} ROI regions, "
            f"OCR text length {len(ocr_result['text'])}, "
            f"and found {len(suspicious_keywords)} suspicious keywords."
        )

        return {
            "module_id": "C",
            "module_name": "OCR + ROI + SerpAPI",
            "status": "success",
            "summary": summary,
            "ocr_text": ocr_result["text"],
            "ocr_available": ocr_result["available"],
            "ocr_error": ocr_result.get("error"),
            "roi_items": roi_result["roi_items"],
            "roi_output_files": roi_result["roi_output_files"],
            "serpapi_available": serpapi_result["available"],
            "serpapi_error": serpapi_result.get("error"),
            "serpapi_summary": serpapi_result,
            "suspicious_keywords": suspicious_keywords,
            "evidence": evidence,
        }

    def annotate(self, image, analysis: Dict[str, Any]):
        text = f"Module C | OCR chars {len(analysis.get('ocr_text', ''))}"
        return draw_text(image, text=text, position=(18, 60), color=(0, 120, 220), font_scale=0.75, thickness=2)

    def _extract_rois(self, image_path: Path, image, route_result: Dict[str, Any]) -> Dict[str, Any]:
        roi_items: List[Dict[str, Any]] = []
        roi_output_files: List[str] = []
        evidence: List[str] = []

        output_dir = image_path.parent.parent / "runs" / "roi_exports" / image_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)

        for index, detection in enumerate(route_result["selected_object_detections"][:4], start=1):
            try:
                cropped = crop_bbox(image, detection["bbox"], pad=8)
                roi_name = f"{image_path.stem}_roi_{index}.jpg"
                save_image(output_dir / roi_name, cropped)
                roi_output_files.append(str((output_dir / roi_name).name))
                roi_items.append(
                    {
                        "index": index,
                        "label": detection.get("class_name", "object"),
                        "score": float(detection.get("score", 0.0)),
                        "bbox": detection.get("bbox", []),
                    }
                )
            except Exception:
                continue

        if roi_items:
            labels = ", ".join(item["label"] for item in roi_items[:3])
            evidence.append(f"ROI extraction focused on detected objects: {labels}.")

        return {
            "roi_items": roi_items,
            "roi_output_files": roi_output_files,
            "evidence": evidence,
        }

    def _run_ocr(self, image_path: Path) -> Dict[str, Any]:
        if self.ocr_engine is None:
            return {"available": False, "text": "", "error": self.ocr_init_error or "OCR engine unavailable."}

        try:
            results = self.ocr_engine.predict(str(image_path))
            lines: List[str] = []
            for result in results or []:
                for item in result.get("rec_texts", []) or []:
                    text = str(item).strip()
                    if text:
                        lines.append(text)
            return {"available": True, "text": " ".join(lines)}
        except Exception as exc:
            return {"available": False, "text": "", "error": str(exc)}

    def _search_sources(self, image_path: Path) -> Dict[str, Any]:
        if not IMGBB_API_KEY or not SERPAPI_API_KEY:
            return {"available": False, "exact_match_count": 0, "top_sources": [], "error": "Missing API key."}
        if requests is None or GoogleSearch is None:
            return {"available": False, "exact_match_count": 0, "top_sources": [], "error": "Missing requests or serpapi package."}

        try:
            image_url = self._upload_to_imgbb(image_path)
            exact = GoogleSearch(
                {
                    "engine": "google_lens",
                    "type": "exact_matches",
                    "url": image_url,
                    "api_key": SERPAPI_API_KEY,
                }
            ).get_dict()
            exact_matches = exact.get("exact_matches", []) or []
            top_sources = [str(item.get("source")) for item in exact_matches[:5] if item.get("source")]
            return {
                "available": True,
                "image_url": image_url,
                "exact_match_count": len(exact_matches),
                "top_sources": top_sources,
            }
        except Exception as exc:
            return {"available": False, "exact_match_count": 0, "top_sources": [], "error": str(exc)}

    @staticmethod
    def _find_suspicious_keywords(text: str) -> List[str]:
        keywords = [
            "投資",
            "獲利",
            "保證",
            "line",
            "whatsapp",
            "匯款",
            "立即",
            "限時",
            "中獎",
            "虛擬貨幣",
        ]
        lowered = (text or "").lower()
        return [keyword for keyword in keywords if keyword.lower() in lowered]

    def _init_ocr_engine(self):
        if PaddleOCR is None:
            self.ocr_init_error = "paddleocr package is not installed."
            return None
        try:
            return PaddleOCR(
                lang=SCAM_OCR_LANG,
                device="cpu",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception as exc:
            self.ocr_init_error = str(exc)
            return None

    @staticmethod
    def _upload_to_imgbb(image_path: Path) -> str:
        assert requests is not None
        with image_path.open("rb") as file:
            encoded = base64.b64encode(file.read())
        response = requests.post(
            "https://api.imgbb.com/1/upload",
            data={"key": IMGBB_API_KEY, "image": encoded},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"]["url"]
