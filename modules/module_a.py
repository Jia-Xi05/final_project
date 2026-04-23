from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import IMGBB_API_KEY, SCAM_CLIP_MODEL_NAME, SCAM_CLIP_PRETRAINED, SERPAPI_API_KEY

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

try:
    import open_clip
except ImportError:  # pragma: no cover - optional dependency
    open_clip = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    import requests
except ImportError:  # pragma: no cover - optional dependency
    requests = None

try:
    from serpapi import GoogleSearch
except ImportError:  # pragma: no cover - optional dependency
    GoogleSearch = None


class ModuleAOpenClipSerpApi:
    """Module A: similarity and semantic image analysis."""

    def __init__(self) -> None:
        self.clip_init_error: Optional[str] = None
        self.clip_bundle = self._init_clip_bundle()
        self.semantic_labels = [
            "news event photo",
            "social media portrait",
            "chat screenshot scam",
            "financial advertisement",
            "political rally photo",
            "celebrity portrait",
        ]

    def run(self, image_path: str | Path, route_result: Dict[str, Any]) -> Dict[str, Any]:
        image_path = Path(image_path)
        should_run = bool(route_result["routing_flags"]["run_module_a"])

        if not should_run:
            return {
                "module_id": "A",
                "module_name": "OpenCLIP + SerpAPI",
                "status": "skipped",
                "summary": "Module A was skipped because the router did not select the similarity-analysis branch.",
                "semantic_scores": {},
                "serpapi_available": False,
                "serpapi_summary": {},
                "signals": [],
            }

        semantic_result = self._analyze_semantics(image_path)
        serpapi_result = self._search_similar_images(image_path)

        signals: List[str] = []
        if semantic_result["available"] and semantic_result["scores"]:
            top_label = max(semantic_result["scores"], key=semantic_result["scores"].get)
            signals.append(f"Top semantic label: {top_label}")
        if serpapi_result["available"]:
            signals.append(
                f"SerpAPI matches: visual {serpapi_result['visual_match_count']}, exact {serpapi_result['exact_match_count']}"
            )
        if route_result["face_count"] == 0 and route_result["person_count"] > 0:
            signals.append("Router detected people without clear faces, matching the left-side branch in the project design.")

        summary = self._build_summary(
            route_result=route_result,
            semantic_result=semantic_result,
            serpapi_result=serpapi_result,
        )

        return {
            "module_id": "A",
            "module_name": "OpenCLIP + SerpAPI",
            "status": "success",
            "semantic_scores": semantic_result["scores"],
            "semantic_available": semantic_result["available"],
            "semantic_error": semantic_result.get("error"),
            "serpapi_available": serpapi_result["available"],
            "serpapi_error": serpapi_result.get("error"),
            "serpapi_summary": serpapi_result,
            "signals": signals,
            "summary": summary,
        }

    def _analyze_semantics(self, image_path: Path) -> Dict[str, Any]:
        if self.clip_bundle is None:
            return {"available": False, "scores": {}, "error": self.clip_init_error or "OpenCLIP unavailable."}
        if Image is None:
            return {"available": False, "scores": {}, "error": "Pillow unavailable."}

        model = self.clip_bundle["model"]
        preprocess = self.clip_bundle["preprocess"]
        tokenizer = self.clip_bundle["tokenizer"]

        try:
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
            tokens = tokenizer(self.semantic_labels)
            with torch.no_grad():
                image_features = model.encode_image(image)
                text_features = model.encode_text(tokens)
                probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

            scores = {
                self.semantic_labels[index]: round(float(probs[index].item()), 4)
                for index in range(len(self.semantic_labels))
                if float(probs[index].item()) >= 0.05
            }
            return {"available": True, "scores": scores}
        except Exception as exc:
            return {"available": False, "scores": {}, "error": str(exc)}

    def _search_similar_images(self, image_path: Path) -> Dict[str, Any]:
        if not IMGBB_API_KEY or not SERPAPI_API_KEY:
            return {
                "available": False,
                "visual_match_count": 0,
                "exact_match_count": 0,
                "top_sources": [],
                "error": "IMGBB_API_KEY or SERPAPI_API_KEY is missing.",
            }
        if requests is None or GoogleSearch is None:
            return {
                "available": False,
                "visual_match_count": 0,
                "exact_match_count": 0,
                "top_sources": [],
                "error": "requests or serpapi package is missing.",
            }

        try:
            image_url = self._upload_to_imgbb(image_path)
            visual = GoogleSearch(
                {
                    "engine": "google_lens",
                    "type": "visual_matches",
                    "url": image_url,
                    "api_key": SERPAPI_API_KEY,
                }
            ).get_dict()
            exact = GoogleSearch(
                {
                    "engine": "google_lens",
                    "type": "exact_matches",
                    "url": image_url,
                    "api_key": SERPAPI_API_KEY,
                }
            ).get_dict()

            visual_matches = visual.get("visual_matches", []) or []
            exact_matches = exact.get("exact_matches", []) or []
            top_sources = []
            for item in visual_matches[:5]:
                source = item.get("source")
                if source:
                    top_sources.append(str(source))

            return {
                "available": True,
                "image_url": image_url,
                "visual_match_count": len(visual_matches),
                "exact_match_count": len(exact_matches),
                "top_sources": top_sources,
            }
        except Exception as exc:
            return {
                "available": False,
                "visual_match_count": 0,
                "exact_match_count": 0,
                "top_sources": [],
                "error": str(exc),
            }

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

    def _init_clip_bundle(self) -> Optional[Dict[str, Any]]:
        if torch is None or open_clip is None:
            self.clip_init_error = "torch or open_clip is not installed."
            return None
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(
                SCAM_CLIP_MODEL_NAME,
                pretrained=SCAM_CLIP_PRETRAINED,
            )
            model.eval()
            return {
                "model": model,
                "preprocess": preprocess,
                "tokenizer": open_clip.get_tokenizer(SCAM_CLIP_MODEL_NAME),
            }
        except Exception as exc:
            self.clip_init_error = str(exc)
            return None

    @staticmethod
    def _build_summary(
        route_result: Dict[str, Any],
        semantic_result: Dict[str, Any],
        serpapi_result: Dict[str, Any],
    ) -> str:
        base = (
            f"Module A analyzed the similarity branch for a route with "
            f"{route_result['person_count']} person detections and {route_result['face_count']} faces."
        )

        if semantic_result["available"] and semantic_result["scores"]:
            top_label = max(semantic_result["scores"], key=semantic_result["scores"].get)
            base += f" OpenCLIP considered the image closest to '{top_label}'."

        if serpapi_result["available"]:
            base += (
                f" SerpAPI returned {serpapi_result['visual_match_count']} visual matches "
                f"and {serpapi_result['exact_match_count']} exact matches."
            )
        else:
            base += " SerpAPI evidence was unavailable."

        return base
