from __future__ import annotations

from typing import Any, Dict, List


class RiskAggregator:
    """Combine router and module evidence into the final project verdict."""

    def aggregate(
        self,
        router_result: Dict[str, Any],
        module_a_result: Dict[str, Any],
        module_c_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        risk_score = 0.0
        evidence: List[str] = []
        limitations: List[str] = []

        if router_result["routing_flags"]["run_deepfake_branch"]:
            risk_score += 0.8
            limitations.append("Deepfake branch is defined in the architecture but not implemented in this version.")
            evidence.append("Router detected a person with a face, which would normally trigger deepfake inspection.")

        if module_a_result.get("semantic_available") and module_a_result.get("semantic_scores"):
            top_label = max(module_a_result["semantic_scores"], key=module_a_result["semantic_scores"].get)
            top_score = float(module_a_result["semantic_scores"][top_label])
            if any(keyword in top_label for keyword in ["scam", "financial", "social media"]):
                risk_score += min(1.4, top_score * 1.5)
                evidence.append(f"Module A semantic label '{top_label}' increased risk.")

        serpapi_a = module_a_result.get("serpapi_summary", {})
        if serpapi_a.get("available") and serpapi_a.get("exact_match_count", 0) == 0 and serpapi_a.get("visual_match_count", 0) == 0:
            risk_score += 0.35
            evidence.append("Module A found no strong external image matches.")

        suspicious_keywords = module_c_result.get("suspicious_keywords", [])
        risk_score += min(1.8, len(suspicious_keywords) * 0.45)
        if suspicious_keywords:
            evidence.append(f"Module C OCR found suspicious keywords: {', '.join(suspicious_keywords[:5])}.")

        if router_result["num_faces"] > 0:
            risk_score += min(0.5, router_result["num_faces"] * 0.15)
        if router_result["num_detections"] > 0:
            risk_score += min(0.4, router_result["num_detections"] * 0.08)

        serpapi_c = module_c_result.get("serpapi_summary", {})
        if serpapi_c.get("available") and serpapi_c.get("exact_match_count", 0) == 0:
            risk_score += 0.25
            evidence.append("Module C found no exact-match source support from SerpAPI.")

        if not module_c_result.get("ocr_available"):
            limitations.append("OCR evidence is currently unavailable or degraded.")
        if not module_a_result.get("serpapi_available"):
            limitations.append("Module A external search evidence is unavailable.")
        if not module_c_result.get("serpapi_available"):
            limitations.append("Module C external source lookup is unavailable.")

        risk_score_percent = min(100.0, round((risk_score / 4.0) * 100.0, 2))
        if risk_score_percent >= 70:
            risk_level = "high"
            authenticity_label = "Highly Suspicious"
        elif risk_score_percent >= 35:
            risk_level = "medium"
            authenticity_label = "Needs Human Review"
        else:
            risk_level = "low"
            authenticity_label = "Likely Benign"

        if not evidence:
            evidence.append("Current evidence is weak; the image should be interpreted with human review.")

        report = (
            f"Final decision: {authenticity_label}. "
            f"The aggregated risk level is {risk_level.upper()} with score {risk_score_percent:.0f}/100. "
            f"Key evidence: {' '.join(evidence[:3])}"
        )

        return {
            "risk_level": risk_level,
            "risk_score": round(risk_score, 4),
            "risk_score_percent": risk_score_percent,
            "authenticity_verdict": authenticity_label == "Likely Benign",
            "authenticity_label": authenticity_label,
            "evidence": evidence,
            "limitations": limitations,
            "report": report,
            "headline": f"{authenticity_label} | risk {risk_score_percent:.0f}/100",
        }
