from __future__ import annotations

from typing import Optional

from config.settings import GEMINI_API_KEY, GEMINI_MODEL_NAME


def generate_gemini_text(
    prompt: str,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Generate text with Gemini using whichever SDK is available."""
    resolved_api_key = (api_key or GEMINI_API_KEY or "").strip()
    if not resolved_api_key:
        raise RuntimeError("Gemini API key is not configured.")

    resolved_model = model_name or GEMINI_MODEL_NAME

    try:
        from google import genai

        client = genai.Client(api_key=resolved_api_key)
        response = client.models.generate_content(
            model=resolved_model,
            contents=prompt,
        )
        text = getattr(response, "text", None)
        if text:
            return text
        raise RuntimeError("Gemini response did not contain text.")
    except ImportError:
        pass

    try:
        import google.generativeai as legacy_genai

        legacy_genai.configure(api_key=resolved_api_key)
        model = legacy_genai.GenerativeModel(resolved_model)
        response = model.generate_content(prompt)
        text = getattr(response, "text", None)
        if text:
            return text
        raise RuntimeError("Gemini response did not contain text.")
    except ImportError as exc:
        raise RuntimeError("No Gemini SDK is installed.") from exc
