from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from uuid import uuid4

import cv2
import numpy as np
import requests

from utils.image_utils import save_image

DEFAULT_TIMEOUT = 20
MAX_HTML_BYTES = 1_000_000
MAX_IMAGE_BYTES = 12_000_000
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class _ImageUrlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.image_candidates: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_map = {key.lower(): value for key, value in attrs if key and value}
        tag_name = tag.lower()

        if tag_name == "meta":
            prop = attrs_map.get("property", "").lower()
            name = attrs_map.get("name", "").lower()
            if prop in {"og:image", "og:image:url"} or name in {"twitter:image", "twitter:image:src"}:
                content = attrs_map.get("content")
                if content:
                    self.image_candidates.append(content.strip())
            return

        if tag_name == "img":
            src = attrs_map.get("src")
            if src:
                self.image_candidates.append(src.strip())


def validate_http_url(raw_url: str) -> str:
    url = (raw_url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must start with http:// or https://")
    return url


def fetch_image_from_source_url(source_url: str) -> tuple[bytes, str]:
    normalized_url = validate_http_url(source_url)

    try:
        image_bytes, final_url = _fetch_single_image(normalized_url)
        return image_bytes, final_url
    except ValueError:
        pass

    response = requests.get(
        normalized_url,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        allow_redirects=True,
    )
    response.raise_for_status()
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "html" not in content_type and "<html" not in response.text[:512].lower():
        raise ValueError("Provided URL is neither an image nor a valid HTML page with image candidates.")

    html_text = response.text[:MAX_HTML_BYTES]
    parser = _ImageUrlParser()
    parser.feed(html_text)

    candidates = _normalize_candidates(parser.image_candidates, response.url)
    if not candidates:
        raise ValueError("No image was found on this URL.")

    for candidate in candidates[:12]:
        try:
            return _fetch_single_image(candidate)
        except Exception:
            continue

    raise ValueError("Unable to download a usable image from this URL.")


def save_downloaded_image(image_bytes: bytes, input_dir: Path) -> tuple[Path, str]:
    np_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Downloaded content is not a decodable image.")

    filename = f"url_{uuid4().hex[:8]}.jpg"
    save_path = input_dir / filename
    save_image(save_path, image)
    return save_path, filename


def _fetch_single_image(image_url: str) -> tuple[bytes, str]:
    response = requests.get(
        image_url,
        timeout=DEFAULT_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        stream=True,
        allow_redirects=True,
    )
    response.raise_for_status()

    content_type = (response.headers.get("Content-Type") or "").lower()
    if content_type and "image" not in content_type:
        raise ValueError("URL did not return an image content type.")

    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_content(chunk_size=8192):
        if not chunk:
            continue
        total += len(chunk)
        if total > MAX_IMAGE_BYTES:
            raise ValueError("Image file is too large.")
        chunks.append(chunk)

    image_bytes = b"".join(chunks)
    if not image_bytes:
        raise ValueError("Image content is empty.")

    decoded = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Image decoding failed.")

    return image_bytes, response.url


def _normalize_candidates(candidates: list[str], base_url: str) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()

    for candidate in candidates:
        if not candidate:
            continue
        absolute = urljoin(base_url, candidate.strip())
        parsed = urlparse(absolute)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if absolute in seen:
            continue
        seen.add(absolute)
        unique.append(absolute)

    return unique
