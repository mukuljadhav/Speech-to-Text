"""
Image Service
=============
Responsible for ONE thing: turning an uploaded image file into a
validated, reasonably-sized, base64-encoded string ready to hand to
Gemma 4 via Ollama's `images` field.

We don't do any "understanding" here -- Gemma 4 does that itself once it
receives the image. This module is purely about safe, fast preparation:
- confirm the upload is actually a readable image (not a corrupt/foreign file)
- downscale very large photos so the request to Ollama stays fast
- return base64 text, since that's the format Ollama's API expects
"""

import base64
import io
import logging

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

# Keep the longest side under this so large phone photos don't slow down
# Gemma 4 or blow up the request payload unnecessarily.
MAX_DIMENSION = 1024


class ImageServiceError(Exception):
    """Raised when the uploaded file isn't a usable image."""


def prepare_image_for_gemma(image_file: UploadFile) -> str:
    """
    Reads the uploaded image, validates it, downsizes if needed, and
    returns a base64-encoded JPEG string (no data URI prefix -- Ollama
    wants raw base64).
    Raises ImageServiceError on any failure.
    """
    try:
        raw_bytes = image_file.file.read()
    except Exception as exc:  # noqa: BLE001
        raise ImageServiceError(f"Could not read uploaded image: {exc}") from exc

    if not raw_bytes:
        raise ImageServiceError("Uploaded image file is empty.")

    try:
        img = Image.open(io.BytesIO(raw_bytes))
        img.load()  # force-read pixel data now, so corrupt files fail here, not later
    except UnidentifiedImageError as exc:
        raise ImageServiceError("Uploaded file is not a valid image.") from exc
    except Exception as exc:  # noqa: BLE001
        raise ImageServiceError(f"Failed to open uploaded image: {exc}") from exc

    # Normalize to RGB (handles PNG transparency, CMYK, etc. consistently)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Downscale if the image is larger than MAX_DIMENSION on its longest side
    if max(img.size) > MAX_DIMENSION:
        img.thumbnail((MAX_DIMENSION, MAX_DIMENSION), Image.LANCZOS)
        logger.info("Resized uploaded image to %s for Gemma 4 input", img.size)

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    logger.info("Prepared image for Gemma 4 (%d bytes base64)", len(encoded))
    return encoded