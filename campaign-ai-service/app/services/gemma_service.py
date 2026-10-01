"""
Gemma 4 Service
================
Responsible for ONE thing: turning raw input text into a structured
CampaignContent object (headline, subheadline, CTA, colors) by prompting
Gemma 4, which is served locally through Ollama.

Ollama exposes a simple REST API on http://localhost:11434 once you've
run `ollama pull gemma4:e2b`. We call the /api/chat endpoint and ask
Ollama to force valid JSON output (format="json"), which makes parsing
Gemma's response reliable.
"""

import json
import logging
import re

import requests

from app.config import settings
from app.models.schemas import CampaignContent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a senior marketing copywriter and brand designer.
Given a description of a product, offer, or campaign, produce a concise,
punchy marketing banner brief.

You may sometimes also receive a reference image (e.g. a product photo or
mood board). When an image is provided, look at it carefully and let its
subject, colors, and style inform your headline, tone, and color choices.

You MUST respond with ONLY a valid JSON object (no markdown, no commentary)
with EXACTLY these keys:
{
  "headline": "short punchy headline, max 6 words",
  "subheadline": "supporting line, max 12 words",
  "cta": "call to action button text, max 3 words, e.g. 'Shop Now'",
  "background_color": "a hex color code like #1A2B3C that fits the mood",
  "accent_color": "a hex color code for the CTA button/highlights, must contrast the background",
  "text_color": "a hex color code for text, must be readable on the background (usually #FFFFFF or #111111)"
}

Rules:
- Colors must be valid 6-digit hex codes starting with #.
- text_color must have strong contrast against background_color.
- Keep language energetic and marketing-appropriate.
- Do not include any text outside the JSON object.
"""


class GemmaServiceError(Exception):
    """Raised when Gemma 4 / Ollama cannot produce usable campaign content."""


def _extract_json(raw_text: str) -> dict:
    """
    Gemma sometimes wraps JSON in ```json ... ``` fences even when asked
    not to. This strips fences and extracts the first {...} block found.
    """
    text = raw_text.strip()
    text = re.sub(r"^```(json)?", "", text.strip(), flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    # Fallback: grab the first {...} block in case of stray text around it
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise GemmaServiceError(f"No JSON object found in Gemma response: {raw_text[:300]}")

    return json.loads(match.group(0))


def _validate_hex_color(value: str, field_name: str) -> str:
    if not re.match(r"^#([A-Fa-f0-9]{6})$", value.strip()):
        raise GemmaServiceError(f"Invalid hex color for {field_name}: {value}")
    return value.strip()


def generate_campaign_content(
    user_text: str,
    brand_name: str | None = None,
    image_base64: str | None = None,
) -> CampaignContent:
    """
    Calls Gemma 4 (via Ollama) with the user's text (and optionally a
    reference image) and returns a validated CampaignContent object.
    Raises GemmaServiceError on any failure so the API layer can turn it
    into a clean HTTP error response.
    """
    prompt = user_text.strip()
    if brand_name:
        prompt = f"Brand name: {brand_name}\nCampaign description: {prompt}"

    user_message: dict = {"role": "user", "content": prompt}
    if image_base64:
        # Ollama's chat API accepts a list of base64-encoded images per message
        user_message["images"] = [image_base64]

    payload = {
        "model": settings.GEMMA_MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            user_message,
        ],
        "format": "json",  # Ask Ollama to force valid JSON output
        "stream": False,
        "options": {"temperature": 0.7},
    }

    try:
        response = requests.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=180 if image_base64 else 120,  # images take a bit longer to process
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError as exc:
        raise GemmaServiceError(
            "Could not connect to Ollama. Is it running? Try `ollama serve` "
            "in a terminal, and make sure you've run "
            f"`ollama pull {settings.GEMMA_MODEL_NAME}`."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise GemmaServiceError(f"Ollama request failed: {exc}") from exc

    data = response.json()
    raw_content = data.get("message", {}).get("content", "")
    if not raw_content:
        raise GemmaServiceError(f"Empty response from Gemma 4. Full payload: {data}")

    try:
        parsed = _extract_json(raw_content)
    except (json.JSONDecodeError, GemmaServiceError) as exc:
        raise GemmaServiceError(f"Gemma 4 did not return valid JSON: {exc}") from exc

    required_keys = {"headline", "subheadline", "cta", "background_color", "accent_color", "text_color"}
    missing = required_keys - parsed.keys()
    if missing:
        raise GemmaServiceError(f"Gemma 4 response missing keys: {missing}. Got: {parsed}")

    for color_field in ("background_color", "accent_color", "text_color"):
        parsed[color_field] = _validate_hex_color(parsed[color_field], color_field)

    return CampaignContent(**parsed)