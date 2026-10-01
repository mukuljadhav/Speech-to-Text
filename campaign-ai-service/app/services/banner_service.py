"""
Banner Service
==============
Takes a CampaignContent object (headline, subheadline, cta, colors) and
renders it into an actual image file using Pillow, then saves it to the
output folder. This is a template-based renderer -- fully local, no extra
AI model needed, and 100% deterministic (same input -> same layout).
"""

import os
import textwrap
import uuid
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.config import settings
from app.models.schemas import CampaignContent

BASE_DIR = Path(__file__).resolve().parent.parent.parent
FONTS_DIR = BASE_DIR / "assets" / "fonts"

FONT_BOLD = FONTS_DIR / "Poppins-Bold.ttf"
FONT_SEMIBOLD = FONTS_DIR / "Poppins-SemiBold.ttf"
FONT_REGULAR = FONTS_DIR / "Poppins-Regular.ttf"


def _load_font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:
        # Fallback so the service never crashes even if a font is missing
        return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Wraps text so it never overflows the banner width."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def render_banner(content: CampaignContent) -> str:
    """
    Draws the banner and saves it to OUTPUT_DIR.
    Returns the local file path (string) of the saved image.
    """
    width, height = settings.BANNER_WIDTH, settings.BANNER_HEIGHT
    img = Image.new("RGB", (width, height), color=content.background_color)
    draw = ImageDraw.Draw(img)

    margin = int(width * 0.08)
    max_text_width = width - (2 * margin)

    # ---- Headline ----
    headline_font_size = int(width * 0.09)
    headline_font = _load_font(FONT_BOLD, headline_font_size)
    headline_lines = _wrap_text(draw, content.headline, headline_font, max_text_width)

    # ---- Subheadline ----
    sub_font_size = int(width * 0.035)
    sub_font = _load_font(FONT_REGULAR, sub_font_size)
    sub_lines = _wrap_text(draw, content.subheadline, sub_font, max_text_width)

    # ---- CTA button text ----
    cta_font_size = int(width * 0.032)
    cta_font = _load_font(FONT_SEMIBOLD, cta_font_size)

    # --- Layout: vertically center the whole text block ---
    line_spacing = int(headline_font_size * 0.15)
    headline_block_height = len(headline_lines) * (headline_font_size + line_spacing)
    sub_block_height = len(sub_lines) * (sub_font_size + int(sub_font_size * 0.3))
    cta_button_height = int(cta_font_size * 2.4)
    gap_after_headline = int(height * 0.03)
    gap_after_sub = int(height * 0.05)

    total_block_height = (
        headline_block_height + gap_after_headline + sub_block_height + gap_after_sub + cta_button_height
    )
    current_y = (height - total_block_height) // 2

    # Draw headline (centered horizontally)
    for line in headline_lines:
        line_width = draw.textlength(line, font=headline_font)
        x = (width - line_width) / 2
        draw.text((x, current_y), line, font=headline_font, fill=content.text_color)
        current_y += headline_font_size + line_spacing

    current_y += gap_after_headline

    # Draw subheadline
    for line in sub_lines:
        line_width = draw.textlength(line, font=sub_font)
        x = (width - line_width) / 2
        draw.text((x, current_y), line, font=sub_font, fill=content.text_color)
        current_y += sub_font_size + int(sub_font_size * 0.3)

    current_y += gap_after_sub

    # Draw CTA button (rounded rectangle + centered text)
    cta_text_width = draw.textlength(content.cta, font=cta_font)
    button_padding_x = int(cta_font_size * 1.4)
    button_width = cta_text_width + (2 * button_padding_x)
    button_x0 = (width - button_width) / 2
    button_y0 = current_y
    button_x1 = button_x0 + button_width
    button_y1 = button_y0 + cta_button_height

    draw.rounded_rectangle(
        [button_x0, button_y0, button_x1, button_y1],
        radius=cta_button_height / 2,
        fill=content.accent_color,
    )
    text_x = button_x0 + (button_width - cta_text_width) / 2
    text_y = button_y0 + (cta_button_height - cta_font_size) / 2 - (cta_font_size * 0.15)
    draw.text((text_x, text_y), content.cta, font=cta_font, fill=content.text_color)

    # ---- Save file ----
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
    ext = "png" if settings.BANNER_FORMAT.upper() == "PNG" else "jpg"
    filename = f"banner_{uuid.uuid4().hex[:12]}.{ext}"
    output_path = Path(settings.OUTPUT_DIR) / filename

    if ext == "jpg":
        img = img.convert("RGB")
        img.save(output_path, format="JPEG", quality=92)
    else:
        img.save(output_path, format="PNG")

    return str(output_path)
