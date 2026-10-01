"""
Input Resolver
==============
Central place that decides the final text prompt handed to Gemma 4,
regardless of which combination of inputs (text / voice / image) arrived
in the request.

Phase 2: knows about `text` and `voice_text` (already transcribed).
Phase 3: adds `has_image` -- the image bytes themselves are sent straight
to Gemma 4 by gemma_service, not routed through here, but the resolver
needs to know an image exists so it (a) doesn't demand text/voice when an
image alone is enough, and (b) can note the image's presence for the
caller in `source_text`.
Phase 4 will just combine all three -- this signature already supports it.
"""

from dataclasses import dataclass

DEFAULT_IMAGE_ONLY_PROMPT = (
    "Analyze the attached reference image and create a marketing campaign "
    "banner brief inspired by it."
)


class InputResolverError(Exception):
    """Raised when there isn't enough input to build a prompt."""


@dataclass
class ResolvedInput:
    final_text: str          # the text that will actually be sent to Gemma 4
    source_text: str         # human-readable record of what contributed to it


def resolve_input(
    text: str | None = None,
    voice_text: str | None = None,
    has_image: bool = False,
) -> ResolvedInput:
    """
    Combines whichever inputs are present into a single prompt string.

    - text and/or voice provided -> merged as before (Phase 2 behavior).
    - Only an image provided (no text, no voice) -> a sensible default
      instruction is used, since Gemma still needs *some* user text
      alongside the image.
    - Any combination with an image present gets a short note appended
      so Gemma is explicitly reminded to look at the attached image.
    - Nothing at all provided -> raises InputResolverError.
    """
    text = text.strip() if text else ""
    voice_text = voice_text.strip() if voice_text else ""

    if text and voice_text:
        final_text = f"Typed instructions: {text}\nVoice instructions: {voice_text}"
        source_label = "text + voice"
    elif text:
        final_text = text
        source_label = "text"
    elif voice_text:
        final_text = voice_text
        source_label = "voice"
    elif has_image:
        final_text = DEFAULT_IMAGE_ONLY_PROMPT
        source_label = "image only"
    else:
        raise InputResolverError(
            "No usable input provided. Please provide at least text, a voice recording, or an image."
        )

    if has_image and source_label != "image only":
        final_text += "\nAn image has also been attached -- use it as visual reference for the design."
        source_label += " + image"

    source_text = f"[{source_label}] {final_text}"
    return ResolvedInput(final_text=final_text, source_text=source_text)