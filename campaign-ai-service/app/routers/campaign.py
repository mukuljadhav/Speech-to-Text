"""
Campaign Router
================
Single unified endpoint: POST /api/v1/campaign/generate

Accepts any combination of:
  - text        (optional form field)
  - voice       (optional audio file -- transcribed via voice_service)
  - image       (optional image file -- understood directly by Gemma 4, Phase 3)
  - brand_name  (optional form field)

Whatever combination of text/voice/image arrives is merged by
input_resolver into one final prompt (image bytes travel separately),
which is then handed to the same Gemma 4 + banner pipeline built in
Phase 1 -- unchanged.
"""

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.schemas import CampaignResponse
from app.services.banner_service import render_banner
from app.services.gemma_service import GemmaServiceError, generate_campaign_content
from app.services.image_service import ImageServiceError, prepare_image_for_gemma
from app.services.input_resolver import InputResolverError, resolve_input
from app.services.voice_service import VoiceServiceError, transcribe_audio

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/campaign", tags=["campaign"])


@router.post("/generate", response_model=CampaignResponse)
def generate_campaign(
    text: Optional[str] = Form(default=None, description="Text description of the campaign."),
    brand_name: Optional[str] = Form(default=None, description="Optional brand/shop name."),
    voice: Optional[UploadFile] = File(default=None, description="Optional voice recording (.wav, .mp3, .m4a, ...)."),
    image: Optional[UploadFile] = File(default=None, description="Optional reference image (e.g. product photo)."),
) -> CampaignResponse:
    image_base64: Optional[str] = None
    if image is not None:
        try:
            image_base64 = prepare_image_for_gemma(image)
        except ImageServiceError as exc:
            logger.error("Image preparation failed: %s", exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    voice_text: Optional[str] = None
    if voice is not None:
        try:
            voice_text = transcribe_audio(voice)
        except VoiceServiceError as exc:
            logger.error("Voice transcription failed: %s", exc)
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        resolved = resolve_input(text=text, voice_text=voice_text, has_image=image_base64 is not None)
    except InputResolverError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        content = generate_campaign_content(
            user_text=resolved.final_text,
            brand_name=brand_name,
            image_base64=image_base64,
        )
    except GemmaServiceError as exc:
        logger.error("Gemma service failed: %s", exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    try:
        banner_path = render_banner(content)
    except Exception as exc:  # noqa: BLE001 - want to convert any render failure to a clean 500
        logger.exception("Banner rendering failed")
        raise HTTPException(status_code=500, detail=f"Banner rendering failed: {exc}") from exc

    return CampaignResponse(
        status="success",
        banner_path=banner_path,
        campaign_content=content,
        source_text=resolved.source_text,
    )