"""
Voice Service
=============
Responsible for ONE thing: turning an uploaded audio file into plain text,
using faster-whisper running fully locally (no cloud, no API key).

The Whisper model is loaded once and reused across requests (loading it
fresh every request would be slow), so we use a simple lazy-singleton
pattern here.
"""

import logging
import os
import uuid
from pathlib import Path

from faster_whisper import WhisperModel
from fastapi import UploadFile

from app.config import settings

logger = logging.getLogger(__name__)

_model: WhisperModel | None = None


class VoiceServiceError(Exception):
    """Raised when audio cannot be saved or transcribed."""


def _get_model() -> WhisperModel:
    """Loads the Whisper model once and caches it for reuse."""
    global _model
    if _model is None:
        logger.info("Loading Whisper model '%s' (first request only)...", settings.WHISPER_MODEL_SIZE)
        # compute_type="int8" keeps this fast and light enough for CPU-only laptops
        _model = WhisperModel(settings.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _model


def transcribe_audio(voice_file: UploadFile) -> str:
    """
    Saves the uploaded audio to a temp file, transcribes it, deletes the
    temp file, and returns the transcribed text (stripped, plain string).
    Raises VoiceServiceError on any failure.
    """
    os.makedirs(settings.TEMP_AUDIO_DIR, exist_ok=True)

    suffix = Path(voice_file.filename or "audio.wav").suffix or ".wav"
    temp_path = Path(settings.TEMP_AUDIO_DIR) / f"voice_{uuid.uuid4().hex[:12]}{suffix}"

    try:
        with open(temp_path, "wb") as f:
            f.write(voice_file.file.read())
    except Exception as exc:  # noqa: BLE001
        raise VoiceServiceError(f"Could not save uploaded audio: {exc}") from exc

    try:
        model = _get_model()
        segments, info = model.transcribe(str(temp_path), beam_size=5)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:  # noqa: BLE001
        raise VoiceServiceError(f"Transcription failed: {exc}") from exc
    finally:
        # Always clean up the temp audio file, even if transcription failed
        try:
            os.remove(temp_path)
        except OSError:
            pass

    if not transcript:
        raise VoiceServiceError("Transcription produced empty text. Try a clearer recording.")

    logger.info("Transcribed voice input: %s", transcript)
    return transcript