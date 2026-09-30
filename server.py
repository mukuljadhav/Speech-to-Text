"""REST API for CPU speech-to-text with automatic routing by spoken language.

- English (detected by a small classifier on VoxLingua107 features trained on Indian-accented
  English, with Whisper's language ID as fallback when unsure): Whisper large-v3-turbo, as is.
- Indian languages: AI4Bharat IndicConformer speech recognition, then IndicTrans2
  translation to English (see indic.py).
- Other languages (Whisper detects them, or an explicit language hint): Whisper transcription.

Run:
    python server.py
Then open http://127.0.0.1:8000 in a browser for the voice chat UI.

Environment variables:
    WHISPER_MODEL      Whisper model for English (default: large-v3-turbo)
    INDIC              1 = enable the Indian-language pipeline (default), 0 = Whisper only
    INDIC_DECODING     rnnt (default, slightly more accurate) or ctc
    AUTO_LANGUAGES     languages to auto-detect (default: en,hi,mr,bn,gu,pa,ta,te,kn,ml,or,as,ne); rarely
                       needed, since the server learns which languages you speak (language memory)
    LANGUAGE_MEMORY    where that memory is saved (default: ~/.cache/speech-to-text/language_memory.json)
    HOST               bind address (default: 127.0.0.1)
    PORT               port (default: 8000)
    MAX_UPLOAD_MB      max upload size in MB (default: 25)
    MAX_AUDIO_SECONDS  max audio length in seconds (default: 1800)
    MAX_CONCURRENT     transcriptions running at once (default: 2)
    ALLOWED_HOSTS      comma-separated extra Host names to accept, e.g. a LAN IP (* = any);
                       127.0.0.1 and localhost are always allowed
    CORS_ORIGINS       comma-separated extra origins allowed to call the API from a browser,
                       e.g. http://localhost:3000, null (index.html opened from disk), or * (any)
"""

import gc
import hashlib
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import av
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from faster_whisper import WhisperModel
from faster_whisper.vad import VadOptions, collect_chunks, get_speech_timestamps
from starlette.middleware.trustedhost import TrustedHostMiddleware


def env_list(name: str, default: str = "") -> list:
    return [v.strip() for v in os.environ.get(name, default).split(",") if v.strip()]


MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
INDIC_ENABLED = os.environ.get("INDIC", "1") != "0"
INDIC_DECODING = os.environ.get("INDIC_DECODING", "rnnt")
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_AUDIO_SECONDS = int(os.environ.get("MAX_AUDIO_SECONDS", "1800"))
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "2"))
ALLOWED_HOSTS = ["127.0.0.1", "localhost"] + env_list("ALLOWED_HOSTS")
CORS_ORIGINS = env_list("CORS_ORIGINS")
SAMPLE_RATE = 16000
BASE_DIR = Path(__file__).resolve().parent

state = {}
transcribe_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


class CachedWhisperModel(WhisperModel):
    """Remembers the last encoder output per thread. Language detection and transcription then
    share the encoder pass for the first 30 s window instead of running it twice (~2 s on CPU)."""

    _cache = threading.local()

    def encode(self, features: np.ndarray):
        key = (features.shape, hashlib.blake2b(np.ascontiguousarray(features).tobytes(), digest_size=16).digest())
        last = getattr(self._cache, "last", None)
        if last is not None and last[0] == key:
            return last[1]
        output = super().encode(features)
        self._cache.last = (key, output)
        return output


def whisper_language_probs(model: WhisperModel, samples: np.ndarray) -> dict:
    """Whisper language ID on exactly the features transcribe(vad_filter=True) will decode
    (same VAD, same features, last frame dropped), so CachedWhisperModel reuses the encoder pass."""
    speech = get_speech_timestamps(samples, VadOptions())
    audio = np.concatenate(collect_chunks(samples, speech)[0]) if speech else samples
    features = model.feature_extractor(audio)
    return dict(model.detect_language(features=features[..., :-1])[2])


def load_model() -> None:
    # Runs in a background thread so the server (and UI) is reachable while models
    # download on first run, and Ctrl+C still works.
    print(f"Loading Whisper '{MODEL_NAME}' (CPU, int8)... first run downloads ~1.6 GB", flush=True)
    try:
        state["model"] = CachedWhisperModel(
            MODEL_NAME,
            device="cpu",
            compute_type="int8",
            cpu_threads=os.cpu_count() or 4,
        )
        print("Whisper ready.", flush=True)
    except Exception as e:
        state["error"] = f"Model failed to load: {e}"
        print(state["error"], flush=True)
        return

    if INDIC_ENABLED:
        print("Loading Indian-language models... first run downloads ~3.7 GB", flush=True)
        try:
            import indic  # imported here so INDIC=0 works without torch installed

            state["indic"] = indic.IndicPipeline(INDIC_DECODING, os.environ.get("AUTO_LANGUAGES", indic.DEFAULT_AUTO_LANGUAGES))
            print("Indian-language models ready.", flush=True)
        except ImportError as e:
            state["indic_error"] = (f"Indian-language pipeline not installed ({e}). "
                                    "On Intel Macs it is not available; run with INDIC=0.")
            print(state["indic_error"], flush=True)
        except Exception as e:
            # English keeps working. Indian codes Whisper also knows (hi, mr, ta...) fall back to
            # Whisper transcription without translation; the others (kok, brx...) return 400.
            state["indic_error"] = f"Indian-language models failed to load: {e}"
            print(state["indic_error"], flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=load_model, daemon=True).start()
    shown = "127.0.0.1" if HOST in ("0.0.0.0", "::", "") else HOST
    print(f"Voice chat UI: http://{shown}:{PORT}", flush=True)
    yield
    state.clear()


def get_model() -> WhisperModel:
    model = state.get("model")
    if model is None:
        detail = state.get("error", "Model is still loading, try again shortly")
        raise HTTPException(503, detail, headers={"Retry-After": "5"})
    return model


class AudioTooLong(Exception):
    pass


def load_audio(data: bytes) -> np.ndarray:
    """Decode to 16 kHz mono float32, like faster_whisper.audio.decode_audio, except that it:
    - stops once the audio exceeds MAX_AUDIO_SECONDS (a small file can hold hours of silence)
    - disables FFmpeg network protocols, so uploads like SDP or playlists can't open sockets
    """
    max_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    parts, total = [], 0

    def keep(frames):
        nonlocal total
        for frame in frames:
            array = frame.to_ndarray().reshape(-1)
            total += array.size
            if total > max_samples:
                raise AudioTooLong()
            parts.append(array)

    try:
        with av.open(
            io.BytesIO(data),
            mode="r",
            metadata_errors="ignore",
            options={"protocol_whitelist": "none"},
        ) as container:
            frames = container.decode(audio=0)
            while True:
                try:
                    frame = next(frames)
                except StopIteration:
                    break
                except av.error.InvalidDataError:
                    continue  # skip bad frames, as faster-whisper does
                frame.pts = None
                keep(resampler.resample(frame))
            keep(resampler.resample(None))  # flush
    finally:
        # Frees resampler memory, see https://github.com/SYSTRAN/faster-whisper/issues/390
        del resampler
        gc.collect()

    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32) / 32768.0


app = FastAPI(title="Speech-to-Text API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def guard(request: Request, call_next):
    # A multipart POST is a CORS "simple request", so browsers send it from any
    # website without asking. Reject POSTs from other origins before doing any work.
    origin = request.headers.get("origin")
    host = request.headers.get("host", "")
    if (
        request.method == "POST"
        and origin is not None
        and origin not in (f"http://{host}", f"https://{host}")
        and origin not in CORS_ORIGINS
        and "*" not in CORS_ORIGINS
    ):
        return JSONResponse({"detail": "Cross-origin request blocked"}, status_code=403)

    # Starlette writes uploads to a temp file with no size limit, so check the
    # declared size before the body is read. The endpoint re-checks the real size.
    # Browsers, curl -F and requests always send Content-Length; chunked uploads could
    # stream an unlimited body, so they are refused.
    length = request.headers.get("content-length", "")
    if request.method == "POST" and not length.isdigit():
        return JSONResponse({"detail": "Content-Length header required"}, status_code=411)
    if length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 64 * 1024:  # + form overhead
        return JSONResponse({"detail": f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)"}, status_code=413)

    return await call_next(request)


# Middleware added last runs first: Host check, then CORS, then guard.
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
# Blocks DNS-rebinding attacks against the local server.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)


# The light endpoints are async so they never wait behind transcriptions in the thread pool.
@app.get("/", include_in_schema=False)
async def index():
    return HTMLResponse((BASE_DIR / "index.html").read_text(encoding="utf-8"))


def get_indic():
    """The Indian-language pipeline, or None if disabled or failed to load (English still works)."""
    if not INDIC_ENABLED or "indic_error" in state:
        return None
    pipe = state.get("indic")
    if pipe is None:
        raise HTTPException(503, "Indian-language models are still loading, try again shortly", headers={"Retry-After": "5"})
    return pipe


@app.get("/api/health")
async def health():
    whisper = "error" if "error" in state else "ok" if "model" in state else "loading"
    indic = ("disabled" if not INDIC_ENABLED else "error" if "indic_error" in state
             else "ok" if "indic" in state else "loading")
    status = "error" if whisper == "error" else "loading" if "loading" in (whisper, indic) else "ok"
    result = {"status": status, "model": MODEL_NAME, "components": {"whisper": whisper, "indic": indic}}
    if "indic" in state:
        memory = state["indic"].memory
        result["learned_languages"] = sorted(memory.active())  # used as a preference (~2+ confident uses)
        result["language_memory"] = memory.snapshot()  # progress: how much each language has been heard
    detail = state.get("error") or state.get("indic_error")
    if detail:
        result["detail"] = detail
    return result


@app.get("/api/languages")
async def languages():
    model = get_model()
    result = [{"code": "en", "name": "English", "auto": True, "translated": False}]
    pipe = state.get("indic")
    if pipe is not None:
        import indic

        auto = set(pipe.candidates())
        for code, name in sorted(indic.LANGUAGE_NAMES.items(), key=lambda kv: kv[1]):
            result.append({"code": code, "name": name, "auto": code in auto, "translated": True})
    # Any other Whisper language can be requested explicitly; it is transcribed, not translated.
    return {"languages": result, "whisper_languages": model.supported_languages}


# Declared as a plain `def` so FastAPI runs the CPU-heavy work in a worker thread
# instead of blocking the event loop.
@app.post("/api/transcribe")
def transcribe(
    audio: UploadFile = File(..., description="Audio file (webm, ogg, mp4/m4a, wav, mp3, ...)"),
    language: Optional[str] = Form(None, description="Spoken language code, e.g. en, hi, mr. Empty = auto-detect"),
    output: str = Form("english", description="english = translate Indian languages to English; original = no translation"),
    beam_size: int = Form(5, ge=1, le=10, description="Whisper beam size (English and other languages)"),
):
    model = get_model()
    pipe = get_indic()

    language = (language or "").strip().lower() or None
    if language == "auto":
        language = None
    output = output.strip().lower()
    if output not in ("english", "original"):
        raise HTTPException(400, "output must be 'english' or 'original'")
    indic_codes = set(pipe.asr.masks) if pipe else set()
    if language and language not in indic_codes and language not in model.supported_languages:
        raise HTTPException(400, f"Unsupported language code: {language}")

    # Limit how many requests decode and transcribe at once, to bound CPU and memory.
    if not transcribe_slots.acquire(timeout=300):
        raise HTTPException(503, "Server busy, try again shortly", headers={"Retry-After": "10"})
    try:
        return run_transcription(model, pipe, audio, language, output, beam_size)
    finally:
        transcribe_slots.release()


def run_transcription(model: WhisperModel, pipe, audio: UploadFile, language: Optional[str],
                      output: str, beam_size: int) -> dict:
    data = audio.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)")
    if not data:
        raise HTTPException(400, "Empty audio file")

    try:
        samples = load_audio(data)
    except AudioTooLong:
        raise HTTPException(413, f"Audio too long (max {MAX_AUDIO_SECONDS} seconds)")
    except Exception:
        raise HTTPException(400, "Could not decode audio. Send a valid audio file.")
    if samples.size == 0:
        raise HTTPException(400, "Audio contains no samples")

    start = time.perf_counter()
    duration = samples.size / SAMPLE_RATE

    def result(text, transcript, lang, probability, translated, engine, segments):
        names = {"en": "English"}
        if pipe is not None:
            import indic

            names.update(indic.LANGUAGE_NAMES)
        elapsed = time.perf_counter() - start
        # One line per request in the server console, to see how each utterance was routed.
        print(f"[{engine or 'no speech'}] {names.get(lang, lang) or '-'}"
              f"{' -> English' if translated else ''} in {elapsed:.2f}s: {text[:60]!r}", flush=True)
        return {
            "text": text,  # final text: English when translated, otherwise the transcript
            "transcript": transcript,  # what was said, in the spoken language's script
            "language": lang,
            "language_name": names.get(lang, lang),
            "language_probability": None if probability is None else round(probability, 3),
            "translated": translated,
            "engine": engine,
            "duration": round(duration, 2),
            "processing_time": round(time.perf_counter() - start, 2),
            "segments": segments,
        }

    # No speech at all: skip the models, whose language detection would only guess on silence.
    if not get_speech_timestamps(samples, VadOptions()):
        return result("", "", language, 1.0 if language else 0.0, False, None, [])

    # Route: English -> Whisper; Indian languages -> IndicConformer + IndicTrans2; other -> Whisper.
    engine, probability, vox, chunks, whisper_probs = "whisper", (1.0 if language else None), {}, None, None
    if pipe is not None and (language is None or language in pipe.asr.masks):
        import indic

        chunks = indic.speech_chunks(samples)
        if not chunks:
            return result("", "", language, 1.0 if language else 0.0, False, None, [])
        if language in pipe.asr.masks:
            engine = "indic"
        else:
            vox, p_english = pipe.quick_check(samples, chunks)
            if p_english >= indic.ENGLISH_SURE and pipe.plausible(vox):
                language, probability = "en", p_english
            elif p_english <= indic.NOT_ENGLISH_SURE and pipe.plausible(vox):
                engine, language = "indic", "auto"
            else:
                # Unsure (or maybe neither English nor Indian): Whisper's language ID decides. It costs
                # one encoder pass, which transcription then reuses if the speech is English.
                whisper_probs = whisper_language_probs(model, samples)
                top = max(whisper_probs, key=whisper_probs.get)
                if pipe.whisper_says_english(whisper_probs):
                    language, probability = "en", whisper_probs["en"]
                elif top not in pipe.asr.masks and top != "en" and not pipe.plausible(vox):
                    language, probability = top, whisper_probs[top]  # e.g. Chinese: plain Whisper
                else:
                    engine, language = "indic", "auto"

    if engine == "indic":
        translate = output == "english"
        res = pipe.transcribe(samples, chunks, language, vox, translate=translate)
        if res["english_like"]:
            # English that the Indian-language model wrote out phonetically: redo it as English.
            engine, language, probability = "whisper", "en", None
        elif vox and not res["segments"]:
            # Auto-detected, but the Indian-language model heard no words at all. That's typical for
            # English (and other languages), rare for Indian speech: let Whisper handle it.
            whisper_probs = whisper_probs or whisper_language_probs(model, samples)
            engine, probability = "whisper", None
            language = "en" if pipe.whisper_says_english(whisper_probs) else None
        else:
            if vox:  # auto-detected: VoxLingua's share for the chosen language, if it knows it
                probability = pipe.share(vox, res["language"])
            segments = res["segments"]
            return result(
                " ".join(s["text"] for s in segments),
                " ".join(s["transcript"] for s in segments),
                res["language"], probability, translate, "indic", segments,
            )

    segments_iter, info = model.transcribe(samples, language=language, beam_size=beam_size, vad_filter=True)
    # Segments are a lazy generator: consume them here so decoding happens now.
    raw = list(segments_iter)
    # Whisper puts the spacing in each segment's text (none for Chinese/Japanese).
    text = "".join(s.text for s in raw).strip()
    return result(
        text, text, info.language,
        probability if probability is not None else info.language_probability,
        False, "whisper",
        [
            {
                "start": round(min(s.start, duration), 2),
                "end": round(min(s.end, duration), 2),
                "text": s.text.strip(),
                "transcript": s.text.strip(),
            }
            for s in raw
        ],
    )


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
