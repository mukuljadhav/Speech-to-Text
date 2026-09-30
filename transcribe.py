"""Cross-platform CPU speech-to-text using faster-whisper.

Works on macOS (Intel / Apple Silicon), Windows and Linux. No GPU or system
ffmpeg required.

Usage:
    python transcribe.py audio.mp3
    python transcribe.py audio.mp3 --output audio.srt
    python transcribe.py audio.wav --model small --language en
"""

import argparse
import os
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


def format_timestamp(seconds: float, srt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    sep = "," if srt else "."
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def write_output(path: Path, segments: list) -> None:
    if path.suffix.lower() == ".srt":
        lines = []
        for i, seg in enumerate(segments, start=1):
            lines.append(str(i))
            lines.append(
                f"{format_timestamp(seg.start, srt=True)} --> "
                f"{format_timestamp(seg.end, srt=True)}"
            )
            lines.append(seg.text.strip())
            lines.append("")
        text = "\n".join(lines)
    else:
        text = "\n".join(seg.text.strip() for seg in segments) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Transcribe audio on CPU with Whisper.")
    parser.add_argument("audio", type=Path, help="Audio/video file (mp3, wav, m4a, mp4, ...)")
    parser.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Whisper model: large-v3-turbo (default), large-v3, medium, small, base, "
        "tiny, distil-large-v3 (English only)",
    )
    parser.add_argument(
        "--language", default=None, help="Language code, e.g. en, hi, mr (default: auto-detect)"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Save transcript to a .txt or .srt file"
    )
    parser.add_argument("--beam-size", type=int, default=5, help="Beam size (lower = faster)")
    args = parser.parse_args()

    if not args.audio.is_file():
        print(f"Error: file not found: {args.audio}", file=sys.stderr)
        return 1

    print(f"Loading model '{args.model}' (CPU, int8)...", file=sys.stderr)
    model = WhisperModel(
        args.model,
        device="cpu",
        compute_type="int8",
        cpu_threads=os.cpu_count() or 4,
    )

    start = time.perf_counter()
    segments_iter, info = model.transcribe(
        str(args.audio),
        language=args.language,
        beam_size=args.beam_size,
        vad_filter=True,
    )
    print(
        f"Detected language: {info.language} (p={info.language_probability:.2f}), "
        f"duration: {info.duration:.1f}s",
        file=sys.stderr,
    )

    segments = []
    for seg in segments_iter:
        segments.append(seg)
        print(f"[{format_timestamp(seg.start)} -> {format_timestamp(seg.end)}] {seg.text.strip()}")
    elapsed = time.perf_counter() - start

    rtf = elapsed / info.duration if info.duration else 0.0
    print(f"\nTranscribed in {elapsed:.1f}s (real-time factor {rtf:.2f})", file=sys.stderr)

    if args.output:
        write_output(args.output, segments)
        print(f"Saved: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
