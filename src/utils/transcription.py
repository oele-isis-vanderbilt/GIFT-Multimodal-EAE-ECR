"""Transcription sidecar schema + writer.

Speech-to-text is produced by the streaming Parakeet path
(:mod:`src.utils.asr_parakeet` via :mod:`src.utils.drill_stream`), which emits
segments already in the canonical schema:

    {"start": float, "end": float, "text": str,
     "words": [{"word": str, "start": float, "end": float, "score": float|None}]}

This module owns the ``{basename}_Transcription.json`` on-disk schema and the
helper that writes it, so every producer yields an identical sidecar.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

SCHEMA_VERSION = "1.2"


def save_transcription_sidecar(
    *,
    output_dir: str,
    video_basename: str,
    segments,
    model: str,
    language: Optional[str] = "en",
    aligned: bool = True,
    audio_window: Optional[Dict[str, Any]] = None,
    denoise: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Write ``{basename}_Transcription.json`` in the standard schema."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_video_basename": video_basename,
        "language": language,
        "model": model,
        "compute_type": None,
        "aligned": bool(aligned),
        "audio_window": audio_window,
        "denoise": denoise,
        "segments": segments or [],
    }
    out_path = os.path.join(output_dir, f"{video_basename}_Transcription.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logging.info("Saved transcription to %s (%d segments)", out_path, len(payload["segments"]))
        return out_path
    except OSError:
        logging.exception("Failed to write transcription JSON to %s", out_path)
        return None
