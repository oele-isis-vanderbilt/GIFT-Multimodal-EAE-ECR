"""Streaming drill-end locator: forward-chunked transcription that stops the
moment the drill end is confirmed.

Ties together an :class:`AudioChunkSource` (forward, never-to-EOF), a load-once
:class:`TranscriptionSession`, an optional streaming :class:`DenoiseSession`,
and the incremental :class:`DrillEndDetector`. The engine calls
:func:`locate_drill_end_streaming` from its background worker; the same
function is exercised directly in tests.

Approach: keep a growing denoised buffer from ``audio_start``; each step read
the next forward chunk, (optionally) denoise it, append, re-transcribe the
buffer, and feed the detector. Fire on the first LocalAgreement-confirmed
"room clear" — so transcription stops ~one confirmation chunk past the true
end rather than running to EOF. If the source is exhausted first, fall back to
the batch "latest-passing" rule over everything seen (or ``end_uncertain``).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .audio_stream import AudioChunkSource, FileAudioChunkSource, SAMPLE_RATE
from .drill_window import DrillEndDetector, DrillWindow

logger = logging.getLogger(__name__)


@dataclass
class StreamingResult:
    window: DrillWindow
    segments: List[Dict[str, Any]] = field(default_factory=list)
    last_audio_sec: float = 0.0        # how far into the audio we transcribed
    passes: int = 0                    # number of transcription passes
    elapsed_sec: float = 0.0
    audio_start_sec: float = 0.0       # timeline offset of ``audio`` / segments
    audio: Optional[np.ndarray] = None  # denoised buffer (only when denoising)


def locate_drill_end_streaming(
    *,
    source_path: str,
    drill_start_frame: int,
    fps: float,
    frame_total: int,
    config: Optional[Dict[str, Any]],
    session,                       # TranscriptionSession (already constructed)
    denoise_session=None,          # optional DenoiseSession
    audio_source: Optional[AudioChunkSource] = None,
    clock=time.monotonic,
) -> StreamingResult:
    config = config or {}
    fps = float(fps or config.get("frame_rate") or 30.0)
    preroll = float(config.get("transcription_preroll_sec", 5.0) or 0.0)
    chunk_sec = float(config.get("asr_stream_chunk_sec", 8.0) or 8.0)
    confirm_sec = float(config.get("asr_stream_confirm_sec", 3.0) or 3.0)

    drill_start_sec = (max(1, int(drill_start_frame)) - 1) / fps if fps > 0 else 0.0
    audio_start = max(0.0, drill_start_sec - max(0.0, preroll))

    src = audio_source or FileAudioChunkSource(source_path, sample_rate=SAMPLE_RATE)
    detector = DrillEndDetector(
        drill_start_frame=drill_start_frame, fps=fps,
        total_frames=frame_total, config=config,
    )

    t0 = clock()
    buffer: Optional[np.ndarray] = None
    pos = audio_start
    window: Optional[DrillWindow] = None
    passes = 0
    try:
        while True:
            dur = confirm_sec if detector.has_pending else chunk_sec
            chunk = src.read(pos, dur)
            if chunk is None:                     # source exhausted → EOF
                break
            if denoise_session is not None:
                try:
                    chunk = denoise_session.feed(chunk)
                except Exception:
                    logger.warning("denoise chunk failed; using raw audio.", exc_info=True)
            if chunk is None or chunk.size == 0:
                pos += dur
                continue
            buffer = chunk if buffer is None else np.concatenate([buffer, chunk])
            pos += dur
            segments = session.transcribe_audio(buffer, offset_sec=audio_start)
            passes += 1
            window = detector.update(segments)
            if window is not None:
                break
    finally:
        if audio_source is None:
            src.close()
        if denoise_session is not None:
            # Drain the streamer's retained tail so the denoised buffer (kept
            # as the ``_denoised.wav`` artifact) covers the full span read.
            try:
                tail = denoise_session.flush()
                if tail is not None and getattr(tail, "size", 0) > 0:
                    buffer = tail if buffer is None else np.concatenate([buffer, tail])
            except Exception:
                pass

    if window is None:
        window = detector.finalize()
        # A mid-stream extraction failure (not a real EOF) truncated the
        # transcript — say so in the sidecar so a wrong/missing drill end is
        # diagnosable from the artifacts alone.
        fail_at = getattr(src, "failed_at_sec", None)
        if fail_at is not None:
            total = getattr(src, "audio_duration_sec", None)
            window.decision_reason += (
                f"; audio_read_failed_at_{fail_at:.1f}s"
                + (f"_of_{total:.1f}s_audio" if total else "")
            )

    return StreamingResult(
        window=window,
        segments=detector._segments,          # last full pass = accumulated transcript
        last_audio_sec=pos,
        passes=passes,
        elapsed_sec=clock() - t0,
        audio_start_sec=audio_start,
        audio=buffer if denoise_session is not None else None,
    )
