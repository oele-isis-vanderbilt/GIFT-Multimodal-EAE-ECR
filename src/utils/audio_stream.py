"""Forward-only, stream-honest audio chunk sources for streaming ASR.

The drill-end detector consumes audio in forward chunks as it "arrives",
never seeking to end-of-file and never loading the whole clip at once. This
keeps the transcription pipeline compatible with a live stream (audio
arriving alongside video frames) while, for the current file-based pipeline,
a thin ffmpeg-backed shim reads incrementally at a moving offset.

All sources yield float32 mono waveforms at ``sample_rate`` (16 kHz for the
ASR models). ``read(start_sec, dur_sec)`` returns the samples in
``[start_sec, start_sec + dur_sec)`` or ``None`` when the source is
exhausted (past end-of-audio) — the "stream ended" signal.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

import numpy as np

from .audio import extract_audio_to_wav, probe_audio_duration_sec

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
# One retry after a failed extraction that is provably NOT past end-of-audio
# (a transient subprocess failure under load must not truncate transcription).
EXTRACT_RETRY_DELAY_SEC = 1.0


class AudioChunkSource:
    """Abstract forward audio reader. Implementations must not seek to EOF."""

    sample_rate: int = SAMPLE_RATE

    def read(self, start_sec: float, dur_sec: float) -> Optional[np.ndarray]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial
        pass


class FileAudioChunkSource(AudioChunkSource):
    """File-backed shim: incremental ffmpeg reads at a moving offset.

    Each ``read`` extracts only ``[start_sec, start_sec+dur_sec)`` (``-ss``/
    ``-to`` before ``-i`` for a fast, bounded seek) — never the whole file,
    never to EOF. Returns ``None`` once the requested window lies entirely
    past the available audio, which the worker treats as end-of-stream.
    """

    def __init__(self, source_path: str, sample_rate: int = SAMPLE_RATE) -> None:
        self.source_path = source_path
        self.sample_rate = int(sample_rate)
        self._tmpdir = tempfile.mkdtemp(prefix="asr_chunks_")
        # Known length of the audio stream (None = unknowable). Makes
        # "requested window past end-of-audio" decidable, so an extraction
        # FAILURE mid-stream is never mistaken for a clean EOF.
        self.audio_duration_sec = probe_audio_duration_sec(source_path)
        # Set when reads were abandoned by a mid-stream failure (not EOF) —
        # callers surface this in their artifacts (e.g. DrillWindow reason).
        self.failed_at_sec: Optional[float] = None

    def read(self, start_sec: float, dur_sec: float) -> Optional[np.ndarray]:
        start_sec = max(0.0, float(start_sec))
        dur_sec = float(dur_sec)
        if dur_sec <= 0:
            return None
        known = self.audio_duration_sec
        if known is not None and start_sec >= known - 0.05:
            return None  # genuinely past end-of-audio
        # With a known duration, a failed/near-empty extraction inside the
        # stream is a transient failure (subprocess spawn/timeout under load):
        # retry once before giving up loudly. With an unknown duration we keep
        # the legacy single attempt — near-empty output at the true end of
        # audio is normal there, and retry/noise on every natural EOF is worse.
        attempts = 2 if known is not None else 1
        for attempt in range(attempts):
            if attempt:
                time.sleep(EXTRACT_RETRY_DELAY_SEC)
            data = self._extract_once(start_sec, dur_sec)
            if data is not None and data.size >= int(0.05 * self.sample_rate):
                return data
        if known is not None:
            self.failed_at_sec = start_sec
            logger.error(
                "Audio chunk extraction failed twice at %.2fs although the "
                "audio stream runs to %.2fs — treating as end-of-stream; "
                "transcription (and drill-end detection) will be truncated.",
                start_sec, known,
            )
        return None

    def _extract_once(self, start_sec: float, dur_sec: float) -> Optional[np.ndarray]:
        out_wav = os.path.join(self._tmpdir, f"chunk_{start_sec:.3f}_{dur_sec:.3f}.wav")
        ok = extract_audio_to_wav(
            self.source_path,
            out_wav,
            start_sec=start_sec,
            end_sec=start_sec + dur_sec,
            sample_rate=self.sample_rate,
            channels=1,
        )
        if not ok:
            return None
        try:
            data = _load_wav_mono_f32(out_wav)
        finally:
            try:
                os.remove(out_wav)
            except OSError:
                pass
        return data

    def close(self) -> None:
        try:
            for f in os.listdir(self._tmpdir):
                try:
                    os.remove(os.path.join(self._tmpdir, f))
                except OSError:
                    pass
            os.rmdir(self._tmpdir)
        except OSError:
            pass


class ArrayAudioChunkSource(AudioChunkSource):
    """In-memory source over a pre-loaded waveform — for tests / live queues.

    (A live-stream implementation feeding samples from an audio queue would
    share this same ``read`` contract.)
    """

    def __init__(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
        self.samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        self.sample_rate = int(sample_rate)

    def read(self, start_sec: float, dur_sec: float) -> Optional[np.ndarray]:
        s = int(max(0.0, float(start_sec)) * self.sample_rate)
        e = int((max(0.0, float(start_sec)) + float(dur_sec)) * self.sample_rate)
        if s >= self.samples.size:
            return None
        seg = self.samples[s:min(e, self.samples.size)]
        if seg.size < int(0.05 * self.sample_rate):
            return None
        return seg


def _load_wav_mono_f32(path: str) -> Optional[np.ndarray]:
    """Load a 16-bit PCM WAV as float32 mono in [-1, 1] without extra deps."""
    try:
        import soundfile as sf  # pinned in environment.yml (WAV I/O)
        data, _ = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        return np.asarray(data, dtype=np.float32)
    except Exception:
        # Fallback: stdlib wave (PCM16 only) — avoids any hard soundfile dep.
        try:
            import wave
            with wave.open(path, "rb") as w:
                n = w.getnframes()
                raw = w.readframes(n)
                ch = w.getnchannels()
            arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if ch > 1:
                arr = arr.reshape(-1, ch).mean(axis=1)
            return arr
        except Exception:
            logger.warning("Failed to load WAV %s", path, exc_info=True)
            return None
