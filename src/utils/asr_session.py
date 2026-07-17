"""Load-once, engine-agnostic ASR sessions for streaming drill-end detection.

A :class:`TranscriptionSession` loads its model(s) **once** and exposes
``transcribe_audio(np_16k_mono, offset_sec)`` returning segments in the
canonical schema used by ``drill_window`` — ``{start, end, text, words:[{word,
start, end, score}]}`` with timestamps already offset to the original-video
timeline. No per-call model reload, no file I/O.

Two implementations:

* :class:`WhisperXSession` — wraps the existing WhisperX + wav2vec2 stack
  (transitional: kept only to validate the streaming machinery against the
  current batch pipeline, then removed once Parakeet is signed off).
* :class:`ParakeetSession` — NeMo Parakeet (added in Phase 2).

``build_session(backend, config)`` picks one by name.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class TranscriptionSession:
    """Abstract load-once ASR session."""

    def load(self) -> None:
        raise NotImplementedError

    def transcribe_audio(
        self, audio: np.ndarray, offset_sec: float = 0.0
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def _apply_offset(segments: List[Dict[str, Any]], offset_sec: float) -> List[Dict[str, Any]]:
    if not offset_sec:
        return segments
    for seg in segments:
        for k in ("start", "end"):
            if isinstance(seg.get(k), (int, float)):
                seg[k] = float(seg[k]) + offset_sec
        for w in seg.get("words", []) or []:
            for k in ("start", "end"):
                if isinstance(w.get(k), (int, float)):
                    w[k] = float(w[k]) + offset_sec
    return segments


class WhisperXSession(TranscriptionSession):
    """WhisperX ASR + wav2vec2 alignment, models resident across calls.

    Mirrors the exact knobs the batch ``transcribe_video`` call site pins, so
    that — with the same VAD — a streaming re-transcription of the same audio
    reproduces the batch segments. The ASR model and the (language-fixed)
    align model are both kept loaded; unlike the batch path (which frees the
    ASR model before alignment to avoid coexistence), streaming trades a little
    memory for zero reloads.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.model_name = str(config.get("transcription_model", "large-v2"))
        self.device = str(config.get("transcription_device", "cpu"))
        self.language = config.get("transcription_language", "en")
        self.vad_method = str(config.get("vad_method", "pyannote"))
        self.vad_onset = float(config.get("vad_onset", 0.7))
        self.vad_offset = float(config.get("vad_offset", 0.363))
        self.batch_size = int(config.get("transcription_batch_size", 16))
        self.chunk_size = int(config.get("transcription_chunk_size", 30))
        self.threads = int(config.get("transcription_threads", 4))
        self.interpolate_method = str(config.get("transcription_interpolate", "nearest"))
        self._asr = None
        self._align = None
        self._align_meta = None

    def load(self) -> None:
        from . import transcription as tx  # reuse helpers + lightning patch
        tx._patch_lightning_load_for_torch_26()
        tx._ensure_ffmpeg_on_path()
        from .certs import install_certifi_https
        install_certifi_https()
        import whisperx

        # MPS isn't supported by the CT2/pyannote stack — mirror the batch remap.
        if self.device == "mps":
            logger.warning("transcription device mps unsupported; using cpu.")
            self.device = "cpu"
        compute_type = tx._resolve_compute_type(None, self.device)
        asr_options = tx._build_asr_options(
            beam_size=5, patience=None, length_penalty=None,
            temperatures=(0.0,), compression_ratio_threshold=2.4,
            logprob_threshold=-1.0, no_speech_threshold=0.6,
            condition_on_previous_text=False, suppress_numerals=False,
            initial_prompt=None, hotwords=None,
        )
        vad_options = tx._build_vad_options(vad_onset=self.vad_onset, vad_offset=self.vad_offset)
        self._asr = whisperx.load_model(
            whisper_arch=self.model_name, device=self.device, compute_type=compute_type,
            language=self.language, threads=self.threads, asr_options=asr_options,
            vad_method=self.vad_method, vad_options=vad_options,
        )
        if self.language:
            try:
                self._align, self._align_meta = whisperx.load_align_model(
                    language_code=self.language, device=self.device
                )
            except Exception:
                logger.warning("align model load failed; word scores unavailable.", exc_info=True)
        logger.info("WhisperXSession loaded (%s, %s).", self.model_name, self.device)

    def transcribe_audio(self, audio: np.ndarray, offset_sec: float = 0.0) -> List[Dict[str, Any]]:
        if self._asr is None:
            self.load()
        import whisperx
        from . import transcription as tx
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        kwargs: Dict[str, Any] = {"batch_size": self.batch_size, "chunk_size": self.chunk_size,
                                  "print_progress": False}
        if self.language:
            kwargs["language"] = self.language
        try:
            result = self._asr.transcribe(audio, **kwargs)
        except (IndexError, RuntimeError):
            # WhisperX/transformers raise IndexError when the VAD finds no
            # speech in the chunk (common for the pre-roll/silent lead-in) —
            # an expected, benign condition in streaming. Treat as no segments.
            logger.debug("WhisperX: no speech in chunk (or transient failure).")
            return []
        except Exception:
            logger.warning("WhisperX transcribe failed on chunk.", exc_info=True)
            return []
        segments = result.get("segments", []) if isinstance(result, dict) else []
        lang = result.get("language") if isinstance(result, dict) else self.language
        if self._align is not None and segments and lang:
            try:
                aligned = whisperx.align(
                    segments, self._align, self._align_meta, audio, self.device,
                    interpolate_method=self.interpolate_method, return_char_alignments=False,
                )
                segments = aligned.get("segments", segments)
            except Exception:
                logger.debug("alignment failed on chunk; native timestamps kept.", exc_info=True)
        return _apply_offset(tx._serialize_segments(segments), offset_sec)

    def close(self) -> None:
        from .torch_memory import free_torch_memory
        self._asr = None
        self._align = None
        self._align_meta = None
        try:
            free_torch_memory(self.device)
        except Exception:
            pass


def build_session(backend: str, config: Optional[Dict[str, Any]] = None) -> TranscriptionSession:
    """Instantiate the requested ASR session. Unknown/parakeet-unavailable
    falls back to WhisperX so the pipeline never breaks."""
    backend = (backend or "whisperx").lower()
    if backend == "parakeet":
        # Probe NeMo up front so an environment without it cleanly falls back
        # to WhisperX here (NeMo is imported lazily inside ParakeetSession.load,
        # so a bare import of the class would not surface the missing dep).
        try:
            import importlib.util
            if importlib.util.find_spec("nemo") is None:
                raise ImportError("nemo not installed")
            from .asr_parakeet import ParakeetSession
            return ParakeetSession(config)
        except Exception:
            logger.warning("Parakeet backend unavailable; falling back to WhisperX.", exc_info=True)
    return WhisperXSession(config)
