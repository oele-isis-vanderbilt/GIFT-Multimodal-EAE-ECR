"""NeMo Parakeet ASR session for streaming drill-end detection.

Wraps ``nvidia/parakeet-tdt-0.6b-v2`` (an ``EncDecRNNTBPEModel``) behind the
same :class:`TranscriptionSession` contract as the WhisperX path: load once,
``transcribe_audio(np_16k_mono, offset_sec)`` returns segments in the canonical
``{start, end, text, words:[{word, start, end, score}]}`` schema with
timestamps offset to the original-video timeline.

Parakeet emits word- and segment-level timestamps directly (no separate
alignment model). It has no wav2vec2-style per-word confidence, so ``score``
is ``None`` — the drill-end gate is lenient with ``None`` (word-presence is
the primary rule) and the LocalAgreement-2 confirmation guards against
transient mis-hearings.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .asr_session import TranscriptionSession

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v2"


class ParakeetSession(TranscriptionSession):
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        config = config or {}
        self.model_name = str(config.get("parakeet_model", DEFAULT_MODEL))
        self.device = str(config.get("transcription_device", "cpu"))
        self._model = None

    def load(self) -> None:
        import nemo.collections.asr as nemo_asr
        self._model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)
        self._model.eval()
        # NeMo runs on CPU when CUDA is absent; MPS is not a supported NeMo
        # device, so we leave placement to NeMo (CPU on this box).
        logger.info("ParakeetSession loaded (%s).", self.model_name)

    def transcribe_audio(self, audio: np.ndarray, offset_sec: float = 0.0) -> List[Dict[str, Any]]:
        if self._model is None:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        try:
            out = self._model.transcribe([audio], timestamps=True, verbose=False)
        except Exception:
            logger.warning("Parakeet transcribe failed on chunk.", exc_info=True)
            return []
        if not out:
            return []
        hyp = out[0]
        segments = _hyp_to_segments(hyp)
        for seg in segments:
            for k in ("start", "end"):
                if isinstance(seg.get(k), (int, float)):
                    seg[k] = float(seg[k]) + offset_sec
            for w in seg.get("words", []) or []:
                for k in ("start", "end"):
                    if isinstance(w.get(k), (int, float)):
                        w[k] = float(w[k]) + offset_sec
        return segments

    def close(self) -> None:
        self._model = None
        try:
            from .torch_memory import free_torch_memory
            free_torch_memory("cpu")
        except Exception:
            pass


def _hyp_to_segments(hyp) -> List[Dict[str, Any]]:
    """Convert a NeMo Hypothesis (timestamps=True) to canonical segments.

    Prefers the model's own segment timestamps; attaches the words that fall
    within each segment. Falls back to a single whole-utterance segment when
    only word timestamps (or only text) are available.
    """
    ts = getattr(hyp, "timestamp", None) or {}
    words = _norm_units(ts.get("word"))
    segs = _norm_units(ts.get("segment"))
    text = getattr(hyp, "text", "") or ""

    if segs:
        out = []
        for i, s in enumerate(segs):
            s_start, s_end = s.get("start"), s.get("end")
            seg_words = [
                {"word": w["word"], "start": w.get("start"), "end": w.get("end"), "score": None}
                for w in words
                if _within(w, s_start, s_end)
            ]
            entry: Dict[str, Any] = {
                "id": i,
                "start": s_start,
                "end": s_end,
                "text": str(s.get("text", "")).strip(),
            }
            if seg_words:
                entry["words"] = seg_words
            out.append(entry)
        return out

    if words:
        entry = {
            "id": 0,
            "start": words[0].get("start"),
            "end": words[-1].get("end"),
            "text": text.strip(),
            "words": [
                {"word": w["word"], "start": w.get("start"), "end": w.get("end"), "score": None}
                for w in words
            ],
        }
        return [entry]

    return [{"id": 0, "start": None, "end": None, "text": text.strip()}] if text.strip() else []


def _norm_units(units) -> List[Dict[str, Any]]:
    """Normalize NeMo timestamp entries to {word/text, start, end} in seconds.

    NeMo may key the token under ``word``/``segment``/``char`` and the times
    under ``start``/``end`` (seconds) or ``start_offset``/``end_offset``
    (frames). We prefer the seconds keys.
    """
    out = []
    for u in units or []:
        if not isinstance(u, dict):
            continue
        tok = u.get("word") or u.get("segment") or u.get("char") or u.get("text") or ""
        start = u.get("start", u.get("start_time"))
        end = u.get("end", u.get("end_time"))
        out.append({"word": str(tok).strip(), "text": str(tok).strip(),
                    "start": _f(start), "end": _f(end)})
    return out


def _within(w, s_start, s_end) -> bool:
    ws = w.get("start")
    if ws is None or s_start is None or s_end is None:
        return False
    return float(s_start) - 0.01 <= float(ws) <= float(s_end) + 0.01


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None
