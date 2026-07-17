"""Load-once ASR session for streaming drill-end detection.

A :class:`TranscriptionSession` loads its model(s) **once** and exposes
``transcribe_audio(np_16k_mono, offset_sec)`` returning segments in the
canonical schema used by ``drill_window`` — ``{start, end, text, words:[{word,
start, end, score}]}`` with timestamps already offset to the original-video
timeline. No per-call model reload, no file I/O.

The shipped implementation is :class:`~src.utils.asr_parakeet.ParakeetSession`
(NeMo Parakeet-TDT). ``build_session(config)`` constructs it.
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


def build_session(config: Optional[Dict[str, Any]] = None):
    """Instantiate the ASR session (NeMo Parakeet)."""
    from .asr_parakeet import ParakeetSession
    return ParakeetSession(config)
