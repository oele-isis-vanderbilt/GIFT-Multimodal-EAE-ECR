"""Optional pre-ASR speech enhancement using Facebook Denoiser (streaming).

Used by the streaming drill-end transcription (``src/utils/drill_stream.py``)
to clean up noisy field audio before the ASR. Opt-in via ``enable_denoise``;
on any failure the session degrades to passing the raw audio through — it
never raises into the pipeline.

Models run at 16 kHz mono (the ASR's native rate). The dry/wet blend follows
FB's convention: ``dry=0`` is fully denoised, ``dry=1`` is fully original; a
small dry fraction keeps original speech transients, which empirically helps
WER vs. fully denoised output.
"""

import logging
from typing import Optional

from .torch_memory import free_torch_memory


VALID_MODELS = ("dns48", "dns64", "master64")


class DenoiseSession:
    """Load-once, smoothly-streaming FB Denoiser for the streaming ASR path.

    Wraps ``denoiser.demucs.DemucsStreamer`` — the package's purpose-built
    real-time API that carries the model's internal LSTM + resampling state
    across ``feed()`` calls, so chunked denoising has **no boundary
    discontinuities** (unlike independent per-chunk passes). The model is
    loaded once and reused for every chunk. Input/output are float32 mono at
    16 kHz. Never raises: on any failure ``feed`` returns the input unchanged
    so the pipeline degrades to raw audio.
    """

    def __init__(self, *, model_name: str = "dns48", dry: float = 0.04, device: str = "cpu") -> None:
        self.model_name = model_name
        self.dry = float(dry)
        # Demucs's conv1d path exceeds the MPS channel limit — keep it on cpu.
        self.device = "cpu" if device == "mps" else device
        self._model = None
        self._streamer = None
        self._ok = model_name in VALID_MODELS

    def _ensure(self) -> bool:
        if self._streamer is not None:
            return True
        if not self._ok:
            return False
        try:
            import torch  # noqa: F401
            from denoiser import pretrained
            from denoiser.demucs import DemucsStreamer
            factories = {"dns48": pretrained.dns48, "dns64": pretrained.dns64,
                         "master64": pretrained.master64}
            self._model = factories[self.model_name]().to(self.device).eval()
            self._streamer = DemucsStreamer(self._model, dry=self.dry)
            return True
        except Exception:
            logging.warning("DenoiseSession init failed; passing audio through raw.", exc_info=True)
            self._ok = False
            return False

    def feed(self, samples):
        """Denoise one forward chunk (float32 mono 16 kHz) → float32 mono."""
        import numpy as np
        arr = np.asarray(samples, dtype="float32").reshape(-1)
        if not self._ensure():
            return arr
        try:
            import torch
            with torch.no_grad():
                wav = torch.from_numpy(arr).view(1, -1).to(self.device)
                out = self._streamer.feed(wav)
            if out is None or out.numel() == 0:
                # Streamer buffered everything (needs more for a full frame);
                # emit nothing this step — the tail flushes later.
                return np.zeros(0, dtype="float32")
            return out.view(-1).cpu().numpy().astype("float32")
        except Exception:
            logging.warning("DenoiseSession.feed failed; passing chunk raw.", exc_info=True)
            return arr

    def flush(self):
        import numpy as np
        if self._streamer is None:
            return np.zeros(0, dtype="float32")
        try:
            import torch
            with torch.no_grad():
                out = self._streamer.flush()
            return out.view(-1).cpu().numpy().astype("float32") if out is not None else np.zeros(0, dtype="float32")
        except Exception:
            return np.zeros(0, dtype="float32")

    def close(self) -> None:
        self._model = None
        self._streamer = None
        try:
            free_torch_memory(self.device)
        except Exception:
            pass
