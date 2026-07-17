"""Optional pre-ASR speech enhancement using Facebook Denoiser.

Used by ``transcribe_video`` to clean up noisy field audio before WhisperX.
The denoiser is opt-in: callers pass a model name (``dns48``/``dns64``/
``master64``); when the name is missing or anything goes wrong we return
``False`` and the caller falls back to the raw audio. Never raises.

Models run at 16 kHz mono — input is resampled/downmixed via
``denoiser.dsp.convert_audio`` and the output WAV is written at 16 kHz mono,
which is also WhisperX's native sample rate (no further resample on the ASR
side).

The dry/wet blend follows FB's convention: ``dry=0`` is fully denoised,
``dry=1`` is fully original. Default 0.04 keeps a sliver of original speech
transients which empirically helps WER vs. fully denoised output.
"""

import logging
import os
import subprocess
from typing import Optional

from .audio import _find_binary
from .torch_memory import free_torch_memory


VALID_MODELS = ("dns48", "dns64", "master64")

# Chunked-inference geometry (seconds at the model's 16 kHz rate). Demucs
# activation memory scales linearly with input length (resample=4 internal
# upsampling), so long clips are processed in fixed windows of at most
# CHUNK + LEFT + RIGHT seconds — constant memory regardless of clip length.
# Clips up to one window take the identical single-forward path (bit-exact).
#
# Chunked output is *not* bit-equal to a full-clip forward: the models are
# causal and their bottleneck LSTM carries clip-long state (effectively a
# running noise estimate), so a window-local pass is an equally valid but
# slightly different denoise. Measured on real field audio (110 s, dns48):
# 10 s left context + 60 s chunks reconstruct the full-pass output at ~40 dB
# SNR (p99.9 |diff| ~0.4% of full scale), before the dry/wet blend further
# dilutes the residual. 0.25 s right context covers the resampling filters.
_CHUNK_SEC = 60.0
_CTX_LEFT_SEC = 10.0
_CTX_RIGHT_SEC = 0.25


def _denoise_forward(model, noisy):
    """Run ``model`` over ``noisy`` (B, C, T) with bounded activation memory.

    Returns a tensor the same length as ``noisy`` so the caller's dry/wet
    blend lines up sample-for-sample.

    Demucs's ``normalize=True`` scales the input by the std of the *whole*
    clip; per-chunk normalization would scale each chunk differently on
    non-stationary audio and audibly change the output. So the chunked path
    applies the global scaling once, runs the chunks with the model's
    internal normalization disabled, and rescales at the end. The remaining
    full-pass mismatch is the bottleneck LSTM's clip-long state (see module
    constants above for the measured ~40 dB parity).
    """
    import torch  # local import — module stays importable without torch

    sr = int(model.sample_rate)
    chunk = int(_CHUNK_SEC * sr)
    lctx = int(_CTX_LEFT_SEC * sr)
    rctx = int(_CTX_RIGHT_SEC * sr)
    total = noisy.shape[-1]

    if total <= chunk + lctx + rctx:
        with torch.no_grad():
            return model(noisy)

    saved_normalize = model.normalize
    if saved_normalize:
        mono = noisy.mean(dim=1, keepdim=True)
        std = mono.std(dim=-1, keepdim=True)
        noisy = noisy / (model.floor + std)
        model.normalize = False
    else:
        std = 1.0

    try:
        cores = []
        for start in range(0, total, chunk):
            end = min(start + chunk, total)
            a = max(0, start - lctx)
            b = min(total, end + rctx)
            with torch.no_grad():
                est = model(noisy[..., a:b])
            cores.append(est[..., start - a : start - a + (end - start)])
            del est
        return std * torch.cat(cores, dim=-1)
    finally:
        model.normalize = saved_normalize


def _extract_audio_16k_mono(source_video: str, output_wav: str) -> bool:
    """Extract first audio stream as 16 kHz mono PCM WAV via ffmpeg."""
    if not source_video or not os.path.exists(source_video):
        return False

    ffmpeg = _find_binary("ffmpeg")
    if ffmpeg is None:
        logging.debug("ffmpeg not available; cannot extract audio for %s", source_video)
        return False

    try:
        result = subprocess.run(
            [
                ffmpeg, "-y",
                "-i", source_video,
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                output_wav,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logging.debug("ffmpeg audio extraction failed for %s: %s", source_video, exc)
        return False

    if result.returncode != 0:
        logging.debug(
            "ffmpeg 16k mono extraction returned %d for %s; stderr tail: %s",
            result.returncode, source_video, (result.stderr or "")[-400:],
        )
        return False

    return os.path.exists(output_wav) and os.path.getsize(output_wav) > 0


def _safe_remove(path: Optional[str]) -> None:
    if not path:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def denoise_audio_to_wav(
    source_video_path: str,
    output_wav_path: str,
    *,
    model_name: str,
    dry: float = 0.04,
    device: str = "cpu",
) -> bool:
    """Run FB denoiser on the audio of ``source_video_path``; write 16 kHz mono WAV.

    Returns True on success; False on any failure (no denoiser pkg, bad model
    name, ffmpeg failure, model load/inference error). Never raises — the
    caller falls back to the un-denoised input on False.
    """
    if model_name not in VALID_MODELS:
        logging.warning(
            "denoise: unknown model %r; valid options are %s",
            model_name, VALID_MODELS,
        )
        return False

    try:
        from denoiser import pretrained  # type: ignore
        from denoiser.dsp import convert_audio  # type: ignore
        import torch  # type: ignore
        # ``torchaudio`` >= 2.8 routes I/O through ``torchcodec``, which has
        # an FFmpeg ABI dependency that frequently mis-matches conda-forge's
        # ffmpeg build. ``soundfile`` (libsndfile) is a much simpler audio
        # loader, ships with pyannote.audio's transitive deps, and produces
        # numpy arrays we can feed straight into the denoiser.
        import soundfile as sf  # type: ignore
    except ImportError:
        logging.warning(
            "denoiser/soundfile not installed; skipping denoising. "
            "Install via `pip install denoiser soundfile` to enable."
        )
        return False

    # FB Demucs's upsample/downsample path uses ``F.conv1d`` after a
    # ``view(-1, 1, time)`` reshape, which exceeds the MPS 65536-output-channel
    # kernel limit on long slices and raises NotImplementedError even with
    # PYTORCH_ENABLE_MPS_FALLBACK=1 (the fallback fires per-op, but the model
    # is already constructed on MPS by then). Remap to cpu transparently —
    # mirrors the whisperx mps->cpu fallback in src/utils/transcription.py.
    if device == "mps":
        logging.warning(
            "denoise_device=mps is not supported by FB Denoiser (MPS conv1d "
            "channel limit); falling back to cpu."
        )
        device = "cpu"

    factories = {
        "dns48": pretrained.dns48,
        "dns64": pretrained.dns64,
        "master64": pretrained.master64,
    }

    out_dir = os.path.dirname(output_wav_path) or "."
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError:
        logging.exception("denoise: cannot create output dir %s", out_dir)
        return False

    tmp_wav = os.path.join(out_dir, f"_denoise_input_{os.getpid()}.wav")

    if not _extract_audio_16k_mono(source_video_path, tmp_wav):
        logging.warning("denoise: failed to extract 16 kHz mono audio from %s", source_video_path)
        return False

    model = wav = noisy = estimate = None
    try:
        try:
            # soundfile returns (samples, channels) for stereo or (samples,)
            # for mono; torchaudio convention is (channels, samples).
            wav_np, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
            wav = torch.from_numpy(wav_np.T)  # (channels, samples)
        except Exception:
            logging.exception("denoise: soundfile.read failed for %s", tmp_wav)
            return False

        try:
            model = factories[model_name]().to(device).eval()
        except Exception:
            logging.exception("denoise: failed to load model %s on %s", model_name, device)
            return False

        try:
            noisy = convert_audio(wav.unsqueeze(0), sr, model.sample_rate, model.chin)
            noisy = noisy.to(device)
            estimate = _denoise_forward(model, noisy)
            estimate = (1.0 - dry) * estimate + dry * noisy
            out = estimate.squeeze(0).cpu().numpy()  # (channels, samples)
        except Exception:
            logging.exception("denoise: inference failed for %s", source_video_path)
            return False

        try:
            # soundfile expects (samples,) mono or (samples, channels) stereo.
            sf.write(output_wav_path, out.T, model.sample_rate, subtype="PCM_16")
        except Exception:
            logging.exception("denoise: failed to write %s", output_wav_path)
            return False

        return os.path.exists(output_wav_path) and os.path.getsize(output_wav_path) > 0
    finally:
        _safe_remove(tmp_wav)
        # Drop the model + waveform tensors and return their allocator blocks
        # before WhisperX loads — the denoiser shouldn't linger alongside ASR.
        model = wav = noisy = estimate = None
        free_torch_memory(device)


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
        # DemucsStreamer's causal conv1d path hits the MPS channel limit like
        # the batch path — keep it on cpu.
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
