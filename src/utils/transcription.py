"""WhisperX transcription with word-level alignment.

Runs WhisperX on the session source audio and writes a sidecar JSON with
segment- and word-level timestamps. Alignment is performed via wav2vec2
forced alignment for ~40 ms word-timestamp accuracy — segment boundaries
are derived from the aligned words.

All arguments are split into two layers:

* **Layer 1** — the four core ASR kwargs (``model``, ``device``, ``language``,
  ``compute_type``) plus the two denoise toggles (``denoise_model``,
  ``denoise_device``) are exposed via ``config.json`` and passed through by
  the processing engine.
* **Layer 2** — every other kwarg is an "expert knob" with a sensible default.
  The processing engine call site sets them explicitly so tuning happens in
  one obvious place; they are deliberately NOT exposed in the user-facing
  config to keep that surface small. ``denoise_dry`` (FB Denoiser dry/wet
  mix) lives here too — only the model + device toggles reach the GUI.

Every error path is swallowed with a log message and ``None`` returned so
that a failed transcription never breaks the pipeline.
"""

import json
import logging
import os
import shutil
import sys
from typing import Any, Dict, Optional

from .audio import extract_audio_to_wav, has_audio_stream
from .certs import install_certifi_https
from .denoise import denoise_audio_to_wav
from .torch_memory import free_torch_memory


def _ensure_ffmpeg_on_path() -> None:
    """``whisperx.load_audio`` shells out to a bare ``ffmpeg``.

    When the engine is invoked via the env's python without activating the
    env, the env bin (where conda installs ffmpeg) isn't on PATH and whisperx
    raises FileNotFoundError. Prepend the interpreter's directory so the
    subprocess resolves — mirrors ``audio._find_binary``'s fallback.
    """
    if shutil.which("ffmpeg"):
        return
    exe_dir = os.path.dirname(sys.executable)
    if os.path.isfile(os.path.join(exe_dir, "ffmpeg")):
        os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")


SCHEMA_VERSION = "1.2"


def _patch_lightning_load_for_torch_26() -> None:
    """Make ``lightning_fabric.utilities.cloud_io._load`` (used by
    ``pyannote.audio.core.model.Model.from_pretrained`` via the ``pl_load``
    alias) call ``torch.load(weights_only=False)``.

    PyTorch 2.6 changed the ``weights_only`` default to ``True``, which
    rejects the heterogeneous pickle classes (omegaconf, typing.Any,
    pyannote internals, ...) inside pyannote.audio's bundled VAD checkpoint.
    Lightning's ``_load`` doesn't accept a ``weights_only`` override, and
    allowlisting via ``add_safe_globals`` is whack-a-mole — every checkpoint
    update could surface a new class. The VAD weights ship inside the
    ``whisperx`` package itself (not downloaded at runtime), so falling
    back to ``weights_only=False`` for this load is safe. Idempotent.
    """
    try:
        import torch
        import lightning_fabric.utilities.cloud_io as cio
    except ImportError:
        return
    if getattr(cio._load, "_giftpose_patched", False):
        return

    def _patched_load(path_or_url, map_location=None, **kwargs):  # type: ignore[no-untyped-def]
        # Force weights_only=False regardless of what the caller passed —
        # pyannote 3.4.0+ forwards a weights_only kwarg from from_pretrained,
        # which on torch 2.6+ defaults to True and rejects the omegaconf /
        # typing.Any classes inside the VAD pickle.
        kwargs["weights_only"] = False
        return torch.load(path_or_url, map_location=map_location, **kwargs)

    _patched_load._giftpose_patched = True  # type: ignore[attr-defined]
    cio._load = _patched_load


_patch_lightning_load_for_torch_26()


def _resolve_compute_type(explicit: Optional[str], device: str) -> str:
    """Map ``None`` → sensible default for the device."""
    if explicit:
        return explicit
    return "float16" if device.startswith("cuda") else "float32"


def _build_asr_options(
    *,
    beam_size: int,
    patience: Optional[float],
    length_penalty: Optional[float],
    temperatures: Any,
    compression_ratio_threshold: float,
    logprob_threshold: float,
    no_speech_threshold: float,
    condition_on_previous_text: bool,
    suppress_numerals: bool,
    initial_prompt: Optional[str],
    hotwords: Optional[str],
) -> Dict[str, Any]:
    """Compose the ``asr_options`` dict accepted by ``whisperx.load_model``."""
    asr_options: Dict[str, Any] = {
        "beam_size": beam_size,
        "temperatures": temperatures,
        "compression_ratio_threshold": compression_ratio_threshold,
        "log_prob_threshold": logprob_threshold,
        "no_speech_threshold": no_speech_threshold,
        "condition_on_previous_text": condition_on_previous_text,
        "suppress_numerals": suppress_numerals,
    }
    if patience is not None:
        asr_options["patience"] = patience
    if length_penalty is not None:
        asr_options["length_penalty"] = length_penalty
    if initial_prompt is not None:
        asr_options["initial_prompt"] = initial_prompt
    if hotwords is not None:
        asr_options["hotwords"] = hotwords
    return asr_options


def _build_vad_options(*, vad_onset: float, vad_offset: float) -> Dict[str, Any]:
    """Compose the ``vad_options`` dict accepted by ``whisperx.load_model``.

    Only VAD-backbone kwargs belong here. ``chunk_size`` is a transcription
    parameter, not a VAD one, and whisperx 3.2.0's ``load_vad_model`` rejects
    it. We pass ``chunk_size`` to ``model.transcribe()`` instead.
    """
    return {
        "vad_onset": vad_onset,
        "vad_offset": vad_offset,
    }


def _serialize_segments(segments):
    """Convert WhisperX segment dicts into a JSON-clean list of dicts.

    Handles both the pre-alignment shape (``text`` only) and the aligned shape
    (``text`` + ``words`` with ``start``/``end``/``score``). Missing fields fall
    back to ``None`` rather than raising.
    """
    out = []
    for idx, seg in enumerate(segments or []):
        start = _float_or_none(seg.get("start"))
        end = _float_or_none(seg.get("end"))
        # wav2vec2 alignment can fail on numeric / non-pronounceable tokens
        # (e.g., a bare "1611"), leaving the word's timestamps null and the
        # segment's start/end inverted (start > end). Swap them so downstream
        # consumers (subtitle overlay, timeline) always see a non-negative
        # duration; the per-word ``None`` timestamps still signal that
        # alignment was unreliable.
        if start is not None and end is not None and start > end:
            start, end = end, start
        entry: Dict[str, Any] = {
            "id": idx,
            "start": start,
            "end": end,
            "text": str(seg.get("text", "")).strip(),
        }
        words = []
        for w in seg.get("words", []) or []:
            w_start = _float_or_none(w.get("start"))
            w_end = _float_or_none(w.get("end"))
            if w_start is not None and w_end is not None and w_start > w_end:
                w_start, w_end = w_end, w_start
            words.append(
                {
                    "word": str(w.get("word", "")).strip(),
                    "start": w_start,
                    "end": w_end,
                    "score": _float_or_none(w.get("score")),
                }
            )
        if words:
            entry["words"] = words
        out.append(entry)
    return out


def _float_or_none(value) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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
    """Write ``{basename}_Transcription.json`` in the standard schema.

    Used by the streaming path (which already has serialized, offset segment
    dicts) to produce the same sidecar the batch ``transcribe_video`` writes,
    so downstream consumers and the viewer see an identical artifact.
    """
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


def transcribe_video(
    source_video_path: str,
    output_dir: str,
    video_basename: str,
    *,
    # ===== Layer 1: exposed in config.json =====
    model: str = "large-v3",
    device: str = "cpu",
    language: Optional[str] = None,
    compute_type: Optional[str] = None,
    # ===== Optional drill-window audio slice =====
    audio_start_sec: Optional[float] = None,       # If set, only transcribe audio from this time onward.
    audio_end_sec: Optional[float] = None,         # Optional upper bound. Timestamps in the saved sidecar are offset back to the original-video timeline.
    # ===== Optional pre-ASR denoising (Facebook Denoiser) =====
    denoise_model: Optional[str] = None,           # Layer 1: None | "dns48" | "dns64" | "master64"
    denoise_device: Optional[str] = None,          # Layer 1: cpu / cuda / mps; None reuses ``device``
    denoise_dry: float = 0.04,                     # Layer 2: dry/wet mix (0=full denoise, 1=original)
    keep_denoised_wav: bool = True,                # Layer 2: persist {basename}_denoised.wav as artifact (False = delete after ASR)
    # ===== Layer 2: expert knobs, set at call site =====
    batch_size: int = 16,
    beam_size: int = 5,
    patience: Optional[float] = None,
    length_penalty: Optional[float] = None,
    temperature: float = 0.0,
    temperature_increment_on_fallback: float = 0.2,
    compression_ratio_threshold: float = 2.4,
    logprob_threshold: float = -1.0,
    no_speech_threshold: float = 0.6,
    condition_on_previous_text: bool = False,
    suppress_numerals: bool = False,
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
    vad_method: str = "pyannote",
    vad_onset: float = 0.500,
    vad_offset: float = 0.363,
    chunk_size: int = 30,
    run_alignment: bool = True,
    return_char_alignments: bool = False,
    interpolate_method: str = "nearest",
    threads: int = 4,
) -> Optional[Dict[str, Any]]:
    """Transcribe ``source_video_path`` with WhisperX.

    Returns the JSON-ready dict that was written to disk, or ``None`` if
    transcription could not run (no audio, whisperx missing, or an error
    occurred — all logged).

    The output sidecar lives at
    ``{output_dir}/{video_basename}_Transcription.json``.
    """
    if not has_audio_stream(source_video_path):
        logging.info(
            "Transcription skipped: no audio stream in %s", source_video_path
        )
        return None

    _ensure_ffmpeg_on_path()

    # Denoiser / wav2vec2 / pyannote weights download via torch.hub (urllib);
    # route TLS through certifi so a broken OS cert store doesn't block them.
    install_certifi_https()

    try:
        import whisperx  # type: ignore
    except ImportError:
        logging.warning(
            "whisperx not installed; skipping transcription. "
            "Install via pip to enable."
        )
        return None

    # WhisperX/faster-whisper (CTranslate2) only supports cpu and cuda — never
    # mps. Apple-Silicon configs commonly set device="mps" for the pose
    # pipeline; transparently remap to cpu here so the rest of the pipeline
    # doesn't have to care. Denoising already ran (or skipped) above using its
    # own ``denoise_device``, so this remap only affects the ASR pass.
    if device == "mps":
        logging.warning(
            "transcription_device=mps is not supported by whisperx/CTranslate2; "
            "falling back to cpu for ASR + alignment."
        )
        device = "cpu"

    effective_compute_type = _resolve_compute_type(compute_type, device)

    # Classic Whisper temperature ladder used as fallback if decoding fails.
    if temperature_increment_on_fallback and temperature_increment_on_fallback > 0:
        temperatures = tuple(
            round(temperature + temperature_increment_on_fallback * i, 2)
            for i in range(int((1.0 - temperature) / temperature_increment_on_fallback) + 1)
        )
    else:
        temperatures = (float(temperature),)

    asr_options = _build_asr_options(
        beam_size=beam_size,
        patience=patience,
        length_penalty=length_penalty,
        temperatures=temperatures,
        compression_ratio_threshold=compression_ratio_threshold,
        logprob_threshold=logprob_threshold,
        no_speech_threshold=no_speech_threshold,
        condition_on_previous_text=condition_on_previous_text,
        suppress_numerals=suppress_numerals,
        initial_prompt=initial_prompt,
        hotwords=hotwords,
    )
    vad_options = _build_vad_options(vad_onset=vad_onset, vad_offset=vad_offset)

    # Only the transcription pipeline reads ``audio_input_path``. The video
    # mux step (``attach_audio_in_place`` in processing_engine) operates on
    # the ORIGINAL ``config["video_path"]``, never on this denoised WAV — so
    # saved annotated videos always retain the original (un-denoised) audio.
    audio_input_path = source_video_path
    denoise_used: Optional[str] = None
    denoise_wav_path: Optional[str] = None

    # If a slice was requested, pre-extract the source audio into a temp WAV
    # so denoising and whisperx both consume the post-entry portion only.
    # Timestamps returned by whisperx are relative to the slice; we offset
    # them by ``audio_start_sec`` before serializing so the sidecar matches
    # the original-video timeline.
    audio_offset_sec = float(audio_start_sec) if audio_start_sec else 0.0
    sliced_input_path: Optional[str] = None
    if audio_offset_sec > 0 or audio_end_sec is not None:
        sliced_input_path = os.path.join(
            output_dir, f"{video_basename}__transcription_slice_{os.getpid()}.wav"
        )
        if extract_audio_to_wav(
            source_video_path,
            sliced_input_path,
            start_sec=audio_offset_sec if audio_offset_sec > 0 else None,
            end_sec=audio_end_sec,
            sample_rate=16000,
            channels=1,
        ):
            audio_input_path = sliced_input_path
            logging.info(
                "Sliced audio for transcription: start=%.3fs end=%s -> %s",
                audio_offset_sec,
                f"{float(audio_end_sec):.3f}s" if audio_end_sec else "EOF",
                sliced_input_path,
            )
        else:
            logging.warning(
                "Failed to slice audio at start=%.3fs; falling back to full source.",
                audio_offset_sec,
            )
            sliced_input_path = None
            audio_offset_sec = 0.0

    if denoise_model:
        denoise_wav_path = os.path.join(output_dir, f"{video_basename}_denoised.wav")
        ok = denoise_audio_to_wav(
            audio_input_path,
            denoise_wav_path,
            model_name=denoise_model,
            dry=denoise_dry,
            device=denoise_device or device,
        )
        if ok:
            audio_input_path = denoise_wav_path
            denoise_used = denoise_model
            logging.info(
                "Denoised audio with %s (dry=%.3f) -> %s (kept as artifact)",
                denoise_model, denoise_dry, denoise_wav_path,
            )
        else:
            logging.warning(
                "Denoising with %s failed; falling back to raw audio.",
                denoise_model,
            )
            denoise_wav_path = None

    def _cleanup_slice() -> None:
        # Temp slice WAV must go on every exit path, not just the happy one.
        if sliced_input_path and os.path.exists(sliced_input_path):
            try:
                os.remove(sliced_input_path)
            except OSError:
                logging.debug(
                    "Could not remove sliced input WAV %s", sliced_input_path, exc_info=True
                )

    try:
        audio = whisperx.load_audio(audio_input_path)
    except Exception:
        logging.exception("whisperx.load_audio failed for %s", audio_input_path)
        _cleanup_slice()
        return None

    try:
        asr_model = whisperx.load_model(
            whisper_arch=model,
            device=device,
            compute_type=effective_compute_type,
            language=language,
            threads=threads,
            asr_options=asr_options,
            vad_method=vad_method,
            vad_options=vad_options,
        )
    except Exception:
        logging.exception("whisperx.load_model failed (model=%s, device=%s)", model, device)
        _cleanup_slice()
        return None

    try:
        transcribe_kwargs: Dict[str, Any] = {
            "batch_size": batch_size,
            "chunk_size": chunk_size,
            "print_progress": False,
        }
        if language is not None:
            transcribe_kwargs["language"] = language
        result = asr_model.transcribe(audio, **transcribe_kwargs)
    except Exception:
        logging.exception("whisperx transcribe failed for %s", source_video_path)
        _cleanup_slice()
        return None
    finally:
        # The ASR model (~3 GB for large-v2) is done after transcribe; free it
        # before the wav2vec2 alignment model loads so the two never coexist.
        del asr_model
        free_torch_memory(device)

    detected_language = result.get("language") if isinstance(result, dict) else None
    segments = result.get("segments", []) if isinstance(result, dict) else []
    aligned = False

    if run_alignment and detected_language and segments:
        align_model = None
        align_metadata = None
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=detected_language, device=device
            )
            aligned_result = whisperx.align(
                segments,
                align_model,
                align_metadata,
                audio,
                device,
                interpolate_method=interpolate_method,
                return_char_alignments=return_char_alignments,
            )
            segments = aligned_result.get("segments", segments)
            aligned = True
        except Exception:
            logging.warning(
                "wav2vec2 alignment failed for language=%s; falling back to "
                "whisper's native timestamps.",
                detected_language,
                exc_info=True,
            )
        finally:
            align_model = None
            align_metadata = None
            free_torch_memory(device)

    # The raw waveform (float32 @16 kHz) has no further use — only the
    # segment dicts feed serialization below.
    del audio
    serialized_segments = _serialize_segments(segments)
    if audio_offset_sec > 0:
        for seg in serialized_segments:
            if isinstance(seg.get("start"), (int, float)):
                seg["start"] = float(seg["start"]) + audio_offset_sec
            if isinstance(seg.get("end"), (int, float)):
                seg["end"] = float(seg["end"]) + audio_offset_sec
            for w in seg.get("words", []) or []:
                if isinstance(w.get("start"), (int, float)):
                    w["start"] = float(w["start"]) + audio_offset_sec
                if isinstance(w.get("end"), (int, float)):
                    w["end"] = float(w["end"]) + audio_offset_sec

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_video_basename": video_basename,
        "language": detected_language,
        "model": model,
        "compute_type": effective_compute_type,
        "aligned": aligned,
        "audio_window": (
            {
                "start_sec": audio_offset_sec if audio_offset_sec > 0 else None,
                "end_sec": float(audio_end_sec) if audio_end_sec else None,
            }
            if (audio_offset_sec > 0 or audio_end_sec is not None)
            else None
        ),
        "denoise": (
            {
                "model": denoise_used,
                "dry": denoise_dry,
                "audio_artifact": (
                    os.path.basename(denoise_wav_path) if denoise_wav_path else None
                ),
            }
            if denoise_used
            else None
        ),
        "segments": serialized_segments,
    }

    out_path = os.path.join(output_dir, f"{video_basename}_Transcription.json")
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logging.info("Saved transcription to %s (%d segments)", out_path, len(payload["segments"]))
    except OSError:
        logging.exception("Failed to write transcription JSON to %s", out_path)
        _cleanup_slice()
        return None

    if denoise_wav_path and not keep_denoised_wav:
        # Developer flag: discard the denoised audio after ASR. The transcription
        # JSON's ``denoise.audio_artifact`` field is overwritten to ``null`` so it
        # doesn't reference a missing file.
        try:
            os.remove(denoise_wav_path)
            payload["denoise"]["audio_artifact"] = None
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logging.debug("Removed denoised WAV (keep_denoised_wav=False): %s", denoise_wav_path)
        except OSError:
            logging.debug("Could not remove denoised WAV %s", denoise_wav_path, exc_info=True)

    _cleanup_slice()

    return payload
