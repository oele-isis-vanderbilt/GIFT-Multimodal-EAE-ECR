"""Audio preservation helpers.

cv2.VideoWriter does not write audio, so camera-view annotated outputs lose
the source audio track. These helpers probe for and re-mux the original audio
using ffmpeg/ffprobe. Every function fails closed: any error returns a falsy
value or leaves the input in place, so the pipeline never breaks when ffmpeg
is missing or the source has no audio.
"""

import logging
import os
import shutil
import subprocess
import sys
import threading
from typing import Optional


def _find_binary(name: str) -> Optional[str]:
    found = shutil.which(name)
    if found:
        return found
    # When invoked via the env's python without activating the env (PATH has
    # no env bin), look next to the interpreter — conda/venv installs place
    # ffmpeg/ffprobe there. Otherwise audio mux/extraction silently degrades.
    candidate = os.path.join(os.path.dirname(sys.executable), name)
    if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def has_audio_stream(video_path: str) -> bool:
    """Return True if ``video_path`` has at least one audio stream.

    Returns False on any failure: missing ffprobe, unreadable file, no audio
    stream, or non-zero exit.
    """
    if not video_path or not os.path.exists(video_path):
        return False

    ffprobe = _find_binary("ffprobe")
    if ffprobe is None:
        logging.debug("ffprobe not available; assuming no audio stream for %s", video_path)
        return False

    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logging.debug("ffprobe failed for %s: %s", video_path, exc)
        return False

    return result.returncode == 0 and "audio" in result.stdout


def probe_audio_duration_sec(video_path: str) -> Optional[float]:
    """Duration (seconds) of ``video_path``'s first audio stream, or None.

    Tries the audio stream's own duration, then the container duration (some
    muxers only stamp the latter). Returns None on any failure — missing
    ffprobe, unreadable file, no audio stream — callers must treat None as
    "unknown", never as zero.
    """
    if not video_path or not os.path.exists(video_path):
        return None
    ffprobe = _find_binary("ffprobe")
    if ffprobe is None:
        return None
    for entries in ("stream=duration", "format=duration"):
        cmd = [ffprobe, "-v", "error"]
        if entries.startswith("stream="):
            cmd += ["-select_streams", "a:0"]
        cmd += ["-show_entries", entries, "-of", "csv=p=0", video_path]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (subprocess.TimeoutExpired, OSError) as exc:
            logging.debug("ffprobe duration probe failed for %s: %s", video_path, exc)
            return None
        if result.returncode == 0:
            try:
                val = float((result.stdout or "").strip().splitlines()[0])
                if val > 0:
                    return val
            except (ValueError, IndexError):
                pass
    return None


def mux_audio_from_source(
    silent_video: str,
    audio_source: str,
    output_path: str,
    *,
    audio_start_sec: Optional[float] = None,
    audio_end_sec: Optional[float] = None,
) -> bool:
    """Copy the video stream from ``silent_video`` and the first audio stream
    from ``audio_source`` into ``output_path``.

    When ``audio_start_sec`` / ``audio_end_sec`` are given, the audio source
    is sliced to that range using ffmpeg's ``-ss`` / ``-to`` (input-side) and
    re-encoded to AAC (stream-copy doesn't survive sub-second seeks). When no
    slice is requested both streams are stream-copied as before.

    Returns True on success, False on any failure. Never raises.
    """
    if not os.path.exists(silent_video) or not os.path.exists(audio_source):
        return False

    ffmpeg = _find_binary("ffmpeg")
    if ffmpeg is None:
        logging.debug("ffmpeg not available; skipping audio mux for %s", silent_video)
        return False

    cmd = [ffmpeg, "-y", "-i", silent_video]
    slicing = audio_start_sec is not None or audio_end_sec is not None
    if audio_start_sec is not None and audio_start_sec > 0:
        cmd += ["-ss", f"{float(audio_start_sec):.3f}"]
    if audio_end_sec is not None and audio_end_sec > 0:
        cmd += ["-to", f"{float(audio_end_sec):.3f}"]
    cmd += ["-i", audio_source, "-map", "0:v", "-map", "1:a:0", "-c:v", "copy"]
    if slicing:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-c:a", "copy"]
    cmd += ["-shortest", output_path]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logging.debug("ffmpeg mux failed for %s: %s", silent_video, exc)
        return False

    if result.returncode != 0:
        logging.debug(
            "ffmpeg mux returned %d for %s; stderr tail: %s",
            result.returncode,
            silent_video,
            (result.stderr or "")[-400:],
        )
        return False

    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def extract_audio_to_wav(
    source_video: str,
    output_wav: str,
    *,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
    sample_rate: int = 48000,
    channels: int = 2,
) -> bool:
    """Extract the first audio stream from ``source_video`` to a WAV file.

    Optional ``start_sec`` / ``end_sec`` slice the source on the way out;
    ``-ss`` and ``-to`` are placed before ``-i`` for fast seek and exact-end
    semantics. Default output is 48 kHz, 16-bit PCM, stereo; set
    ``sample_rate`` / ``channels`` to override (the transcription pipeline
    feeds the ASR 16 kHz mono, for example). Returns True on success.
    """
    if not source_video or not os.path.exists(source_video):
        return False

    ffmpeg = _find_binary("ffmpeg")
    if ffmpeg is None:
        logging.debug("ffmpeg not available; cannot extract audio for %s", source_video)
        return False

    cmd = [ffmpeg, "-y"]
    if start_sec is not None and start_sec > 0:
        cmd += ["-ss", f"{float(start_sec):.3f}"]
    if end_sec is not None and end_sec > 0:
        cmd += ["-to", f"{float(end_sec):.3f}"]
    cmd += [
        "-i", source_video,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(int(sample_rate)),
        "-ac", str(int(channels)),
        output_wav,
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        logging.debug("ffmpeg audio extraction failed for %s: %s", source_video, exc)
        return False

    if result.returncode != 0:
        logging.debug(
            "ffmpeg audio extraction returned %d for %s; stderr tail: %s",
            result.returncode,
            source_video,
            (result.stderr or "")[-400:],
        )
        return False

    return os.path.exists(output_wav) and os.path.getsize(output_wav) > 0


def attach_audio_in_place(
    annotated_video_path: str,
    audio_source: str,
    *,
    audio_start_sec: Optional[float] = None,
    audio_end_sec: Optional[float] = None,
) -> None:
    """Attach ``audio_source``'s audio track to ``annotated_video_path`` in place.

    Algorithm:
    1. Rename ``annotated_video_path`` → ``*_noaudio<ext>``.
    2. Run ``mux_audio_from_source`` targeting the original path; pass through
       optional ``audio_start_sec`` / ``audio_end_sec`` so the muxed audio is
       sliced to match a trimmed annotated video.
    3. On success, delete the ``_noaudio`` temp.
    4. On failure, restore the original path from the ``_noaudio`` temp.

    Silent on failure — the annotated video is preserved; only a debug log is
    emitted. Never raises.
    """
    if not os.path.exists(annotated_video_path):
        return

    base, ext = os.path.splitext(annotated_video_path)
    tmp_silent_path = f"{base}_noaudio{ext}"

    try:
        os.replace(annotated_video_path, tmp_silent_path)
    except OSError as exc:
        logging.debug("Could not rename %s for audio mux: %s", annotated_video_path, exc)
        return

    ok = mux_audio_from_source(
        tmp_silent_path,
        audio_source,
        annotated_video_path,
        audio_start_sec=audio_start_sec,
        audio_end_sec=audio_end_sec,
    )

    if ok:
        try:
            os.remove(tmp_silent_path)
        except OSError:
            pass
        return

    # mux failed — restore the original silent file in place.
    try:
        if os.path.exists(annotated_video_path):
            os.remove(annotated_video_path)
        os.replace(tmp_silent_path, annotated_video_path)
    except OSError as exc:
        logging.debug("Could not restore %s after failed mux: %s", annotated_video_path, exc)


