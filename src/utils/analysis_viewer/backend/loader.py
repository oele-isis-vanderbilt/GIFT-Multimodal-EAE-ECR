"""v2 schema loader for the analysis viewer sidecar.

The backend's only owner of the JSON schema — fully self-contained, no
dependency on any other GUI layer.

Inputs:
- {basename}_Analysis.json (schema_version >= "2.0")
- optional sidecars: {basename}_Transcription.json, {basename}_DrillWindow.json

Outputs the JSON enriched with:
- top-level transcription block (if sidecar present)
- top-level drill_window (preferring embedded block, falling back to sidecar)
- top-level resolved_video_path / session_json_path
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv", ".m4v"]
SUPPORTED_SCHEMA_MAJORS = {"2"}


class SessionLoadError(Exception):
    """Raised when the session JSON cannot be parsed or fails validation."""


def resolve_run_or_analysis(path: str) -> str:
    """Accept either a run folder (with RunInfo.json) or a direct Analysis.json.

    Returns the absolute path to the Analysis.json. Raises SessionLoadError on
    structural problems so the FastAPI handler can surface a clean error.
    """
    abs_path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(abs_path):
        manifest = os.path.join(abs_path, "RunInfo.json")
        if not os.path.isfile(manifest):
            raise SessionLoadError(
                f"{abs_path} is a directory but has no RunInfo.json — not a v1 run folder."
            )
        try:
            with open(manifest, "r", encoding="utf-8") as fh:
                info = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise SessionLoadError(f"Could not read RunInfo.json: {exc}") from exc
        analysis_rel = (info or {}).get("analysis_file") or ""
        if not analysis_rel:
            raise SessionLoadError(f"RunInfo.json at {manifest} is missing 'analysis_file'.")
        candidate = os.path.join(abs_path, analysis_rel)
        if not os.path.isfile(candidate):
            raise SessionLoadError(f"Analysis file referenced by RunInfo.json does not exist: {candidate}")
        return candidate
    if os.path.isfile(abs_path):
        return abs_path
    raise SessionLoadError(f"Path not found: {abs_path}")


def load_session(json_path: str) -> Dict[str, Any]:
    """Load and enrich an Analysis.json file. Raises SessionLoadError on bad input.

    ``json_path`` may also be a run folder; in that case the embedded
    ``RunInfo.json`` is read to find the analysis JSON.
    """
    abs_path = resolve_run_or_analysis(json_path)
    if not os.path.isfile(abs_path):
        raise SessionLoadError(f"Session file not found: {abs_path}")

    try:
        with open(abs_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SessionLoadError(f"Invalid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise SessionLoadError("Session JSON root must be an object.")

    schema_version = str(data.get("schema_version") or "")
    major = schema_version.split(".", 1)[0]
    if major not in SUPPORTED_SCHEMA_MAJORS:
        raise SessionLoadError(
            f"Unsupported schema_version {schema_version!r}; backend supports majors {sorted(SUPPORTED_SCHEMA_MAJORS)}"
        )

    for required in ("session", "video", "timeline", "metrics", "flags"):
        if required not in data:
            raise SessionLoadError(f"Missing required top-level key: {required!r}")

    resolved_video = _resolve_video_path(abs_path, data)
    if resolved_video:
        data.setdefault("video", {})["video_path"] = resolved_video

    _attach_transcription_sidecar(abs_path, data)
    _attach_drill_window(abs_path, data)
    _attach_run_info(abs_path, data)

    data["session_json_path"] = abs_path
    data["resolved_video_path"] = resolved_video

    # Instructor-owned adjustments (POD frame override, drill-end override,
    # deleted flags) layer on top of the engine's immutable Analysis.json.
    from .overrides import apply_overrides

    apply_overrides(data)
    _attach_drill_info_flags(data)
    return data


def _attach_drill_info_flags(data: Dict[str, Any]) -> None:
    """Synthesize info flags for the drill start/end so they appear in the
    flag list (click-to-seek); works for legacy sessions too. The end flag
    reflects any instructor override applied above."""
    dw = data.get("drill_window") or {}
    fps = float((data.get("video") or {}).get("frame_rate") or 30.0)
    flags = data.setdefault("flags", [])
    existing = {f.get("flag_id") for f in flags}
    start = dw.get("start_frame")
    end = dw.get("end_frame")
    if start is not None and "drill_start_info" not in existing:
        flags.insert(0, {
            "flag_id": "drill_start_info",
            "metric_id": None,
            "linked_item_id": None,
            "type": "drill_start",
            "severity": "info",
            "frame": int(start),
            "time_sec": dw.get("start_time_sec", (int(start) - 1) / fps),
            "title": f"Drill start · frame {start}",
            "message": "Drill start detected from the tracker's first entry crossing.",
        })
    if end is not None and "drill_end_info" not in existing:
        adjusted = dw.get("source") == "instructor"
        uncertain = bool(dw.get("end_uncertain"))
        suffix = " (instructor-adjusted)" if adjusted else (" (uncertain)" if uncertain else "")
        flags.insert(1 if start is not None else 0, {
            "flag_id": "drill_end_info",
            "metric_id": None,
            "linked_item_id": None,
            "type": "drill_end",
            "severity": "info",
            "frame": int(end),
            "time_sec": dw.get("end_time_sec", int(end) / fps),
            "title": f"Drill end · frame {end}{suffix}",
            "message": (
                "Instructor-adjusted drill end; window-dependent metrics were recomputed."
                if adjusted else
                "Drill end located from the 'room clear' callout in the audio."
            ),
        })


def _attach_run_info(json_path: str, data: Dict[str, Any]) -> None:
    """Lift the run-folder ``RunInfo.json`` to the top level under ``run_info``.

    Also exposes ``run_dir`` so the Compare tab knows the run's location.
    """
    run_dir = os.path.dirname(os.path.abspath(json_path))
    manifest = os.path.join(run_dir, "RunInfo.json")
    if not os.path.isfile(manifest):
        return
    try:
        with open(manifest, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(payload, dict):
        data["run_info"] = payload
        data["run_dir"] = run_dir


def collect_whitelist_paths(data: Dict[str, Any], json_path: str) -> List[str]:
    """Every file path the session is allowed to expose via /video and /image.

    With the cleaned v2 artifact shape every leaf string under ``artifacts``
    is a path, so we collect them all directly.
    """
    json_dir = os.path.dirname(os.path.abspath(json_path))
    paths: List[str] = []

    def add(value: Any) -> None:
        if not isinstance(value, str) or not value:
            return
        candidate = value if os.path.isabs(value) else os.path.join(json_dir, value)
        if os.path.exists(candidate):
            paths.append(candidate)

    add(data.get("resolved_video_path"))
    add((data.get("video") or {}).get("video_path"))
    _walk_strings(data.get("artifacts"), add)

    basename = (data.get("session") or {}).get("video_basename") or ""
    for suffix in ("_DrillWindow.json", "_Transcription.json"):
        candidate = os.path.join(json_dir, f"{basename}{suffix}")
        if os.path.isfile(candidate):
            paths.append(candidate)

    return paths


# --- internal helpers -------------------------------------------------------


def _resolve_video_path(json_path: str, data: Dict[str, Any]) -> Optional[str]:
    for candidate in _candidate_video_paths(json_path, data):
        if os.path.isfile(candidate):
            return candidate
    return None


def _candidate_video_paths(json_path: str, data: Dict[str, Any]) -> List[str]:
    json_dir = os.path.dirname(os.path.abspath(json_path))
    out: List[str] = []
    seen = set()

    def _push(p: Optional[str]) -> None:
        if not isinstance(p, str) or not p.strip():
            return
        norm = os.path.abspath(os.path.expanduser(p))
        if norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    video_info = data.get("video") or {}
    session_info = data.get("session") or {}
    artifacts = data.get("artifacts") or {}

    declared = video_info.get("video_path")
    if isinstance(declared, str) and declared.strip():
        _push(declared)
        _push(os.path.join(json_dir, declared))

    basename = session_info.get("video_basename")
    if isinstance(basename, str) and basename.strip():
        for ext in SUPPORTED_VIDEO_EXTENSIONS:
            _push(os.path.join(json_dir, f"{basename}{ext}"))

        tracker_path = ((artifacts.get("data") or {}).get("tracker_output")) or artifacts.get("tracker_output_json")
        if isinstance(tracker_path, dict):
            tracker_path = tracker_path.get("path")
        if isinstance(tracker_path, str) and tracker_path.strip():
            tracker_dir = os.path.dirname(tracker_path)
            if tracker_dir:
                for ext in SUPPORTED_VIDEO_EXTENSIONS:
                    _push(os.path.join(tracker_dir, f"{basename}{ext}"))

    return out


def _attach_transcription_sidecar(json_path: str, data: Dict[str, Any]) -> None:
    """Lift the {basename}_Transcription.json sidecar to the top level."""
    basename = (data.get("session") or {}).get("video_basename")
    if not isinstance(basename, str) or not basename:
        return
    sidecar = os.path.join(os.path.dirname(json_path), f"{basename}_Transcription.json")
    if not os.path.isfile(sidecar):
        return
    try:
        with open(sidecar, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    segments = payload.get("segments")
    if not isinstance(segments, list):
        return

    data["transcription"] = {
        "language": payload.get("language"),
        "model": payload.get("model"),
        "aligned": bool(payload.get("aligned", False)),
        "schema_version": payload.get("schema_version"),
        "audio_window": payload.get("audio_window"),
        "segments": segments,
    }


def _attach_drill_window(json_path: str, data: Dict[str, Any]) -> None:
    """Ensure drill_window lives at the top level (sidecar fallback)."""
    embedded = data.get("drill_window")
    if isinstance(embedded, dict) and embedded:
        return
    basename = (data.get("session") or {}).get("video_basename")
    if not isinstance(basename, str) or not basename:
        return
    sidecar = os.path.join(os.path.dirname(json_path), f"{basename}_DrillWindow.json")
    if not os.path.isfile(sidecar):
        return
    try:
        with open(sidecar, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return
    if isinstance(payload, dict):
        data["drill_window"] = payload


def _walk_strings(node: Any, add) -> None:
    """Walk a tree and call ``add`` on every leaf string. Used for collecting
    artifact paths under the new structure where every leaf string IS a path."""
    if isinstance(node, dict):
        for value in node.values():
            _walk_strings(value, add)
    elif isinstance(node, list):
        for item in node:
            _walk_strings(item, add)
    elif isinstance(node, str):
        add(node)
