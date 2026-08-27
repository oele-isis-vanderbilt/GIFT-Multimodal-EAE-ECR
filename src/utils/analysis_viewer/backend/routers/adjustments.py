"""Instructor-adjustment endpoints — flag delete/restore, POD-frame
override, and drill-end override.

All four mutate the instructor-owned ``{basename}_AnalysisOverrides.json``
sidecar next to the run's Analysis.json (the engine output itself is never
touched) and return the freshly re-merged session payload, so the frontend
simply replaces its session store with the response.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..config import enforce_under_root
from ..loader import SessionLoadError, collect_whitelist_paths, load_session
from ..overrides import load_overrides, save_overrides
from ..security import WHITELIST

router = APIRouter()


class FlagRequest(BaseModel):
    path: str
    flag_id: str


class PodFrameRequest(BaseModel):
    path: str
    frame: Optional[int] = None  # None resets to the engine's automatic result


class DrillEndRequest(BaseModel):
    path: str
    frame: Optional[int] = None  # None resets to the engine's detected end


def _reload(path: str) -> Dict[str, Any]:
    try:
        data = load_session(path)
    except SessionLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    json_path = data["session_json_path"]
    WHITELIST.register(json_path, collect_whitelist_paths(data, json_path),
                       root_dir=os.path.dirname(json_path))
    return data


def _validated_session_json_path(path: str) -> str:
    """Resolve AND validate the target before any sidecar write.

    Loading the session up front (schema check included) means a bogus
    ``path`` can never cause an ``_AnalysisOverrides.json`` to be created
    next to an arbitrary file on disk.
    """
    try:
        data = load_session(path)
    except SessionLoadError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return data["session_json_path"]


@router.post("/flags/delete")
def delete_flag(req: FlagRequest) -> Dict[str, Any]:
    path = enforce_under_root(req.path)
    json_path = _validated_session_json_path(path)
    overrides = load_overrides(json_path)
    if req.flag_id not in overrides["deleted_flag_ids"]:
        overrides["deleted_flag_ids"].append(req.flag_id)
        save_overrides(json_path, overrides)
    return _reload(path)


@router.post("/flags/restore")
def restore_flag(req: FlagRequest) -> Dict[str, Any]:
    path = enforce_under_root(req.path)
    json_path = _validated_session_json_path(path)
    overrides = load_overrides(json_path)
    if req.flag_id in overrides["deleted_flag_ids"]:
        overrides["deleted_flag_ids"].remove(req.flag_id)
        save_overrides(json_path, overrides)
    return _reload(path)


@router.post("/pod/frame")
def set_pod_frame(req: PodFrameRequest) -> Dict[str, Any]:
    path = enforce_under_root(req.path)
    json_path = _validated_session_json_path(path)
    if req.frame is not None and req.frame < 1:
        raise HTTPException(status_code=400, detail="frame must be >= 1")

    # Validate the recompute BEFORE persisting anything, so a failure never
    # leaves a broken override on disk (state divergence + a failed recompute
    # on every subsequent load).
    effective_frame: Optional[int] = None
    if req.frame is not None:
        from src.orientation import recompute_pod_for_run

        run_dir = os.path.dirname(json_path)
        try:
            pod_data = recompute_pod_for_run(run_dir, int(req.frame))
        except (KeyboardInterrupt, SystemExit) as exc:
            raise HTTPException(status_code=422,
                                detail="POD recompute aborted") from exc
        except Exception as exc:
            raise HTTPException(status_code=422,
                                detail=f"POD recompute failed: {exc}") from exc
        # Persist the EFFECTIVE frame: recompute clamps to the processed
        # range, and a sidecar holding a different value than what is applied
        # would silently diverge (the drill-end path notices its clamp; here
        # the clamped value simply becomes the override).
        effective_frame = int(pod_data.get("pod_frame") or int(req.frame))

    overrides = load_overrides(json_path)
    overrides["pod_frame_override"] = effective_frame
    save_overrides(json_path, overrides)
    return _reload(path)


@router.post("/drill/end")
def set_drill_end(req: DrillEndRequest) -> Dict[str, Any]:
    """Instructor-set drill end. Validates end > start and dry-runs the full
    recompute BEFORE persisting, so a failure never leaves a broken override."""
    path = enforce_under_root(req.path)
    json_path = _validated_session_json_path(path)

    if req.frame is not None:
        from ..recompute import apply_drill_end_to_payload

        # Dry-run against a fresh copy of the session (validates the frame
        # against the drill start and exercises the metric recompute).
        try:
            probe = load_session(path)
        except SessionLoadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            apply_drill_end_to_payload(probe, os.path.dirname(json_path), int(req.frame))
        except (KeyboardInterrupt, SystemExit) as exc:
            raise HTTPException(status_code=422,
                                detail="drill-end recompute aborted") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=422,
                                detail=f"drill-end recompute failed: {exc}") from exc

    overrides = load_overrides(json_path)
    overrides["drill_end_override"] = int(req.frame) if req.frame is not None else None
    save_overrides(json_path, overrides)
    return _reload(path)
