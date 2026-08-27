"""Instructor-owned analysis overrides.

The engine's ``{basename}_Analysis.json`` is immutable output; instructor
adjustments made in the viewer are layered on top from a sibling sidecar,
``{basename}_AnalysisOverrides.json``:

    {
      "schema_version": "1.0",
      "pod_frame_override": 671 | null,   // instructor-set POD frame
      "drill_end_override": 1830 | null,  // instructor-adjusted drill end
      "deleted_flag_ids": ["pod_flagging_2_3_f671", ...]
    }

(pod_flagging ids are frame-scoped: ``pod_flagging_{from}_{to}_f{pod_frame}``.)

``apply_overrides`` merges the sidecar into a freshly-loaded session payload,
in order:

- ``drill_end_override`` re-runs every window-dependent metric from the run
  caches with the adjusted end frame and re-detects the POD inside the new
  window.
- ``pod_frame_override`` recomputes the whole POD block (members, sectors,
  both scores, facing flags) at the instructor's frame from the run-folder
  caches (no video / pose model) and replaces the engine's block.
- ``deleted_flag_ids`` moves matching flags into a ``flag_bin`` list (so the
  viewer can restore them) and re-derives each affected metric's score with
  the binned violations ignored.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

OVERRIDES_SCHEMA_VERSION = "1.0"
_SUFFIX = "_AnalysisOverrides.json"


def overrides_path_for(session_json_path: str) -> str:
    """`.../<base>_Analysis.json` -> `.../<base>_AnalysisOverrides.json`."""
    folder = os.path.dirname(os.path.abspath(session_json_path))
    name = os.path.basename(session_json_path)
    base = name[: -len("_Analysis.json")] if name.endswith("_Analysis.json") else os.path.splitext(name)[0]
    return os.path.join(folder, f"{base}{_SUFFIX}")


def load_overrides(session_json_path: str) -> Dict[str, Any]:
    path = overrides_path_for(session_json_path)
    if not os.path.isfile(path):
        return {"schema_version": OVERRIDES_SCHEMA_VERSION,
                "pod_frame_override": None, "deleted_flag_ids": []}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {"schema_version": OVERRIDES_SCHEMA_VERSION,
                "pod_frame_override": None, "deleted_flag_ids": []}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("schema_version", OVERRIDES_SCHEMA_VERSION)
    # The sidecar is hand-editable — coerce defensively so a bad value can
    # never make session loading fail.
    for key in ("pod_frame_override", "drill_end_override"):
        try:
            v = data.get(key)
            data[key] = int(v) if v is not None else None
            if data[key] is not None and data[key] < 1:
                data[key] = None
        except (TypeError, ValueError):
            data[key] = None
    ids = data.get("deleted_flag_ids")
    data["deleted_flag_ids"] = [str(x) for x in ids] if isinstance(ids, list) else []
    return data


def save_overrides(session_json_path: str, overrides: Dict[str, Any]) -> str:
    path = overrides_path_for(session_json_path)
    payload = {
        "schema_version": OVERRIDES_SCHEMA_VERSION,
        "pod_frame_override": overrides.get("pod_frame_override"),
        "drill_end_override": overrides.get("drill_end_override"),
        "deleted_flag_ids": list(overrides.get("deleted_flag_ids") or []),
    }
    # Atomic write (temp + rename) so a concurrent load can never observe a
    # torn file — a half-written sidecar would silently read as "no overrides".
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)
    return path


def apply_overrides(data: Dict[str, Any]) -> None:
    """Merge the overrides sidecar into an enriched session payload in place.

    Always leaves ``data['flag_bin']`` (possibly empty) and
    ``data['overrides']`` (the effective sidecar) so the frontend has a
    stable contract.
    """
    session_json_path = data.get("session_json_path") or ""
    overrides = load_overrides(session_json_path) if session_json_path else {
        "schema_version": OVERRIDES_SCHEMA_VERSION,
        "pod_frame_override": None, "drill_end_override": None,
        "deleted_flag_ids": [],
    }
    data["overrides"] = overrides
    overrides.setdefault("notices", [])
    data["flag_bin"] = []

    # Order matters: (1) the drill-end override re-runs the window-dependent
    # metrics and re-detects POD; (2) an explicit instructor POD frame then
    # replaces the POD block; (3) flag deletions recalc on top of the result.
    drill_end = overrides.get("drill_end_override")
    if drill_end is not None:
        _apply_drill_end_override(data, int(drill_end))

    pod_frame = overrides.get("pod_frame_override")
    if pod_frame is not None:
        eff_end = ((data.get("drill_window") or {}).get("end_frame"))
        if drill_end is not None and eff_end is not None and int(pod_frame) > int(eff_end):
            overrides["notices"].append(
                f"The instructor-set POD frame ({pod_frame}) is beyond the adjusted "
                f"drill end ({eff_end}) and was ignored — set the POD again inside "
                "the new window."
            )
        else:
            _apply_pod_frame_override(data, int(pod_frame))

    deleted: List[str] = overrides.get("deleted_flag_ids") or []
    if deleted:
        _apply_deleted_flags(data, deleted)


def _apply_drill_end_override(data: Dict[str, Any], end_frame: int) -> None:
    from .recompute import apply_drill_end_to_payload

    run_dir = data.get("run_dir") or os.path.dirname(data.get("session_json_path") or "")
    try:
        notices = apply_drill_end_to_payload(data, run_dir, end_frame)
    except (KeyboardInterrupt, SystemExit):
        _record_override_failure(data, "drill_recompute_error",
                                 "drill end", end_frame, "recompute aborted")
        return
    except Exception as exc:
        _record_override_failure(data, "drill_recompute_error",
                                 "drill end", end_frame, str(exc))
        return
    data.setdefault("overrides", {}).setdefault("notices", []).extend(notices)


def _apply_pod_frame_override(data: Dict[str, Any], pod_frame: int) -> None:
    from src.analysis import attach_pod_block
    from src.orientation import recompute_pod_for_run

    run_dir = data.get("run_dir") or os.path.dirname(data.get("session_json_path") or "")
    try:
        pod_data = recompute_pod_for_run(run_dir, pod_frame)
    except (KeyboardInterrupt, SystemExit):
        # Never let a BaseException from library code brick session loading.
        _record_override_failure(data, "pod_recompute_error",
                                 "POD frame", pod_frame, "recompute aborted")
        return
    except Exception as exc:  # cache missing / config unresolvable
        _record_override_failure(data, "pod_recompute_error",
                                 "POD frame", pod_frame, str(exc))
        return
    attach_pod_block(data, pod_data)


def _record_override_failure(data: Dict[str, Any], key: str, label: str,
                             value: int, error: str) -> None:
    """A persisted override that fails to recompute must never be silent:
    the session falls back to the engine's original results, so surface that
    in the notices the viewer actually renders (the error field is kept for
    API consumers)."""
    ov = data.setdefault("overrides", {})
    ov[key] = error
    ov.setdefault("notices", []).append(
        f"The saved instructor {label} override ({value}) could not be "
        f"re-applied ({error}) — the engine's original results are shown. "
        "Reset the override or restore the run caches."
    )


def _apply_deleted_flags(data: Dict[str, Any], deleted_ids: List[str]) -> None:
    from src.analysis import apply_flag_deletion_recalcs

    deleted_set = set(deleted_ids)
    kept, binned = [], []
    for flag in data.get("flags") or []:
        (binned if flag.get("flag_id") in deleted_set else kept).append(flag)
    data["flags"] = kept
    data["flag_bin"] = binned

    # Scrub dangling flag references so the timeline/metric cross-refs stay
    # consistent with the effective flag list.
    for item in (data.get("timeline") or {}).get("items", []):
        item["flag_ids"] = [f for f in (item.get("flag_ids") or []) if f not in deleted_set]
    for metric in data.get("metrics") or []:
        metric["flag_ids"] = [f for f in (metric.get("flag_ids") or []) if f not in deleted_set]

    # Every affected metric's score is re-derived with the binned violations
    # ignored (each metric has its own ignore semantics — see analysis.py).
    apply_flag_deletion_recalcs(data, binned)
