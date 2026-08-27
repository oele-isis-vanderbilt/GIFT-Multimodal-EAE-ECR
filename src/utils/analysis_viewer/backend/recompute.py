"""Instructor drill-end adjustment: recompute metrics from run-folder caches.

When the instructor moves the drill END, every window-dependent result must
follow. This module rebuilds a ``MetricContext`` purely from the run's cached
artifacts (TrackerOutput / PositionCache / OrientationCache + the room
config) and runs the REAL metric classes over it — the same code paths the
engine used live, so there is no duplicated scoring math:

- the four entry metrics re-run (STAY_ALONG_WALL with its drill_start=1
  special case, emitting fresh excursion flags for the new window);
- wall flags + wall_excursion timeline items are rebuilt from the metric's
  own flag payloads;
- automatic POD detection re-runs bounded by the new end (via
  ``src.orientation.auto_pod_from_caches``).

No video and no pose model are touched; artifact videos keep their original
trims (they are baked files) — only scores, flags, items, and the
drill_window block change.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from src.metrics._shared import pick_latest


def _build_context(run_folder: str, drill_start: int, drill_end: int):
    from libs.Track.processing.utils import load_pixel_mapper
    from src.metrics.context import MetricContext
    from src.orientation import load_run_pod_inputs

    inputs = load_run_pod_inputs(run_folder)
    config = inputs["config"]
    config = dict(config)  # shallow copy — we inject runtime keys
    config["frame_rate"] = inputs["fps"]

    tracker_path = pick_latest(run_folder, "*_TrackerOutput.json")
    if not tracker_path:
        raise ValueError("No *_TrackerOutput.json in run folder")
    with open(tracker_path) as f:
        tracker_output = json.load(f)

    bbox_details: Dict[tuple, tuple] = {}
    keypoint_details: Dict[tuple, Any] = {}
    for entry in tracker_output:
        fidx = entry["frame"]
        for obj in entry.get("objects", []):
            tid = obj["id"]
            if obj.get("bbox"):
                bbox_details[(fidx, tid)] = tuple(obj["bbox"])
            keypoint_details[(fidx, tid)] = (
                obj.get("keypoints") or [],
                obj.get("keypoint_scores") or [],
            )

    mapper = load_pixel_mapper(config["point_mapping_path"])
    ctx = MetricContext(
        tracker_output=tracker_output,
        tracks_by_id=inputs["tracks"],
        bbox_details=bbox_details,
        keypoint_details=keypoint_details,
        inroom_ids=sorted(inputs["inroom"]),
        drill_start_frame=int(drill_start),
        drill_end_frame=int(drill_end),
        pixel_mapper=mapper,
    )
    return ctx, config, inputs


def recompute_entry_metrics(run_folder: str, drill_start: int,
                            drill_end: int) -> Dict[str, Any]:
    """Run the four enabled entry metrics from caches over an arbitrary
    drill window. Returns scores, fresh wall flag payloads, and the wall
    per-entrant summary."""
    from src.metrics import (
        EntranceHesitation_Metric,
        EntranceVectors_Metric,
        MoveAlongWall_Metric,
        TotalEntryTime_Metric,
    )

    ctx, config, inputs = _build_context(run_folder, drill_start, drill_end)
    scores: Dict[str, float] = {}
    wall_metric = None
    for cls in (EntranceVectors_Metric, EntranceHesitation_Metric,
                TotalEntryTime_Metric, MoveAlongWall_Metric):
        m = cls(config)
        if getattr(m, "metricName", "") == "STAY_ALONG_WALL":
            saved = ctx.drill_start_frame
            ctx.drill_start_frame = 1
            try:
                m.process(ctx)
            finally:
                ctx.drill_start_frame = saved
            wall_metric = m
        else:
            m.process(ctx)
        scores[m.metricName] = float(m.getFinalScore())

    return {
        "scores": scores,
        "wall_flag_payloads": [
            f for f in ctx.metric_flags
            if str(f.get("type", "")) in ("wall_too_close", "wall_too_far")
        ],
        "wall_per_entrant": list(getattr(wall_metric, "_per_entrant_summary", []) or []),
        "config": config,
        "fps": inputs["fps"],
        "total": inputs["total"],
    }


_V2_METRIC_BY_NAME = {
    "ENTRANCE_VECTORS": "entrance_vectors",
    "ENTRANCE_HESITATION": "entrance_hesitation",
    "TOTAL_TIME_OF_ENTRY": "total_time_of_entry",
    "STAY_ALONG_WALL": "move_along_wall",
}


def apply_drill_end_to_payload(data: Dict[str, Any], run_folder: str,
                               end_frame: int) -> List[str]:
    """Mutate an enriched session payload for an instructor-set drill end.

    Returns human-readable notices (e.g. the POD-invalidated warning).
    """
    from src.analysis import _attach_wall_band_polygons, attach_pod_block
    from src.orientation import auto_pod_from_caches

    notices: List[str] = []
    dw = data.get("drill_window") or {}
    drill_start = int(dw.get("start_frame") or 1)
    end_frame = int(end_frame)
    if end_frame <= drill_start:
        raise ValueError(
            f"drill end ({end_frame}) must be after the drill start ({drill_start})")

    prev_pod_frame = None
    for m in data.get("metrics") or []:
        if m.get("metric_id") == "pod_sector_coverage":
            prev_pod_frame = (m.get("summary") or {}).get("pod_frame")

    result = recompute_entry_metrics(run_folder, drill_start, end_frame)
    fps = float(result["fps"] or 30.0)

    # The engine stops processing shortly after the detected drill end
    # (stop_at_drill_end), so pose/tracking/audio caches do not extend beyond
    # the last processed frame. An end beyond that cannot be scored — clamp
    # it and tell the instructor how to actually extend the analysis.
    data_end = int(result["total"])
    if end_frame > data_end:
        notices.append(
            f"The requested drill end (frame {end_frame}) is beyond the last "
            f"processed frame ({data_end}) — the engine stops shortly after the "
            "detected drill end, so no pose/tracking data exists past that point. "
            f"The end was clamped to frame {data_end}. To analyze further into the "
            "video, re-run the engine with stop_at_drill_end disabled."
        )
        end_frame = data_end

    # --- drill window block -------------------------------------------------
    dw = dict(dw)
    dw["end_frame"] = end_frame
    dw["end_time_sec"] = end_frame / fps
    dw["end_uncertain"] = False
    dw["decision_reason"] = "instructor_override"
    dw["source"] = "instructor"
    data["drill_window"] = dw

    # --- entry metric scores ------------------------------------------------
    metrics_by_id = {m.get("metric_id"): m for m in data.get("metrics") or []}
    for name, score in result["scores"].items():
        metric = metrics_by_id.get(_V2_METRIC_BY_NAME.get(name, ""))
        if metric is not None:
            metric["score"] = round(score, 2)

    # --- wall flags + excursion items rebuilt for the new window ------------
    entry_number_by_track: Dict[int, int] = {}
    items = (data.get("timeline") or {}).get("items", [])
    for it in items:
        if it.get("kind") == "entry":
            d = it.get("data") or {}
            if d.get("track_id") is not None and d.get("entry_number") is not None:
                entry_number_by_track[int(d["track_id"])] = int(d["entry_number"])

    new_flags: List[Dict[str, Any]] = []
    new_items: List[Dict[str, Any]] = []
    for payload in result["wall_flag_payloads"]:
        rec = payload.get("_wall_excursion_record") or {}
        tid = int(payload.get("track_id", -1))
        item_id = f"wall_excursion_{tid}_{int(rec.get('n', len(new_items) + 1))}"
        flag = {
            "flag_id": payload.get("flag_id"),
            "metric_id": "move_along_wall",
            "linked_item_id": item_id,
            "type": payload.get("type"),
            "severity": payload.get("severity"),
            "frame": payload.get("frame"),
            "time_sec": payload.get("time_sec"),
            "title": payload.get("title"),
            "message": payload.get("message"),
            "track_id": tid,
            "start_frame": payload.get("start_frame"),
            "end_frame": payload.get("end_frame"),
            "start_time_sec": payload.get("start_time_sec"),
            "end_time_sec": payload.get("end_time_sec"),
        }
        new_flags.append(flag)
        new_items.append({
            "item_id": item_id,
            "metric_id": "move_along_wall",
            "kind": "wall_excursion",
            "label": (f"Track {tid}: "
                      f"{'too close' if payload.get('type') == 'wall_too_close' else 'too far'}"),
            "flag_ids": [str(payload.get("flag_id"))],
            "frame": int(payload.get("start_frame") or 0),
            "time_sec": payload.get("start_time_sec"),
            "start_frame": int(payload.get("start_frame") or 0),
            "end_frame": int(payload.get("end_frame") or 0),
            "start_time_sec": payload.get("start_time_sec"),
            "end_time_sec": payload.get("end_time_sec"),
            "data": {
                "track_id": tid,
                "entry_number": entry_number_by_track.get(tid),
                "label_kind": ("too_close" if payload.get("type") == "wall_too_close"
                               else "too_far"),
                "duration_sec": float(payload.get("duration_sec") or 0.0),
                "L_map": float(rec.get("L_map") or 0.0),
            },
        })

    data["flags"] = [
        f for f in data.get("flags") or [] if f.get("metric_id") != "move_along_wall"
    ] + new_flags
    data.setdefault("timeline", {})["items"] = [
        it for it in items if it.get("kind") != "wall_excursion"
    ] + new_items

    # wall metric summary + per-entrant band polygons
    wall_metric = metrics_by_id.get("move_along_wall")
    if wall_metric is not None:
        per_entrant = [
            {k: v for k, v in e.items() if not str(k).startswith("_")}
            for e in result["wall_per_entrant"]
        ]
        _attach_wall_band_polygons(per_entrant, result["config"].get("Boundary"))
        summary = wall_metric.setdefault("summary", {})
        summary["per_entrant"] = per_entrant
        summary["excursion_count"] = len(new_flags)
        summary["total_too_close_time_sec"] = round(sum(
            float(e.get("too_close_time_sec") or 0.0) for e in per_entrant), 3)
        summary["total_too_far_time_sec"] = round(sum(
            float(e.get("too_far_time_sec") or 0.0) for e in per_entrant), 3)
        wall_metric["flag_ids"] = [f["flag_id"] for f in new_flags]
        wall_metric["timeline_item_ids"] = [it["item_id"] for it in new_items]

    # --- POD re-detection bounded by the new end ----------------------------
    pod_data = auto_pod_from_caches(run_folder, drill_start, end_frame)
    attach_pod_block(data, pod_data)
    if prev_pod_frame is not None and end_frame < int(prev_pod_frame):
        new_pod = pod_data.get("pod_frame")
        notices.append(
            f"The new drill end (frame {end_frame}) is before the previously "
            f"established POD (frame {prev_pod_frame}). POD was re-detected inside "
            f"the new window — now "
            f"{'frame ' + str(new_pod) if new_pod is not None else 'NOT established (uncertain)'}. "
            "Review or adjust the POD mark."
        )
    return notices
