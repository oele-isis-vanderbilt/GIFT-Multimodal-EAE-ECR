
import numpy as np
import cv2
from . import AbstractMetric
import os
import glob
import json
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
# --------------------------------------------------------------------
#  Helpers for gaze-cone construction and intersection
# --------------------------------------------------------------------
def _gaze_triangle(origin, direction, half_angle_deg, length=10000.0):
    """Return a 3-point gaze-cone triangle (origin, left, right)."""
    d = np.asarray(direction, dtype=np.float32)
    n = np.linalg.norm(d)
    if n == 0.0:
        return np.zeros((3, 2), dtype=np.float32)
    d /= n

    ang = np.deg2rad(float(half_angle_deg))
    cos_a, sin_a = float(np.cos(ang)), float(np.sin(ang))
    rot_left = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
    rot_right = np.array([[cos_a, sin_a], [-sin_a, cos_a]], dtype=np.float32)

    o = np.asarray(origin, dtype=np.float32)
    left_vec = rot_left @ d * float(length)
    right_vec = rot_right @ d * float(length)
    return np.stack([o, o + left_vec, o + right_vec], axis=0)


def _triangle_box_intersect(triangle, box):
    """Return True if the triangle intersects the axis-aligned bbox."""
    tri = np.asarray(triangle, dtype=np.float32).reshape(-1, 1, 2)
    x1, y1, x2, y2 = map(float, box)
    rect = (
        np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        .reshape(-1, 1, 2)
    )
    inter_area, _ = cv2.intersectConvexConvex(tri, rect)
    return inter_area > 0.0


class TeammateCoverage_Metric(AbstractMetric):
    def __init__(self, config):
        super().__init__(config)
        self.metricName = "TEAMMATE_COVERAGE"
        self.coverage_angle = float(config.get("visual_angle_degrees", 20.0))
        self._final_score = 0.0

    def process(self, ctx):
        # Support multiple enemies
        enemy_ids = self.config.get("enemy_ids", [99])
        trainee_ids = [tid for tid in ctx.tracks_by_id.keys() if tid not in enemy_ids]

        present_frames = {tid: 0 for tid in trainee_ids}
        unseen_frames = {tid: 0 for tid in trainee_ids}

        for frame_idx in range(1, len(ctx.all_frames) + 1):
            for tid in trainee_ids:
                # Check if soldier is present in this frame
                if (frame_idx, tid) not in ctx.bbox_details:
                    continue
                present_frames[tid] += 1
                bbox = ctx.bbox_details[(frame_idx, tid)]
                seen = False
                for viewer_id in trainee_ids:
                    if viewer_id == tid:
                        continue
                    if (frame_idx, viewer_id) not in ctx.gaze_info:
                        continue
                    ox, oy, dx, dy = ctx.gaze_info[(frame_idx, viewer_id)]
                    origin = np.array([ox, oy])
                    direction = np.array([dx, dy])
                    # --- Triangle‑based gaze check ---
                    half_angle = self.coverage_angle / 2.0
                    if np.linalg.norm(direction) == 0.0:
                        seen = True
                        break
                    tri = _gaze_triangle(origin, direction, half_angle)
                    if _triangle_box_intersect(tri, bbox):
                        seen = True
                        break
                if not seen:
                    unseen_frames[tid] += 1
            
        total_unseen = sum(unseen_frames.values())
        total_present = sum(present_frames.values())
        if total_present == 0:
            self._final_score = 0.0
        else:
            self._final_score = max(0.0, 1.0 - (total_unseen / total_present))

    def getFinalScore(self) -> float:
        return round(self._final_score, 2)

    # ---------------------------------------------------------------------
    # Expert comparison (folder-based)
    # ---------------------------------------------------------------------
    @staticmethod
    def expertCompare(session_folder: str, expert_folder: str, _map_image=None, config=None):
        """Compare trainee vs expert teammate coverage.

        Recomputes coverage from `*_TrackerOutput.json` and `*_GazeCache.txt`, aligned to
        each run's first non-enemy entry. Writes `TEAMMATE_COVERAGE_Comparison.jpg`.
        """

        def _pick_latest(folder: str, patterns: List[str]) -> Optional[str]:
            matches: List[str] = []
            for pat in patterns:
                matches.extend(glob.glob(os.path.join(folder, pat)))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_tracker_output(folder: str) -> Tuple[Dict[Tuple[int, int], Tuple[float, float, float, float]], int, List[int]]:
            path = _pick_latest(folder, ["*_TrackerOutput.json", "*TrackerOutput.json"])
            if path is None:
                raise FileNotFoundError(f"No TrackerOutput found in {folder}")
            with open(path, "r") as f:
                data = json.load(f)
            bbox_details: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
            max_frame = 0
            ids = set()
            for entry in data:
                frame_idx = int(entry.get("frame", 0))
                max_frame = max(max_frame, frame_idx)
                for obj in entry.get("objects", []) or []:
                    tid = obj.get("id")
                    bbox = obj.get("bbox")
                    if tid is None or bbox is None or len(bbox) != 4:
                        continue
                    tid_i = int(tid)
                    ids.add(tid_i)
                    x1, y1, x2, y2 = bbox
                    bbox_details[(frame_idx, tid_i)] = (float(x1), float(y1), float(x2), float(y2))
            return bbox_details, max_frame, sorted(ids)

        def _load_gaze_cache(folder: str) -> Dict[Tuple[int, int], Tuple[float, float, float, float]]:
            path = _pick_latest(folder, ["*_GazeCache.txt", "*GazeCache.txt"])
            if path is None:
                return {}
            gaze: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
            with open(path, "r") as f:
                # expected: frame,id,ox,oy,dx,dy
                header = f.readline()
                for line in f:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 6:
                        continue
                    try:
                        frm = int(float(parts[0]))
                        tid = int(float(parts[1]))
                        ox = float(parts[2]); oy = float(parts[3]); dx = float(parts[4]); dy = float(parts[5])
                        gaze[(frm, tid)] = (ox, oy, dx, dy)
                    except Exception:
                        continue
            return gaze

        def _entry_order_from_bboxes(bbox_details: Dict[Tuple[int, int], Tuple[float, float, float, float]],
                                    ids: List[int],
                                    enemy_ids: List[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
            """Return (first_frame_by_id, entry_number_by_id) for non-enemy IDs."""
            first_frame: Dict[int, int] = {}
            for tid in ids:
                if tid in enemy_ids:
                    continue
                # find min frame where bbox exists
                frames = [f for (f, t) in bbox_details.keys() if t == tid]
                if frames:
                    first_frame[tid] = int(min(frames))
            # sort by first frame
            ordered = sorted(first_frame.items(), key=lambda kv: kv[1])
            entry_num: Dict[int, int] = {tid: i + 1 for i, (tid, _) in enumerate(ordered)}
            return first_frame, entry_num


        def _compute(folder: str, *, enemy_ids: List[int], fps: float, coverage_angle: float):
            bbox_details, max_frame, ids = _load_tracker_output(folder)
            gaze_info = _load_gaze_cache(folder)

            trainee_ids = [tid for tid in ids if tid not in enemy_ids]
            if not trainee_ids or max_frame <= 0:
                return {
                    "fps": fps,
                    "first_entry_frame": None,
                    "trainee_ids": [],
                    "present_frames": {},
                    "unseen_frames": {},
                    "final_score": 0.0,
                    "series": [],
                    "bbox_details": bbox_details,
                    "gaze_info": gaze_info,
                    "ids": ids,
                }

            # Determine first team entry frame (first frame any trainee has a bbox)
            first_entry_frame = None
            for f in range(1, max_frame + 1):
                if any((f, tid) in bbox_details for tid in trainee_ids):
                    first_entry_frame = f
                    break

            present_frames = {tid: 0 for tid in trainee_ids}
            unseen_frames = {tid: 0 for tid in trainee_ids}

            # For time-series: cumulative present/unseen across all trainees
            cum_present = 0
            cum_unseen = 0
            series: List[Tuple[float, float]] = []

            half_angle = float(coverage_angle) / 2.0

            if first_entry_frame is None:
                return {
                    "fps": fps,
                    "first_entry_frame": None,
                    "trainee_ids": trainee_ids,
                    "present_frames": present_frames,
                    "unseen_frames": unseen_frames,
                    "final_score": 0.0,
                    "series": [],
                    "bbox_details": bbox_details,
                    "gaze_info": gaze_info,
                    "ids": ids,
                }

            # Iterate frames from first team entry
            for frame_idx in range(first_entry_frame, max_frame + 1):
                frame_present = 0
                frame_unseen = 0

                for tid in trainee_ids:
                    if (frame_idx, tid) not in bbox_details:
                        continue

                    present_frames[tid] += 1
                    frame_present += 1
                    bbox = bbox_details[(frame_idx, tid)]

                    seen = False
                    for viewer_id in trainee_ids:
                        if viewer_id == tid:
                            continue
                        g = gaze_info.get((frame_idx, viewer_id))
                        if g is None:
                            continue
                        ox, oy, dx, dy = g
                        origin = np.array([ox, oy], dtype=float)
                        direction = np.array([dx, dy], dtype=float)
                        if np.linalg.norm(direction) == 0.0:
                            seen = True
                            break
                        tri = _gaze_triangle(origin, direction, half_angle)
                        if _triangle_box_intersect(tri, bbox):
                            seen = True
                            break

                    if not seen:
                        unseen_frames[tid] += 1
                        frame_unseen += 1

                cum_present += frame_present
                cum_unseen += frame_unseen

                # time in seconds since first team entry frame
                t_sec = (frame_idx - first_entry_frame) / fps if fps > 0 else float(frame_idx - first_entry_frame)
                score = 0.0 if cum_present == 0 else max(0.0, 1.0 - (cum_unseen / cum_present))
                series.append((float(t_sec), float(score)))

            total_present = sum(present_frames.values())
            total_unseen = sum(unseen_frames.values())
            final_score = 0.0 if total_present == 0 else max(0.0, 1.0 - (total_unseen / total_present))

            return {
                "fps": fps,
                "first_entry_frame": first_entry_frame,
                "trainee_ids": trainee_ids,
                "present_frames": present_frames,
                "unseen_frames": unseen_frames,
                "final_score": float(final_score),
                "series": series,
                "bbox_details": bbox_details,
                "gaze_info": gaze_info,
                "ids": ids,
            }

        # -------------------- run both folders --------------------
        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "TEAMMATE_COVERAGE_Comparison.jpg")
        txt_path = os.path.join(session_folder, "TEAMMATE_COVERAGE_Comparison.txt")

        enemy_ids = (config or {}).get("enemy_ids", [99]) if isinstance(config, dict) else [99]
        fps = float((config or {}).get("frame_rate", 30.0)) if isinstance(config, dict) else 30.0
        coverage_angle = float((config or {}).get("visual_angle_degrees", 20.0)) if isinstance(config, dict) else 20.0

        try:
            expert_res = _compute(expert_folder, enemy_ids=enemy_ids, fps=fps, coverage_angle=coverage_angle)
            trainee_res = _compute(session_folder, enemy_ids=enemy_ids, fps=fps, coverage_angle=coverage_angle)
        except Exception:
            err_text = "There was an error while processing this comparison. Missing or invalid TrackerOutput/GazeCache."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "TEAMMATE_COVERAGE",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Entry numbering (per folder) based on bbox appearance
        ex_bbox, ex_ids = expert_res.get("bbox_details", {}), expert_res.get("ids", [])
        tr_bbox, tr_ids = trainee_res.get("bbox_details", {}), trainee_res.get("ids", [])
        _, ex_entry_num = _entry_order_from_bboxes(ex_bbox, ex_ids, enemy_ids)
        _, tr_entry_num = _entry_order_from_bboxes(tr_bbox, tr_ids, enemy_ids)

        # all_entry_numbers = sorted(set(ex_entry_num.values()) | set(tr_entry_num.values()))
        all_entry_numbers = sorted(set(ex_entry_num.values()) | set(tr_entry_num.values()))

        def _id_for_entry(entry_map: Dict[int, int], n: int) -> Optional[int]:
            for tid, en in entry_map.items():
                if en == n:
                    return tid
            return None

        rows: List[Dict] = []
        for n in all_entry_numbers:
            ex_id = _id_for_entry(ex_entry_num, n)
            tr_id = _id_for_entry(tr_entry_num, n)

            ex_p = expert_res.get("present_frames", {}).get(ex_id, 0) if ex_id is not None else 0
            ex_u = expert_res.get("unseen_frames", {}).get(ex_id, 0) if ex_id is not None else 0
            tr_p = trainee_res.get("present_frames", {}).get(tr_id, 0) if tr_id is not None else 0
            tr_u = trainee_res.get("unseen_frames", {}).get(tr_id, 0) if tr_id is not None else 0

            ex_present_sec = (ex_p / fps) if fps > 0 else float(ex_p)
            ex_unseen_sec = (ex_u / fps) if fps > 0 else float(ex_u)
            tr_present_sec = (tr_p / fps) if fps > 0 else float(tr_p)
            tr_unseen_sec = (tr_u / fps) if fps > 0 else float(tr_u)

            ex_score = 0.0 if ex_p == 0 else max(0.0, 1.0 - (ex_u / ex_p))
            tr_score = 0.0 if tr_p == 0 else max(0.0, 1.0 - (tr_u / tr_p))

            rows.append({
                "entry_number": int(n),
                "expert_id": "" if ex_id is None else int(ex_id),
                "expert_present_sec": round(float(ex_present_sec), 3),
                "expert_unseen_sec": round(float(ex_unseen_sec), 3),
                "expert_score": round(float(ex_score), 3),
                "trainee_id": "" if tr_id is None else int(tr_id),
                "trainee_present_sec": round(float(tr_present_sec), 3),
                "trainee_unseen_sec": round(float(tr_unseen_sec), 3),
                "trainee_score": round(float(tr_score), 3),
            })

        # Add OVERALL row
        ex_total_p = sum((expert_res.get("present_frames") or {}).values())
        ex_total_u = sum((expert_res.get("unseen_frames") or {}).values())
        tr_total_p = sum((trainee_res.get("present_frames") or {}).values())
        tr_total_u = sum((trainee_res.get("unseen_frames") or {}).values())

        ex_total_present_sec = (ex_total_p / fps) if fps > 0 else float(ex_total_p)
        ex_total_unseen_sec = (ex_total_u / fps) if fps > 0 else float(ex_total_u)
        tr_total_present_sec = (tr_total_p / fps) if fps > 0 else float(tr_total_p)
        tr_total_unseen_sec = (tr_total_u / fps) if fps > 0 else float(tr_total_u)

        ex_final = float(expert_res.get("final_score", 0.0))
        tr_final = float(trainee_res.get("final_score", 0.0))
        delta_final = float(tr_final - ex_final)

        rows.append({
            "entry_number": "OVERALL",
            "expert_id": "",
            "expert_present_sec": round(float(ex_total_present_sec), 3),
            "expert_unseen_sec": round(float(ex_total_unseen_sec), 3),
            "expert_score": round(float(ex_final), 3),
            "trainee_id": "",
            "trainee_present_sec": round(float(tr_total_present_sec), 3),
            "trainee_unseen_sec": round(float(tr_total_unseen_sec), 3),
            "trainee_score": round(float(tr_final), 3),
        })

        # Write CSV
        # (CSV output removed)

        # Plot time-series + per-entry breakdown.
        try:
            ex_series = expert_res.get("series", []) or []
            tr_series = trainee_res.get("series", []) or []

            # Build per-entry (exclude OVERALL) bar data from the already-built `rows`
            entry_rows = [r for r in rows if str(r.get("entry_number")) != "OVERALL"]
            # keep only numeric entry_numbers
            entry_rows = [r for r in entry_rows if isinstance(r.get("entry_number"), int)]
            entry_rows.sort(key=lambda r: int(r["entry_number"]))

            entry_nums = [int(r["entry_number"]) for r in entry_rows]
            ex_unseen = [float(r.get("expert_unseen_sec", 0.0) or 0.0) for r in entry_rows]
            tr_unseen = [float(r.get("trainee_unseen_sec", 0.0) or 0.0) for r in entry_rows]
            ex_entry_score = [float(r.get("expert_score", 0.0) or 0.0) for r in entry_rows]
            tr_entry_score = [float(r.get("trainee_score", 0.0) or 0.0) for r in entry_rows]

            # If both series exist, align to a common horizon for fair viewing
            if ex_series and tr_series:
                ex_t = [p[0] for p in ex_series]
                tr_t = [p[0] for p in tr_series]
                t_max = min(max(ex_t), max(tr_t))
                ex_plot = [(t, s) for (t, s) in ex_series if t <= t_max]
                tr_plot = [(t, s) for (t, s) in tr_series if t <= t_max]
            else:
                ex_plot = ex_series
                tr_plot = tr_series

            # --- Figure layout: left = time series, right = per-entry bars ---
            fig = plt.figure(figsize=(13.5, 4.8), constrained_layout=True)
            gs = fig.add_gridspec(1, 2, width_ratios=[2.3, 1.2])
            ax_ts = fig.add_subplot(gs[0, 0])
            ax_bar = fig.add_subplot(gs[0, 1])

            # --- Left panel: cumulative score vs time ---
            if ex_plot:
                ax_ts.plot(
                    [t for t, _ in ex_plot],
                    [s for _, s in ex_plot],
                    label="Expert (cumulative)",
                    color="#1f77b4",
                    linewidth=2.5,
                )
            if tr_plot:
                ax_ts.plot(
                    [t for t, _ in tr_plot],
                    [s for _, s in tr_plot],
                    label="Trainee (cumulative)",
                    color="#ff7f0e",
                    linewidth=2.5,
                )

            ax_ts.set_xlabel("Seconds since first team entry")
            ax_ts.set_ylabel("Teammate coverage score (cumulative)")
            ax_ts.set_title("TEAMMATE_COVERAGE: trend over time")
            ax_ts.set_ylim(0.0, 1.02)
            ax_ts.grid(True, axis="y", linestyle="--", alpha=0.35)

            # Annotate final scores
            ax_ts.text(
                0.01,
                0.02,
                f"Final: Expert={ex_final:.2f} | Trainee={tr_final:.2f} (Δ={delta_final:+.2f})\n"
                f"Total unseen sec: Expert={ex_total_unseen_sec:.1f} | Trainee={tr_total_unseen_sec:.1f}",
                transform=ax_ts.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bbbbbb", alpha=0.9),
            )

            # --- Right panel: per-entry breakdown (unseen seconds + entry score) ---
            if entry_nums:
                x = np.arange(len(entry_nums), dtype=float)
                width = 0.38

                ax_bar.bar(x - width / 2.0, ex_unseen, width, label="Expert unseen (sec)", color="#1f77b4", alpha=0.85)
                ax_bar.bar(x + width / 2.0, tr_unseen, width, label="Trainee unseen (sec)", color="#ff7f0e", alpha=0.85)

                ax_bar.set_xticks(x)
                ax_bar.set_xticklabels([str(n) for n in entry_nums])
                ax_bar.set_xlabel("Entry number")
                ax_bar.set_ylabel("Unseen time (sec)")
                ax_bar.set_title("Per-entry unseen time (labels show ~score)")
                ax_bar.grid(True, axis="y", linestyle="--", alpha=0.35)

                # Add per-entry score as small text above bars (keeps chart readable)
                for i, (es, ts) in enumerate(zip(ex_entry_score, tr_entry_score)):
                    ax_bar.text(i - width / 2.0, ex_unseen[i] + 0.02, f"~{es:.2f}", ha="center", va="bottom", fontsize=8)
                    ax_bar.text(i + width / 2.0, tr_unseen[i] + 0.02, f"~{ts:.2f}", ha="center", va="bottom", fontsize=8)

                # Make y-limits readable
                ymax = max(max(ex_unseen) if ex_unseen else 0.0, max(tr_unseen) if tr_unseen else 0.0, 1.0)
                ax_bar.set_ylim(0.0, ymax * 1.18)
            else:
                ax_bar.axis("off")
                ax_bar.text(0.5, 0.5, "No per-entry data", ha="center", va="center")

            # --- Legend: outside (right side), not covering plot ---
            handles_ts, labels_ts = ax_ts.get_legend_handles_labels()
            handles_bar, labels_bar = ax_bar.get_legend_handles_labels()
            handles = handles_ts + handles_bar
            labels = labels_ts + labels_bar
            fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0)

            plt.savefig(img_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        except Exception:
            # If plotting fails, continue with text-only output
            pass

        # Average unseen-time difference across entry numbers where both sides have a present track.
        entry_rows_for_avg = [r for r in rows if isinstance(r.get("entry_number"), int)]
        diffs = []
        for r in entry_rows_for_avg:
            ex_p = float(r.get("expert_present_sec", 0.0) or 0.0)
            tr_p = float(r.get("trainee_present_sec", 0.0) or 0.0)
            if ex_p <= 0 or tr_p <= 0:
                continue
            ex_u = float(r.get("expert_unseen_sec", 0.0) or 0.0)
            tr_u = float(r.get("trainee_unseen_sec", 0.0) or 0.0)
            diffs.append(tr_u - ex_u)
        avg_unseen_dt = float(np.mean(diffs)) if diffs else None

        if abs(delta_final) <= 0.02:
            score_part = f"Overall teammate coverage looks about the same as the expert (Trainee {tr_final:.2f} vs Expert {ex_final:.2f})."
        elif delta_final > 0:
            score_part = f"Overall teammate coverage looks better than the expert (Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
        else:
            score_part = f"Overall teammate coverage looks worse than the expert (Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."

        if avg_unseen_dt is None:
            unseen_part = "I couldn't compute an average unseen-time gap per entrant (missing paired entry tracks)."
        elif abs(avg_unseen_dt) <= 0.10:
            unseen_part = "On average, unseen time per entrant was about the same as the expert."
        elif avg_unseen_dt < 0:
            unseen_part = f"On average, the trainee had about {abs(avg_unseen_dt):.2f}s less unseen time per entrant than the expert (better coverage)."
        else:
            unseen_part = f"On average, the trainee had about {abs(avg_unseen_dt):.2f}s more unseen time per entrant than the expert."

        text = score_part + "\n" + unseen_part

        # Also save the returned summary text alongside the image
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            # Non-fatal: still return the summary even if file write fails
            pass

        return {
            "Name": "TEAMMATE_COVERAGE",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }