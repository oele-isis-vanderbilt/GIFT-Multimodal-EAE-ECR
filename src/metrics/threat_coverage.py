import os
import glob
import json
from typing import Dict, Optional, Tuple, List

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from . import AbstractMetric

# --------------------------------------------------------------------
# Helpers for gaze‐cone construction and intersection
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
    rot_left  = np.array([[cos_a, -sin_a],
                          [sin_a,  cos_a]], dtype=np.float32)
    rot_right = np.array([[cos_a,  sin_a],
                          [-sin_a, cos_a]], dtype=np.float32)

    o = np.asarray(origin, dtype=np.float32)
    left_vec  = rot_left  @ d * length
    right_vec = rot_right @ d * length
    return np.stack([o, o + left_vec, o + right_vec], axis=0)


def _triangle_box_intersect(triangle, box):
    """Return True if the triangle intersects the axis-aligned bbox."""
    tri = np.asarray(triangle, dtype=np.float32).reshape(-1, 1, 2)
    x1, y1, x2, y2 = map(float, box)
    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    dtype=np.float32).reshape(-1, 1, 2)
    inter_area, _ = cv2.intersectConvexConvex(tri, rect)
    return inter_area > 0.0


class ThreatCoverage_Metric(AbstractMetric):
    """Fraction of frames where an active (uncleared) threat is covered by gaze."""
    def __init__(self, config):
        super().__init__(config)
        self.metricName = "THREAT_COVERAGE"
        # Field of view half‐angle (degrees)
        self.coverage_angle = float(config.get("visual_angle_degrees", 20.0))
        # List of enemy track IDs to consider
        self.enemy_ids = config.get("enemy_ids", [99])
        self._final_score = 0.0

    def process(self, ctx):
        """Score = covered_frames / counted_frames while threats are present & uncleared."""
        present_frames = 0
        covered_frames = 0

        all_frame_count = len(ctx.all_frames)
        # Build list of trainee IDs (all tracks minus enemies)
        all_ids = list(ctx.tracks_by_id.keys())
        trainee_ids = [tid for tid in all_ids if tid not in self.enemy_ids]

        for frame_idx in range(1, all_frame_count + 1):
            # Check if any teammate is present this frame
            any_trainee_present = False
            for tid in trainee_ids:
                if (frame_idx, tid) in ctx.bbox_details:
                    any_trainee_present = True
                    break

            # Enemies present this frame and still uncleared.
            uncleared_enemies = []
            for enemy_id in self.enemy_ids:
                # Must appear in this frame
                if (frame_idx, enemy_id) not in ctx.bbox_details:
                    continue

                cleared = False
                if hasattr(ctx, "threat_clearance"):
                    clear_tuple = ctx.threat_clearance.get(enemy_id)
                    if clear_tuple is not None:
                        # clear_tuple is (start_frame, end_frame, friend_id)
                        _, end_frame, _ = clear_tuple
                        # Enemy must be covered until the END of the clearance window.
                        # Treat as cleared only from end_frame onward.
                        if end_frame is not None and frame_idx >= end_frame:
                            cleared = True

                if not cleared:
                    uncleared_enemies.append(enemy_id)

            # Only count this frame if at least one trainee and one uncleared enemy
            if not any_trainee_present or len(uncleared_enemies) == 0:
                continue

            present_frames += 1

            # Check if any trainee is looking at any uncleared enemy
            looked_at = False
            half_angle = self.coverage_angle / 2.0
            for tid in trainee_ids:
                if (frame_idx, tid) not in ctx.gaze_info:
                    continue
                ox, oy, dx, dy = ctx.gaze_info[(frame_idx, tid)]
                origin = np.array([ox, oy], dtype=np.float32)
                direction = np.array([dx, dy], dtype=np.float32)
                if np.linalg.norm(direction) == 0.0:
                    # Zero‐vector gaze counts as “looking”
                    looked_at = True
                    break
                tri = _gaze_triangle(origin, direction, half_angle)
                for enemy_id in uncleared_enemies:
                    bbox = ctx.bbox_details[(frame_idx, enemy_id)]
                    if _triangle_box_intersect(tri, bbox):
                        looked_at = True
                        break
                if looked_at:
                    break

            if looked_at:
                covered_frames += 1

        # Compute final score
        if present_frames == 0:
            self._final_score = 0.0
        else:
            self._final_score = covered_frames / present_frames

    def getFinalScore(self) -> float:
        return round(self._final_score, 2)

    # --------------------------------------------------------------------- #
    # Expert comparison (folder-based)
    # --------------------------------------------------------------------- #
    @staticmethod
    def expertCompare(
        session_folder: str,
        expert_folder: str,
        _map_image=None,
        config: Optional[dict] = None,
    ):
        """Compare trainee vs expert threat coverage.

        Coverage is counted only while at least one enemy is present and uncleared.
        Time is aligned to each run's first non-enemy entry.
        Writes `THREAT_COVERAGE_Comparison.jpg`.
        """

        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_tracker_output(folder: str) -> List[Dict]:
            path = _pick_latest(folder, "*_TrackerOutput.json")
            if path is None:
                raise FileNotFoundError(f"No TrackerOutput found in {folder}")
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"Unexpected TrackerOutput format: {path}")
            return data

        def _load_gaze_cache(folder: str) -> Dict[Tuple[int, int], Tuple[float, float, float, float]]:
            path = _pick_latest(folder, "*_GazeCache.txt")
            if path is None:
                return {}
            df = pd.read_csv(path)
            if df is None or df.empty:
                return {}
            cols = {c.lower(): c for c in df.columns}
            f_col = cols.get("frame")
            id_col = cols.get("id")
            ox_col = cols.get("ox")
            oy_col = cols.get("oy")
            dx_col = cols.get("dx")
            dy_col = cols.get("dy")
            if None in (f_col, id_col, ox_col, oy_col, dx_col, dy_col):
                return {}

            out: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
            for _, r in df.iterrows():
                try:
                    fr = int(r[f_col])
                    tid = int(r[id_col])
                    out[(fr, tid)] = (
                        float(r[ox_col]),
                        float(r[oy_col]),
                        float(r[dx_col]),
                        float(r[dy_col]),
                    )
                except Exception:
                    continue
            return out

        def _load_clearance_cache(folder: str) -> Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]]:
            """Return {enemy_id: (start_frame, end_frame, friend_id)} from cache."""
            path = _pick_latest(folder, "*_ThreatClearanceCache.txt")
            if path is None:
                return {}
            df = pd.read_csv(path)
            if df is None or df.empty:
                return {}
            cols = {c.lower(): c for c in df.columns}
            eid_col = cols.get("enemy_id") or cols.get("enemy")
            s_col = cols.get("immediate_frame") or cols.get("start_frame") or cols.get("start")
            e_col = cols.get("contact_end_frame") or cols.get("end_frame") or cols.get("end")
            f_col = cols.get("clearing_friend") or cols.get("friend") or cols.get("clearing")
            if eid_col is None:
                return {}

            out: Dict[int, Tuple[Optional[int], Optional[int], Optional[int]]] = {}
            for _, r in df.iterrows():
                try:
                    eid = int(r[eid_col])
                except Exception:
                    continue

                def _to_int(x) -> Optional[int]:
                    try:
                        if pd.isna(x):
                            return None
                        xi = int(x)
                        return None if xi < 0 else xi
                    except Exception:
                        return None

                start = _to_int(r[s_col]) if s_col is not None else None
                end = _to_int(r[e_col]) if e_col is not None else None
                fid = _to_int(r[f_col]) if f_col is not None else None
                out[eid] = (start, end, fid)
            return out

        def _first_team_entry_frame(tracker_output: List[Dict], enemy_ids: List[int]) -> Optional[int]:
            min_fr = None
            for entry in tracker_output:
                fr = int(entry.get("frame", 0))
                for obj in entry.get("objects", []) or []:
                    tid = obj.get("id")
                    if tid is None:
                        continue
                    try:
                        tid = int(tid)
                    except Exception:
                        continue
                    if tid in enemy_ids:
                        continue
                    if min_fr is None or fr < min_fr:
                        min_fr = fr
            return min_fr

        def _extract_bbox_map(tracker_output: List[Dict]) -> Tuple[Dict[Tuple[int, int], Tuple[float, float, float, float]], Dict[int, set]]:
            bbox_map: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
            ids_per_frame: Dict[int, set] = {}
            for entry in tracker_output:
                fr = int(entry.get("frame", 0))
                objs = entry.get("objects", []) or []
                ids_per_frame.setdefault(fr, set())
                for obj in objs:
                    tid = obj.get("id")
                    bb = obj.get("bbox")
                    if tid is None or bb is None:
                        continue
                    try:
                        tid = int(tid)
                        x1, y1, x2, y2 = bb
                        bbox_map[(fr, tid)] = (float(x1), float(y1), float(x2), float(y2))
                        ids_per_frame[fr].add(tid)
                    except Exception:
                        continue
            return bbox_map, ids_per_frame

        def _evaluate_folder(
            folder: str,
            *,
            enemy_ids: List[int],
            fps: float,
        ) -> Dict[str, object]:
            tracker_output = _load_tracker_output(folder)
            gaze_info = _load_gaze_cache(folder)
            clearance = _load_clearance_cache(folder)

            bbox_map, ids_per_frame = _extract_bbox_map(tracker_output)

            # If enemy_ids not provided by config, infer from clearance cache if possible.
            if not enemy_ids:
                enemy_ids = sorted(list(clearance.keys())) if clearance else [99]

            # Team IDs = all ids seen minus enemies
            all_ids = set()
            for s in ids_per_frame.values():
                all_ids |= set(s)
            trainee_ids = sorted([tid for tid in all_ids if tid not in enemy_ids])

            first_entry = _first_team_entry_frame(tracker_output, enemy_ids)
            if first_entry is None:
                first_entry = 1

            max_frame = max(ids_per_frame.keys()) if ids_per_frame else 0

            # Per-enemy counters
            per_enemy_present: Dict[int, int] = {eid: 0 for eid in enemy_ids}
            per_enemy_covered: Dict[int, int] = {eid: 0 for eid in enemy_ids}

            present_frames = 0
            covered_frames = 0

            # Curves for plotting: cumulative coverage over time (counted frames only)
            curve_t: List[float] = []
            curve_cov: List[float] = []

            half_angle = float(config.get("visual_angle_degrees", 20.0)) / 2.0 if isinstance(config, dict) else 10.0

            def _enemy_uncleared(eid: int, frame_idx: int) -> bool:
                # For expert compare: cleared from contact_end_frame onward.
                tup = clearance.get(eid)
                if tup is None:
                    return True
                _start, end, _fid = tup
                if end is None:
                    return True
                return frame_idx < int(end)

            for frame_idx in range(int(first_entry), int(max_frame) + 1):
                # Any trainee present?
                any_trainee_present = any((frame_idx, tid) in bbox_map for tid in trainee_ids)
                if not any_trainee_present:
                    continue

                # Enemies present & uncleared
                uncleared_enemies = [
                    eid for eid in enemy_ids
                    if (frame_idx, eid) in bbox_map and _enemy_uncleared(eid, frame_idx)
                ]
                if not uncleared_enemies:
                    continue

                present_frames += 1

                # Per-enemy present
                for eid in uncleared_enemies:
                    per_enemy_present[eid] += 1

                # Check gaze intersection
                looked_any = False
                looked_by_enemy: Dict[int, bool] = {eid: False for eid in uncleared_enemies}

                for tid in trainee_ids:
                    g = gaze_info.get((frame_idx, tid))
                    if g is None:
                        continue
                    ox, oy, dx, dy = g
                    direction = np.array([dx, dy], dtype=np.float32)
                    if float(np.linalg.norm(direction)) == 0.0:
                        # Treat zero-vector gaze as "looking"
                        looked_any = True
                        for eid in uncleared_enemies:
                            looked_by_enemy[eid] = True
                        break

                    origin = np.array([ox, oy], dtype=np.float32)
                    tri = _gaze_triangle(origin, direction, half_angle)

                    for eid in uncleared_enemies:
                        if looked_by_enemy.get(eid):
                            continue
                        bbox = bbox_map[(frame_idx, eid)]
                        if _triangle_box_intersect(tri, bbox):
                            looked_by_enemy[eid] = True
                            looked_any = True

                    if looked_any and all(looked_by_enemy.values()):
                        break

                if looked_any:
                    covered_frames += 1

                # Per-enemy covered
                for eid, ok in looked_by_enemy.items():
                    if ok:
                        per_enemy_covered[eid] += 1

                # Update curves (seconds since first entry)
                t_sec = (frame_idx - first_entry) / fps
                curve_t.append(float(t_sec))
                curve_cov.append(float(covered_frames / max(1, present_frames)))

            final_cov = float(covered_frames / present_frames) if present_frames > 0 else 0.0


            return {
                "first_entry_frame": int(first_entry),
                "fps": float(fps),
                "final_coverage": float(final_cov),
                "present_frames": int(present_frames),
                "covered_frames": int(covered_frames),
                "per_enemy_present": per_enemy_present,
                "per_enemy_covered": per_enemy_covered,
                "clearance": clearance,
                "curve_t": curve_t,
                "curve_cov": curve_cov,
            }

        def _generate_plot(
            *,
            out_path: str,
            expert_res: Dict[str, object],
            trainee_res: Dict[str, object],
            per_enemy_df: pd.DataFrame,
        ) -> None:
            """Stitched plot: cumulative threat coverage + per-enemy unseen time."""
            ex_t = expert_res.get("curve_t", [])
            ex_c = expert_res.get("curve_cov", [])
            tr_t = trainee_res.get("curve_t", [])
            tr_c = trainee_res.get("curve_cov", [])

            if (not ex_t and not tr_t) and (per_enemy_df is None or per_enemy_df.empty):
                return

            fig, (ax_ts, ax_bar) = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(14.5, 5.6),
                constrained_layout=True,
            )

            # ---- Panel 1: cumulative coverage over time ----
            if ex_t:
                ax_ts.plot(ex_t, ex_c, label="Expert", color="tab:blue", linewidth=2.5)
            if tr_t:
                ax_ts.plot(tr_t, tr_c, label="Trainee", color="tab:orange", linewidth=2.5)

            # Per-enemy clear-time markers (dotted), labeled by enemy id
            ex_clear = expert_res.get("clearance") or {}
            tr_clear = trainee_res.get("clearance") or {}

            ex_first = float(expert_res.get("first_entry_frame", 1))
            tr_first = float(trainee_res.get("first_entry_frame", 1))
            ex_fps = float(expert_res.get("fps", 30.0))
            tr_fps = float(trainee_res.get("fps", 30.0))

            def _mark_clear_times(clearance_map, first_frame, fps_local, color):
                if not clearance_map or fps_local <= 0:
                    return
                for eid, tup in clearance_map.items():
                    try:
                        eid_i = int(eid)
                    except Exception:
                        continue
                    if tup is None or len(tup) < 2 or tup[1] is None:
                        continue
                    try:
                        t_sec = (float(int(tup[1])) - float(first_frame)) / float(fps_local)
                    except Exception:
                        continue
                    if not np.isfinite(t_sec):
                        continue
                    ax_ts.axvline(float(t_sec), linestyle=":", color=color, alpha=0.45, linewidth=1.2)
                    ax_ts.text(
                        float(t_sec),
                        1.045,
                        f"Enemy ID: {eid_i}",
                        rotation=90,
                        color=color,
                        ha="center",
                        va="top",
                        fontsize=7,
                        alpha=0.9,
                        clip_on=True,
                    )

            _mark_clear_times(ex_clear, ex_first, ex_fps, "tab:blue")
            _mark_clear_times(tr_clear, tr_first, tr_fps, "tab:orange")

            ax_ts.set_xlabel("Seconds since first team entry")
            ax_ts.set_ylabel("Cumulative threat coverage")
            ax_ts.set_title("Threat coverage until threats are cleared")
            ax_ts.set_ylim(0.0, 1.05)
            ax_ts.grid(True, axis="y", linestyle="--", alpha=0.35)
            # ax_ts.legend(loc="upper left", framealpha=0.9)

            # ---- Panel 2: per-enemy unseen time (until clearance) ----
            if per_enemy_df is None or per_enemy_df.empty:
                ax_bar.set_axis_off()
            else:
                dfp = per_enemy_df.copy()
                dfp["enemy_id"] = pd.to_numeric(dfp["enemy_id"], errors="coerce")
                dfp = dfp.dropna(subset=["enemy_id"]).sort_values("enemy_id")
                if dfp.empty:
                    ax_bar.set_axis_off()
                else:
                    enemy_ids = [int(x) for x in dfp["enemy_id"].tolist()]
                    idx = np.arange(len(enemy_ids), dtype=float)
                    width = 0.36

                    # Unseen seconds per enemy = (present - covered) / fps (per run)
                    ex_fps = float(expert_res.get("fps", 30.0))
                    tr_fps = float(trainee_res.get("fps", 30.0))

                    ex_p = pd.to_numeric(dfp["expert_enemy_present_frames"], errors="coerce").fillna(0).to_numpy(dtype=float)
                    ex_cov = pd.to_numeric(dfp["expert_enemy_covered_frames"], errors="coerce").fillna(0).to_numpy(dtype=float)
                    tr_p = pd.to_numeric(dfp["trainee_enemy_present_frames"], errors="coerce").fillna(0).to_numpy(dtype=float)
                    tr_cov = pd.to_numeric(dfp["trainee_enemy_covered_frames"], errors="coerce").fillna(0).to_numpy(dtype=float)

                    ex_unseen_sec = np.maximum(0.0, (ex_p - ex_cov) / max(ex_fps, 1.0))
                    tr_unseen_sec = np.maximum(0.0, (tr_p - tr_cov) / max(tr_fps, 1.0))

                    # Coverage scores per enemy
                    ex_score = np.where(ex_p > 0, ex_cov / ex_p, 0.0)
                    tr_score = np.where(tr_p > 0, tr_cov / tr_p, 0.0)

                    ax_bar.bar(idx - width / 2.0, ex_unseen_sec, width, label="Expert", color="tab:blue", alpha=0.9)
                    ax_bar.bar(idx + width / 2.0, tr_unseen_sec, width, label="Trainee", color="tab:orange", alpha=0.9)

                    # Labels show rounded score with ~
                    for i in range(len(enemy_ids)):
                        ax_bar.text(
                            idx[i] - width / 2.0,
                            ex_unseen_sec[i] + 0.02,
                            f"~{float(ex_score[i]):.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )
                        ax_bar.text(
                            idx[i] + width / 2.0,
                            tr_unseen_sec[i] + 0.02,
                            f"~{float(tr_score[i]):.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                        )

                    ax_bar.set_xticks(idx)
                    ax_bar.set_xticklabels([str(e) for e in enemy_ids])
                    ax_bar.set_xlabel("Enemy ID")
                    ax_bar.set_ylabel("Unseen time (seconds)")
                    ax_bar.set_title("Per-enemy unseen time while uncleared (labels show ~coverage score)")
                    ax_bar.grid(True, axis="y", linestyle="--", alpha=0.35)
                    # ax_bar.legend(loc="upper left", framealpha=0.9)

            # Single legend for the full figure (outside, top-right)
            h1, l1 = ax_ts.get_legend_handles_labels()
            h2, l2 = ax_bar.get_legend_handles_labels()
            seen = set()
            handles = []
            labels = []
            for h, lab in list(zip(h1, l1)) + list(zip(h2, l2)):
                if not lab or lab in seen:
                    continue
                seen.add(lab)
                handles.append(h)
                labels.append(lab)

            if handles:
                fig.legend(
                    handles,
                    labels,
                    loc="upper left",
                    bbox_to_anchor=(1.02, 1.0),
                    framealpha=0.9,
                )
                fig.subplots_adjust(right=0.82)

            plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)

        # ------------------ Main ------------------
        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "THREAT_COVERAGE_Comparison.jpg")
        txt_path = os.path.join(session_folder, "THREAT_COVERAGE_Comparison.txt")

        fps = 30.0
        if isinstance(config, dict) and config.get("frame_rate") is not None:
            try:
                fps = float(config.get("frame_rate"))
            except Exception:
                fps = 30.0

        enemy_ids_cfg: List[int] = []
        if isinstance(config, dict):
            try:
                enemy_ids_cfg = [int(x) for x in config.get("enemy_ids", [])]
            except Exception:
                enemy_ids_cfg = []

        try:
            expert_res = _evaluate_folder(expert_folder, enemy_ids=enemy_ids_cfg, fps=fps)
            trainee_res = _evaluate_folder(session_folder, enemy_ids=enemy_ids_cfg, fps=fps)
        except Exception:
            err_text = "There was an error while processing this comparison. Missing TrackerOutput/GazeCache/ThreatClearanceCache."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "THREAT_COVERAGE",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Align enemy list across both results
        ex_enemies = set((expert_res.get("per_enemy_present") or {}).keys())
        tr_enemies = set((trainee_res.get("per_enemy_present") or {}).keys())
        enemies = sorted(list(ex_enemies | tr_enemies))
        if not enemies:
            enemies = [99]

        rows: List[Dict[str, object]] = []


        # Per-enemy rows
        for eid in enemies:
            ex_p = int((expert_res.get("per_enemy_present") or {}).get(eid, 0))
            ex_c = int((expert_res.get("per_enemy_covered") or {}).get(eid, 0))
            tr_p = int((trainee_res.get("per_enemy_present") or {}).get(eid, 0))
            tr_c = int((trainee_res.get("per_enemy_covered") or {}).get(eid, 0))
            rows.append({
                "enemy_id": int(eid),
                "expert_enemy_present_frames": ex_p,
                "expert_enemy_covered_frames": ex_c,
                "trainee_enemy_present_frames": tr_p,
                "trainee_enemy_covered_frames": tr_c,
            })

        per_enemy_df = pd.DataFrame(rows) if rows else pd.DataFrame([])

        # Plot
        try:
            _generate_plot(out_path=img_path, expert_res=expert_res, trainee_res=trainee_res, per_enemy_df=per_enemy_df)
        except Exception:
            pass

        # ---- Summary (score diff + avg uncovered time per enemy) ----
        ex_final = float(expert_res.get("final_coverage", 0.0))
        tr_final = float(trainee_res.get("final_coverage", 0.0))
        delta_final = float(tr_final - ex_final)

        if abs(delta_final) <= 0.02:
            score_part = (
                f"Overall threat coverage looks about the same as the expert "
                f"(Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
            )
        elif delta_final > 0:
            score_part = (
                f"Overall threat coverage looks better than the expert "
                f"(Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
            )
        else:
            score_part = (
                f"Overall threat coverage looks worse than the expert "
                f"(Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
            )

        def _avg_uncovered_sec(res: Dict[str, object]) -> Optional[float]:
            present = res.get("per_enemy_present") or {}
            covered = res.get("per_enemy_covered") or {}
            try:
                fps_local = float(res.get("fps", fps))
            except Exception:
                fps_local = fps
            if fps_local <= 0:
                return None

            vals: List[float] = []
            for eid, p in present.items():
                try:
                    p_int = int(p)
                except Exception:
                    continue
                if p_int <= 0:
                    continue
                try:
                    c_int = int(covered.get(eid, 0))
                except Exception:
                    c_int = 0
                vals.append(max(0.0, (p_int - c_int) / fps_local))
            return float(np.mean(vals)) if vals else None

        ex_avg_u = _avg_uncovered_sec(expert_res)
        tr_avg_u = _avg_uncovered_sec(trainee_res)

        if ex_avg_u is None or tr_avg_u is None:
            uncovered_part = (
                "On average, uncovered time per enemy before clearance couldn't be computed (missing per-enemy counts)."
            )
        else:
            du = float(tr_avg_u - ex_avg_u)
            if abs(du) <= 0.10:
                uncovered_part = "On average, the trainee had about the same uncovered time per enemy before clearance as the expert."
            elif du < 0:
                uncovered_part = (
                    f"On average, the trainee had about {abs(du):.2f}s less uncovered time per enemy before clearance than the expert "
                    f"(better coverage)."
                )
            else:
                uncovered_part = (
                    f"On average, the trainee had about {abs(du):.2f}s more uncovered time per enemy before clearance than the expert."
                )

        text = score_part + "\n" + uncovered_part

        # Also save the returned summary text alongside the image
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            # Non-fatal: still return the summary even if file write fails
            pass

        return {
            "Name": "THREAT_COVERAGE",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }
