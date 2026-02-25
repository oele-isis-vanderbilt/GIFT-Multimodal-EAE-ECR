import os
import glob
from typing import Dict, List, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metric import AbstractMetric


class POD_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "IDENTIFY_AND_HOLD_DESIGNATED_AREA"
        self.pod = config["POD"]
        self.num_tracks = len(self.pod)
        self.tracks = {}
        self.pod_assignment: Dict[int, Optional[int]] = {}

    def process(self, ctx):
        """Cache teammate tracks (non-enemy IDs)."""
        enemy_ids = self.config.get("enemy_ids", [99])
        self.tracks = {
            idx: traj
            for idx, traj in ctx.tracks_by_id.items()
            if idx not in enemy_ids
        }

    @staticmethod
    def _first_non_null(trk: List[Optional[Tuple[float, float]]]) -> Optional[int]:
        """Index of the first non-None point."""
        for i, v in enumerate(trk):
            if v is not None:
                return i
        return None

    @staticmethod
    def _traj_len(trk: List[Optional[Tuple[float, float]]]) -> int:
        """Count of non-None points."""
        return int(sum(1 for v in trk if v is not None))

    def _pod_points_list(self) -> List[np.ndarray]:
        """POD points as float arrays."""
        return [
            np.asarray(p, dtype=float)
            for p in (self.pod.tolist() if isinstance(self.pod, np.ndarray) else list(self.pod))
        ]

    def getFinalScore(self) -> float:
        """Average POD holding score (0–1)."""
        if not self.tracks:
            return -1

        pod_points = self._pod_points_list()
        if not pod_points:
            return -1

        # Select top-N tracks by length, then order by entry time.
        track_items = list(self.tracks.items())
        track_items.sort(key=lambda it: self._traj_len(it[1]), reverse=True)
        track_items = track_items[: self.num_tracks]

        if len(track_items) < self.num_tracks:
            return -1

        track_items.sort(
            key=lambda it: (
                self._first_non_null(it[1])
                if self._first_non_null(it[1]) is not None
                else 10**9
            )
        )

        # Assign PODs using the same alternating-side entrance logic.
        self.pod_assignment = self._assign_pods(track_items, pod_points)

        # Holding score = mean(exp(-d^2/5000)) over frames.
        denom = 5000.0
        scores: List[float] = []
        for tid, trk in track_items:
            pod_idx = self.pod_assignment.get(int(tid))
            if pod_idx is None or pod_idx < 0 or pod_idx >= len(pod_points):
                continue
            pod_xy = pod_points[int(pod_idx)]

            pts = np.array([p for p in trk if p is not None], dtype=float)
            if pts.size == 0:
                continue
            d = np.linalg.norm(pts - pod_xy.reshape(1, 2), axis=1)
            per_frame_score = np.exp(-(d * d) / denom)
            scores.append(float(np.mean(per_frame_score)))

        return round(float(np.mean(scores)) if scores else -1.0, 2)

    def _assign_pods(
        self,
        track_items: List[Tuple[int, List[Optional[Tuple[float, float]]]]],
        pod_points: List[np.ndarray],
    ) -> Dict[int, Optional[int]]:
        """Assign POD indices to track IDs using alternating-side entrance logic."""
        # Determine entrance direction from the first entrant.
        first_trk = track_items[0][1]
        first_pts = [p for p in first_trk if p is not None][: min(30, len(first_trk))]

        if not first_pts:
            x0, y0 = float(pod_points[0][0]), float(pod_points[0][1])
            entrance_sign = 1
        elif len(first_pts) < 2:
            x0, y0 = float(first_pts[0][0]), float(first_pts[0][1])
            pod_mean_x = float(np.mean([p[0] for p in pod_points]))
            entrance_sign = 1 if x0 >= pod_mean_x else -1
        else:
            x0, y0 = float(first_pts[0][0]), float(first_pts[0][1])
            x_coords = [float(p[0]) for p in first_pts]
            deltas = np.diff(x_coords)
            entrance_sign = 1 if float(np.mean(deltas)) >= 0 else -1  # +1 right, -1 left

        # Split PODs into two sides relative to doorway x.
        pods_plus = [(i, p) for i, p in enumerate(pod_points) if float(p[0]) > x0]
        pods_minus = [(i, p) for i, p in enumerate(pod_points) if float(p[0]) <= x0]

        # Sort each side by vertical distance from doorway (furthest first).
        pods_plus.sort(key=lambda t: abs(float(t[1][1]) - y0), reverse=True)
        pods_minus.sort(key=lambda t: abs(float(t[1][1]) - y0), reverse=True)

        out: Dict[int, Optional[int]] = {}
        i_plus = 0
        i_minus = 0
        current_side = int(entrance_sign)

        for tid, _trk in track_items:
            assigned: Optional[int] = None
            if current_side == 1 and i_plus < len(pods_plus):
                assigned = int(pods_plus[i_plus][0])
                i_plus += 1
            elif current_side == -1 and i_minus < len(pods_minus):
                assigned = int(pods_minus[i_minus][0])
                i_minus += 1
            else:
                # If the intended side is exhausted, pull from the other.
                if i_plus < len(pods_plus):
                    assigned = int(pods_plus[i_plus][0])
                    i_plus += 1
                elif i_minus < len(pods_minus):
                    assigned = int(pods_minus[i_minus][0])
                    i_minus += 1

            out[int(tid)] = assigned
            current_side *= -1

        return out

    def expertCompare(
        self,
        session_folder: str,
        expert_folder: str,
        _map_image=None,
        config: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Compare trainee vs expert POD holding and write a comparison plot."""

        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_position_tracks(
            folder: str, enemy_ids_local: Set[int]
        ) -> Tuple[Dict[int, List[Optional[Tuple[float, float]]]], float]:
            path = _pick_latest(folder, "*_PositionCache.txt")
            if path is None:
                return {}, 30.0

            try:
                df = pd.read_csv(path)
            except Exception:
                return {}, 30.0

            if df is None or df.empty:
                return {}, 30.0

            cols = {c.strip().lower(): c for c in df.columns}
            frame_col = cols.get("frame")
            id_col = cols.get("id")
            x_col = cols.get("mapx")
            y_col = cols.get("mapy")

            # Match EntranceVectors: require these exact columns.
            if frame_col is None or id_col is None or x_col is None or y_col is None:
                return {}, 30.0

            df = df[[frame_col, id_col, x_col, y_col]].dropna()
            if df.empty:
                return {}, 30.0

            df[frame_col] = df[frame_col].astype(int)
            df[id_col] = df[id_col].astype(int)
            df[x_col] = df[x_col].astype(float)
            df[y_col] = df[y_col].astype(float)

            fps_local = float((config or {}).get("frame_rate", self.config.get("frame_rate", 30.0)))
            if fps_local <= 0:
                fps_local = 30.0

            # Exclude enemies
            df = df[~df[id_col].isin([int(x) for x in enemy_ids_local])].copy()
            if df.empty:
                return {}, fps_local

            max_frame = int(df[frame_col].max()) if len(df) else 0
            tracks: Dict[int, List[Optional[Tuple[float, float]]]] = {}

            for tid, g in df.groupby(id_col):
                tid_i = int(tid)
                # Already filtered above; no need to check enemy_ids_local
                traj: List[Optional[Tuple[float, float]]] = [None] * max_frame
                for _, row in g.iterrows():
                    fidx = int(row[frame_col])
                    if 1 <= fidx <= max_frame:
                        traj[fidx - 1] = (float(row[x_col]), float(row[y_col]))
                tracks[tid_i] = traj

            return tracks, fps_local

        def _select_top_tracks(
            tracks: Dict[int, List[Optional[Tuple[float, float]]]], n_local: int
        ) -> List[Tuple[int, List[Optional[Tuple[float, float]]]]]:
            items = list(tracks.items())
            items.sort(key=lambda it: self._traj_len(it[1]), reverse=True)
            items = items[:n_local]
            items.sort(
                key=lambda it: (
                    self._first_non_null(it[1])
                    if self._first_non_null(it[1]) is not None
                    else 10**9
                )
            )
            return items

        def _entry_map(tracks: Dict[int, List[Optional[Tuple[float, float]]]]) -> Dict[int, int]:
            starts = []
            for tid, trk in tracks.items():
                f = self._first_non_null(trk)
                if f is not None:
                    starts.append((int(tid), int(f)))
            starts.sort(key=lambda t: t[1])
            return {tid: i + 1 for i, (tid, _f) in enumerate(starts)}

        def _hold_stats(
            track_items: List[Tuple[int, List[Optional[Tuple[float, float]]]]],
            pod_points_local: List[np.ndarray],
            assign: Dict[int, Optional[int]],
            fps_local: float,
        ) -> Tuple[float, Dict[int, Dict[str, float]], List[float], List[float]]:
            """Return overall score, per-POD scores, and a score curve."""
            denom = 5000.0

            firsts = [
                f
                for _tid, trk in track_items
                for f in [self._first_non_null(trk)]
                if f is not None
            ]
            first_entry = int(min(firsts)) if firsts else 0

            per_pod: Dict[int, Dict[str, float]] = {}
            for tid, _trk in track_items:
                pod_idx = assign.get(int(tid))
                if pod_idx is None:
                    continue
                per_pod[int(pod_idx)] = {
                    "track_id": float(int(tid)),
                    "score": 0.0,
                }

            max_len = max((len(trk) for _tid, trk in track_items), default=0)
            curve_t: List[float] = []
            curve_s: List[float] = []

            cum_sum = 0.0
            cum_n = 0

            for fr in range(first_entry, max_len):
                frame_scores: List[float] = []
                for tid, trk in track_items:
                    pt = trk[fr] if fr < len(trk) else None
                    if pt is None:
                        continue

                    pod_idx = assign.get(int(tid))
                    if pod_idx is None or pod_idx < 0 or pod_idx >= len(pod_points_local):
                        continue

                    d = float(
                        np.linalg.norm(
                            np.asarray(pt, dtype=float) - pod_points_local[int(pod_idx)].reshape(2)
                        )
                    )
                    sc = float(np.exp(-(d * d) / denom))
                    frame_scores.append(sc)

                if frame_scores:
                    frame_mean = float(np.mean(frame_scores))
                    cum_sum += frame_mean
                    cum_n += 1
                    curve_t.append((fr - first_entry) / max(fps_local, 1.0))
                    curve_s.append(cum_sum / cum_n)

            total_scores: List[float] = []
            for pod_idx, st in per_pod.items():
                tid = int(st["track_id"])
                trk = dict(track_items).get(tid)
                if trk is None:
                    continue
                pod_xy = pod_points_local[int(pod_idx)]
                pts = np.array([p for p in trk if p is not None], dtype=float)
                if pts.size == 0:
                    continue
                d = np.linalg.norm(pts - pod_xy.reshape(1, 2), axis=1)
                st["score"] = float(np.mean(np.exp(-(d * d) / denom)))
                total_scores.append(st["score"])

            overall = float(np.mean(total_scores)) if total_scores else -1.0
            return overall, per_pod, curve_t, curve_s

        def _pretty_table(headers: List[str], data_rows: List[List[str]]) -> str:
            """Create a fixed-width, pipe-delimited table suitable for plain TXT."""
            if not headers:
                return ""

            norm_rows: List[List[str]] = []
            for r in (data_rows or []):
                rr = list(r)
                if len(rr) < len(headers):
                    rr = rr + ["N/A"] * (len(headers) - len(rr))
                elif len(rr) > len(headers):
                    rr = rr[: len(headers)]
                norm_rows.append(["" if v is None else str(v) for v in rr])

            widths: List[int] = []
            for j, h in enumerate(headers):
                max_len = len(str(h))
                for r in norm_rows:
                    if j < len(r):
                        max_len = max(max_len, len(str(r[j])))
                widths.append(max_len)

            sep = " | "
            header_line = sep.join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
            dash_line = sep.join("-" * widths[i] for i in range(len(headers)))

            out_lines = [header_line, dash_line]
            for r in norm_rows:
                out_lines.append(sep.join(str(r[i]).ljust(widths[i]) for i in range(len(headers))))

            return "\n".join(out_lines)

        enemy_ids = set((config or {}).get("enemy_ids", self.config.get("enemy_ids", [99])))

        pod_points = self._pod_points_list()
        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "IDENTIFY_AND_HOLD_DESIGNATED_AREA_Comparison.jpg")
        txt_path = os.path.join(session_folder, "IDENTIFY_AND_HOLD_DESIGNATED_AREA_Comparison.txt")

        if not pod_points:
            err_text = "There was an error while processing this comparison. Missing POD coordinates."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "IDENTIFY_AND_HOLD_DESIGNATED_AREA",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        ex_tracks, ex_fps = _load_position_tracks(expert_folder, enemy_ids)
        tr_tracks, tr_fps = _load_position_tracks(session_folder, enemy_ids)

        if not ex_tracks and not tr_tracks:
            err_text = "There was an error while processing this comparison. Missing or invalid PositionCache."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "IDENTIFY_AND_HOLD_DESIGNATED_AREA",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        n_pods = int(len(pod_points))
        ex_items = _select_top_tracks(ex_tracks, n_pods)
        tr_items = _select_top_tracks(tr_tracks, n_pods)

        ex_assign = self._assign_pods(ex_items, pod_points) if ex_items else {}
        tr_assign = self._assign_pods(tr_items, pod_points) if tr_items else {}

        ex_score, ex_per_pod, ex_t, ex_curve = _hold_stats(ex_items, pod_points, ex_assign, ex_fps)
        tr_score, tr_per_pod, tr_t, tr_curve = _hold_stats(tr_items, pod_points, tr_assign, tr_fps)

        ex_entry = _entry_map(ex_tracks)
        tr_entry = _entry_map(tr_tracks)

        # Per-POD rows (one row per POD index)
        rows: List[Dict[str, Any]] = []
        for pid in range(n_pods):
            ex = ex_per_pod.get(pid)
            tr = tr_per_pod.get(pid)

            ex_tid = None if ex is None else int(ex.get("track_id", -1))
            tr_tid = None if tr is None else int(tr.get("track_id", -1))

            rows.append(
                {
                    "pod_idx": int(pid),
                    "expert_entry_number": "" if ex_tid is None else ex_entry.get(int(ex_tid), ""),
                    "expert_id": "" if ex_tid is None else int(ex_tid),
                    "expert_score": "" if ex is None else float(ex.get("score", 0.0)),
                    "trainee_entry_number": "" if tr_tid is None else tr_entry.get(int(tr_tid), ""),
                    "trainee_id": "" if tr_tid is None else int(tr_tid),
                    "trainee_score": "" if tr is None else float(tr.get("score", 0.0)),
                }
            )

        # ---- Plot: cumulative score over time + per-POD holding scores ----
        try:
            dfp = pd.DataFrame(rows)

            fig, (ax_ts, ax_bar) = plt.subplots(
                nrows=1,
                ncols=2,
                figsize=(14.5, 5.6),
                constrained_layout=True,
            )

            if ex_t:
                ax_ts.plot(ex_t, ex_curve, label="Expert", linewidth=2.5)
            if tr_t:
                ax_ts.plot(tr_t, tr_curve, label="Trainee", linewidth=2.5)

            ax_ts.set_xlabel("Seconds since first team entry")
            ax_ts.set_ylabel("Cumulative POD hold score")
            ax_ts.set_title("Holding assigned PODs over time")
            ax_ts.set_ylim(0.0, 1.05)
            ax_ts.grid(True, axis="y", linestyle="--", alpha=0.35)

            if dfp.empty:
                ax_bar.set_axis_off()
            else:
                dfp = dfp.copy().sort_values("pod_idx")
                x = dfp["pod_idx"].astype(int).tolist()
                idx = np.arange(len(x), dtype=float)
                width = 0.36

                ex_sc = pd.to_numeric(dfp["expert_score"], errors="coerce").fillna(0).to_numpy(dtype=float)
                tr_sc = pd.to_numeric(dfp["trainee_score"], errors="coerce").fillna(0).to_numpy(dtype=float)

                ax_bar.bar(idx - width / 2.0, ex_sc, width, label="Expert", alpha=0.9)
                ax_bar.bar(idx + width / 2.0, tr_sc, width, label="Trainee", alpha=0.9)

                for i in range(len(x)):
                    ax_bar.text(
                        idx[i] - width / 2.0,
                        ex_sc[i] + 0.02,
                        f"~{ex_sc[i]:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )
                    ax_bar.text(
                        idx[i] + width / 2.0,
                        tr_sc[i] + 0.02,
                        f"~{tr_sc[i]:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

                ax_bar.set_xticks(idx)
                ax_bar.set_xticklabels([f"P{int(i)}" for i in x])
                ax_bar.set_xlabel("POD")
                ax_bar.set_ylabel("Per-POD holding score")
                ax_bar.set_title("Per-POD holding score (labels show ~score)")
                ax_bar.set_ylim(0.0, 1.05)
                ax_bar.grid(True, axis="y", linestyle="--", alpha=0.35)

            # One legend outside (top-right)
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
                fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), framealpha=0.9)
                fig.subplots_adjust(right=0.82)

            plt.savefig(img_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
            plt.close(fig)
        except Exception:
            pass

        # ---- Summary (teammate-coverage style) ----
        ex_final = float(ex_score)
        tr_final = float(tr_score)
        delta_final = float(tr_final - ex_final)

        # Decide better/similar/worse by direct comparison (no epsilon ranges).
        if tr_final == ex_final:
            score_part = (
                f"Overall POD holding looks similar to the expert "
                f"(Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
            )
        elif tr_final > ex_final:
            score_part = (
                f"Overall POD holding looks better than the expert "
                f"(Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
            )
        else:
            score_part = (
                f"Overall POD holding looks worse than the expert "
                f"(Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
            )

        diffs: List[float] = []
        for r in rows:
            ex_sc = r.get("expert_score")
            tr_sc = r.get("trainee_score")
            if ex_sc in (None, "") or tr_sc in (None, ""):
                continue
            diffs.append(float(tr_sc) - float(ex_sc))
        avg_ds = float(np.mean(diffs)) if diffs else None

        if avg_ds is None:
            out_part = "On average, per-POD holding scores couldn't be compared (missing paired POD scores)."
        elif avg_ds == 0:
            out_part = "On average, the trainee's per-POD holding scores were similar to the expert."
        elif avg_ds > 0:
            out_part = (
                f"On average, the trainee's per-POD holding scores were about {abs(avg_ds):.2f} higher than the expert "
                f"(better holding)."
            )
        else:
            out_part = f"On average, the trainee's per-POD holding scores were about {abs(avg_ds):.2f} lower than the expert."

        # Per-POD table
        lines = [
            "POD, Expert Entrant#, Expert ID, Expert ~Score, Trainee Entrant#, Trainee ID, Trainee ~Score, Score Δ (T−E), Performance",
        ]

        for r in rows:
            pid = r.get("pod_idx")
            ex_ent = r.get("expert_entry_number")
            ex_id = r.get("expert_id")
            tr_ent = r.get("trainee_entry_number")
            tr_id = r.get("trainee_id")
            ex_sc = r.get("expert_score")
            tr_sc = r.get("trainee_score")
            ex_sc_f = None if ex_sc in (None, "") else float(ex_sc)
            tr_sc_f = None if tr_sc in (None, "") else float(tr_sc)

            if ex_sc_f is None or tr_sc_f is None:
                dsc = None
            else:
                dsc = float(tr_sc_f - ex_sc_f)

            if dsc is None:
                perf = "N/A"
            elif dsc == 0:
                perf = "SIMILAR"
            elif dsc > 0:
                perf = "BETTER"
            else:
                perf = "WORSE"

            lines.append(
                f"P{int(pid) if pid is not None else 'N/A'}, "
                f"{ex_ent if ex_ent != '' else 'N/A'}, {ex_id if ex_id != '' else 'N/A'}, "
                f"~{(0.0 if ex_sc_f is None else ex_sc_f):.2f}, "
                f"{tr_ent if tr_ent != '' else 'N/A'}, {tr_id if tr_id != '' else 'N/A'}, "
                f"~{(0.0 if tr_sc_f is None else tr_sc_f):.2f}, "
                f"{('N/A' if dsc is None else f'{dsc:+.2f}')}, "
                f"{perf}"
            )

        details_csv = "\n".join(lines)
        text = score_part + "\n" + out_part + "\n\n" + details_csv

        # Save a structured (fixed-width) table in the TXT file, while still returning CSV-like details in `Text`.
        try:
            header_cells = [c.strip() for c in lines[0].split(",")]
            data_cells = [[c.strip() for c in ln.split(",")] for ln in lines[1:]]
            details_pretty = _pretty_table(header_cells, data_cells)

            saved_text = score_part + "\n" + out_part + "\n\n" + details_pretty + "\n"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(saved_text)
        except Exception:
            # Non-fatal: still return the summary even if file write fails
            pass

        return {
            "Name": "IDENTIFY_AND_HOLD_DESIGNATED_AREA",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }