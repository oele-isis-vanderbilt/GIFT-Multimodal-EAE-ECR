import os
import glob
from typing import Dict, Optional, Tuple, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metric import AbstractMetric


class CapturePodTime_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "POD_CAPTURE_TIME"

        # Ordered per-POD limits (seconds). May be shorter than actual POD count.
        self.time_limits: List[float] = config.get("pod_time_limits", [1, 3, 1.5, 2])


        # Filled in process()
        self._scores_by_soldier: Optional[Dict[int, float]] = None

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _exp_penalty(self, overrun: float, limit: float) -> float:
        """
        Exponential penalty that smoothly decreases from 1 to 0 as overrun goes from 0 to limit.
        """
        # If overrun reaches or exceeds the limit, score is zero.
        if overrun >= limit:
            return 0.0
        # Smooth exponential decay to zero at overrun == limit
        return float(np.exp(- (overrun) / (limit - overrun)))

    def _score_single_pod(self, capture_time: Optional[float], limit: float) -> float:
        """
        Score one POD event.

        Parameters
        ----------
        capture_time : float or None
            Seconds from scenario start until capture, or None if never captured.
        limit : float
            Allowed time before penalties kick in. ≤ 0 means no grace period.

        Returns
        -------
        float
            0.0 – 1.0
        """
        if capture_time is None:
            return 0.0

        # Full score up to the limit
        if capture_time <= limit:
            return 1.0

        # Compute overrun beyond the limit
        overrun = capture_time - limit
        # Apply smooth exponential penalty that reaches zero at double the limit
        return self._exp_penalty(overrun, limit)

    # --------------------------------------------------------------------- #
    # Public metric interface
    # --------------------------------------------------------------------- #
    def process(self, ctx):
        """
        Populate per-soldier POD scores from ctx.pod_capture.
        """
        pod_capture = getattr(ctx, "pod_capture", {})
        if not pod_capture:
            self._scores_by_soldier = {}
            return

        # Ensure we have at least as many limits as PODs
        max_idx = max(pod_capture.keys())
        if len(self.time_limits) == 0:
            self.time_limits = [0.0] * (max_idx + 1)
        elif len(self.time_limits) <= max_idx:
            self.time_limits.extend([self.time_limits[-1]] * (max_idx + 1 - len(self.time_limits)))

        per_person: Dict[int, List[float]] = {}

        for idx, info in pod_capture.items():
            soldier_id = info.get("assigned_id")
            if soldier_id is None:
                # POD never assigned → cannot attribute score; skip
                continue

            capture_time = info.get("capture_time_sec")
            limit = self.time_limits[idx]
            score = self._score_single_pod(capture_time, limit)

            per_person.setdefault(soldier_id, []).append(score)

        # Average per person
        self._scores_by_soldier = {
            sid: round(float(np.mean(scores)), 2)
            for sid, scores in per_person.items()
        }

    def getFinalScore(self) -> float:
        """
        Return a single overall score: the mean of each soldier's average POD scores.
        If process() has not run or no soldiers present, returns 0.0.
        """
        if not self._scores_by_soldier:
            return 0.0
        # Compute the mean of per-soldier averages
        overall = float(np.mean(list(self._scores_by_soldier.values())))
        return round(overall, 2)

    # --------------------------------------------------------------------- #
    # Expert comparison (folder-based)
    # --------------------------------------------------------------------- #
    @staticmethod
    def expertCompare(session_folder: str, expert_folder: str, map_image=None, config=None):
        """Compare trainee vs expert POD capture time using the same scoring logic.

        Reads the latest `*_PodCache.txt` (and `*_PositionCache.txt` for entry order),
        writes `POD_CAPTURE_TIME_Comparison.jpg`, and returns a text summary + per-POD lines.
        """

        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_pod_cache(folder: str) -> pd.DataFrame:
            path = _pick_latest(folder, "*_PodCache.txt")
            if path is None:
                raise FileNotFoundError(f"No PodCache found in {folder}")

            df = pd.read_csv(path)
            if df is None or df.empty:
                return pd.DataFrame(columns=["pod_idx", "assigned_id", "capture_time_sec", "capture_frame"])

            # Be tolerant to minor column name variations
            cols = {c.lower(): c for c in df.columns}
            pod_col = cols.get("pod_idx") or cols.get("pod") or cols.get("podindex")
            aid_col = cols.get("assigned_id") or cols.get("assigned") or cols.get("track_id")
            t_col = cols.get("capture_time_sec") or cols.get("capture_time") or cols.get("time")
            f_col = cols.get("capture_frame") or cols.get("frame")

            if pod_col is None or aid_col is None or t_col is None:
                raise ValueError(f"Unexpected PodCache format: {path}")

            out = pd.DataFrame({
                "pod_idx": df[pod_col],
                "assigned_id": df[aid_col],
                "capture_time_sec": df[t_col],
            })
            if f_col is not None:
                out["capture_frame"] = df[f_col]
            else:
                out["capture_frame"] = np.nan

            # Normalize types
            out["pod_idx"] = pd.to_numeric(out["pod_idx"], errors="coerce").astype("Int64")
            out["assigned_id"] = pd.to_numeric(out["assigned_id"], errors="coerce").astype("Int64")
            out["capture_time_sec"] = pd.to_numeric(out["capture_time_sec"], errors="coerce")
            out["capture_frame"] = pd.to_numeric(out["capture_frame"], errors="coerce")

            out = out.dropna(subset=["pod_idx"]).copy()
            out["pod_idx"] = out["pod_idx"].astype(int)
            return out

        def _exp_penalty(overrun: float, limit: float) -> float:
            # Mirrors the metric implementation
            if limit <= 0:
                return 0.0
            if overrun >= limit:
                return 0.0
            return float(np.exp(-(overrun) / (limit - overrun)))

        def _score_single_pod(capture_time: Optional[float], limit: float) -> float:
            # Treat None / NaN / pandas NA as not captured
            if capture_time is None or pd.isna(capture_time):
                return 0.0
            ct = float(capture_time)
            if ct <= limit:
                return 1.0
            overrun = ct - limit
            return _exp_penalty(overrun, limit)

        def _ensure_limits(max_pod_idx: int) -> List[float]:
            # Limits from config if provided; else default fallback (metric default)
            limits = None
            if isinstance(config, dict):
                limits = config.get("pod_time_limits")
            if not limits:
                limits = [1.0, 3.0, 1.5, 2.0]
            if max_pod_idx < 0:
                return limits
            if len(limits) == 0:
                limits = [0.0] * (max_pod_idx + 1)
            elif len(limits) <= max_pod_idx:
                limits.extend([limits[-1]] * (max_pod_idx + 1 - len(limits)))
            return limits

        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "POD_CAPTURE_TIME_Comparison.jpg")
        txt_path = os.path.join(session_folder, "POD_CAPTURE_TIME_Comparison.txt")
        def _load_position_cache(folder: str) -> pd.DataFrame:
            path = _pick_latest(folder, "*_PositionCache.txt")
            if path is None:
                return pd.DataFrame(columns=["frame", "id"])

            try:
                df = pd.read_csv(path)
            except Exception:
                return pd.DataFrame(columns=["frame", "id"])

            if df is None or df.empty:
                return pd.DataFrame(columns=["frame", "id"])

            cols = {c.lower(): c for c in df.columns}
            f_col = cols.get("frame")
            id_col = cols.get("id") or cols.get("track_id") or cols.get("track")
            if f_col is None or id_col is None:
                return pd.DataFrame(columns=["frame", "id"])

            out = pd.DataFrame({
                "frame": pd.to_numeric(df[f_col], errors="coerce"),
                "id": pd.to_numeric(df[id_col], errors="coerce"),
            }).dropna(subset=["frame", "id"]).copy()

            if out.empty:
                return pd.DataFrame(columns=["frame", "id"])

            out["frame"] = out["frame"].astype(int)
            out["id"] = out["id"].astype(int)

            # Ignore sentinel ID used elsewhere in the pipeline
            out = out[out["id"] != 99].copy()
            return out

        def _entry_map_from_position_cache(folder: str) -> Dict[int, int]:
            pos = _load_position_cache(folder)
            if pos is None or pos.empty:
                return {}

            # Entry time = first frame the ID appears
            starts = pos.groupby("id")["frame"].min().sort_values()
            entry_map: Dict[int, int] = {}
            for i, tid in enumerate(starts.index.tolist()):
                entry_map[int(tid)] = int(i + 1)
            return entry_map

        try:
            df_expert = _load_pod_cache(expert_folder)
            df_trn = _load_pod_cache(session_folder)
            expert_entry_map = _entry_map_from_position_cache(expert_folder)
            trainee_entry_map = _entry_map_from_position_cache(session_folder)
        except Exception:
            return {
                "Name": "POD_CAPTURE_TIME",
                "Type": "Single",
                "ImgLocation": img_path,
                "Text": "There was an error while processing this comparison. Missing or invalid PodCache.",
            }

        if (df_expert is None or df_expert.empty) and (df_trn is None or df_trn.empty):
            return {
                "Name": "POD_CAPTURE_TIME",
                "Type": "Single",
                "ImgLocation": img_path,
                "Text": "There was an error while processing this comparison. No POD entries found.",
            }

        # Union of pod indices across both folders
        pod_ids = sorted(set(df_expert["pod_idx"].tolist()) | set(df_trn["pod_idx"].tolist()))
        max_pod_idx = max(pod_ids) if pod_ids else -1
        limits = _ensure_limits(max_pod_idx)

        # Index by pod_idx for quick lookup
        ex_by_pod = df_expert.set_index("pod_idx") if not df_expert.empty else pd.DataFrame().set_index(pd.Index([]))
        tr_by_pod = df_trn.set_index("pod_idx") if not df_trn.empty else pd.DataFrame().set_index(pd.Index([]))

        rows: List[Dict[str, Any]] = []
        ex_scores_by_soldier: Dict[int, List[float]] = {}
        tr_scores_by_soldier: Dict[int, List[float]] = {}

        for pod_idx in pod_ids:
            limit = float(limits[pod_idx]) if pod_idx < len(limits) else float(limits[-1])

            # Expert
            ex_assigned = None
            ex_time = None
            ex_frame = None
            if pod_idx in ex_by_pod.index:
                ex_row = ex_by_pod.loc[pod_idx]
                # Handle possible duplicate pod rows by taking the first
                if isinstance(ex_row, pd.DataFrame):
                    ex_row = ex_row.iloc[0]
                ex_assigned = ex_row.get("assigned_id")
                ex_time = ex_row.get("capture_time_sec")
                ex_frame = ex_row.get("capture_frame")

            # Trainee
            tr_assigned = None
            tr_time = None
            tr_frame = None
            if pod_idx in tr_by_pod.index:
                tr_row = tr_by_pod.loc[pod_idx]
                if isinstance(tr_row, pd.DataFrame):
                    tr_row = tr_row.iloc[0]
                tr_assigned = tr_row.get("assigned_id")
                tr_time = tr_row.get("capture_time_sec")
                tr_frame = tr_row.get("capture_frame")

            ex_score = _score_single_pod(ex_time, limit)
            tr_score = _score_single_pod(tr_time, limit)

            # Aggregate per-soldier scores (separately per folder)
            if ex_assigned is not None and not pd.isna(ex_assigned):
                ex_scores_by_soldier.setdefault(int(ex_assigned), []).append(ex_score)
            if tr_assigned is not None and not pd.isna(tr_assigned):
                tr_scores_by_soldier.setdefault(int(tr_assigned), []).append(tr_score)

            # Time delta: trainee - expert (negative => trainee faster)
            dt = None
            if ex_time is not None and tr_time is not None and not pd.isna(ex_time) and not pd.isna(tr_time):
                dt = float(tr_time) - float(ex_time)

            def _trend(d: Optional[float]) -> str:
                if d is None:
                    return "N/A"
                if d < 0:
                    return "FASTER"
                if d > 0:
                    return "SLOWER"
                return "MATCH"

            rows.append({
                "pod_idx": int(pod_idx),
                "time_limit_sec": round(limit, 3),
                "expert_entry_number": ("" if ex_assigned is None or pd.isna(ex_assigned) else expert_entry_map.get(int(ex_assigned), "")),
                "expert_id": ("" if ex_assigned is None or pd.isna(ex_assigned) else int(ex_assigned)),
                "expert_capture_time_sec": ("" if ex_time is None or pd.isna(ex_time) else round(float(ex_time), 3)),
                "expert_capture_frame": ("" if ex_frame is None or pd.isna(ex_frame) else int(ex_frame)),
                "expert_pod_score": round(float(ex_score), 3),
                "trainee_entry_number": ("" if tr_assigned is None or pd.isna(tr_assigned) else trainee_entry_map.get(int(tr_assigned), "")),
                "trainee_id": ("" if tr_assigned is None or pd.isna(tr_assigned) else int(tr_assigned)),
                "trainee_capture_time_sec": ("" if tr_time is None or pd.isna(tr_time) else round(float(tr_time), 3)),
                "trainee_capture_frame": ("" if tr_frame is None or pd.isna(tr_frame) else int(tr_frame)),
                "trainee_pod_score": round(float(tr_score), 3),
                "delta_time_sec_trainee_minus_expert": ("" if dt is None else round(float(dt), 3)),
                "trainee_vs_expert_time": _trend(dt),
                "delta_score_trainee_minus_expert": round(float(tr_score - ex_score), 3),
            })

        pods_df = pd.DataFrame(rows)

        # Generate the comparison graphic (only if we have at least one POD)
        try:
            CapturePodTime_Metric.__generateExpertCompareGraphic(
                output_path=img_path,
                pods_df=pods_df
            )
        except Exception:
            # If plotting fails, still return text + table.
            pass

        # Average time difference across PODs where both expert and trainee have a capture time.
        dt_series = pd.to_numeric(pods_df.get("delta_time_sec_trainee_minus_expert"), errors="coerce")
        dt_series = dt_series.dropna() if dt_series is not None else pd.Series([], dtype=float)
        avg_dt = float(dt_series.mean()) if dt_series is not None and len(dt_series) > 0 else None
        n_dt = int(len(dt_series)) if dt_series is not None else 0

        # Average score difference across PODs (score always defined per POD row)
        ds_series = pd.to_numeric(pods_df.get("delta_score_trainee_minus_expert"), errors="coerce")
        ds_series = ds_series.dropna() if ds_series is not None else pd.Series([], dtype=float)
        avg_ds = float(ds_series.mean()) if ds_series is not None and len(ds_series) > 0 else None

        # Count PODs with a recorded capture time on each side (for transparency)
        ex_times = pd.to_numeric(pods_df.get("expert_capture_time_sec"), errors="coerce")
        tr_times = pd.to_numeric(pods_df.get("trainee_capture_time_sec"), errors="coerce")

        ex_has = ex_times.notna() if ex_times is not None else pd.Series([], dtype=bool)
        tr_has = tr_times.notna() if tr_times is not None else pd.Series([], dtype=bool)

        ex_time_count = int(ex_has.sum()) if len(ex_has) else 0
        tr_time_count = int(tr_has.sum()) if len(tr_has) else 0
        total_pods = int(len(pod_ids))

        # Exclusions for the average-time gap are PODs where at least one side lacks a capture time.
        missing_expert = int((~ex_has & tr_has).sum()) if len(ex_has) else 0
        missing_trainee = int((ex_has & ~tr_has).sum()) if len(ex_has) else 0
        missing_both = int((~ex_has & ~tr_has).sum()) if len(ex_has) else total_pods
        excluded = missing_expert + missing_trainee + missing_both

        def _excluded_breakdown() -> str:
            parts = []
            if missing_expert:
                parts.append(f"{missing_expert} missing expert")
            if missing_trainee:
                parts.append(f"{missing_trainee} missing trainee")
            if missing_both:
                parts.append(f"{missing_both} missing both")
            return "; ".join(parts) if parts else "0"

        def _excluded_pods_str() -> str:
            # PODs excluded from avg time gap are those where at least one side lacks a capture time.
            if not len(ex_has) or not len(tr_has):
                return ""
            excluded_mask = ~(ex_has & tr_has)
            if not excluded_mask.any():
                return ""
            pod_list = pods_df.loc[excluded_mask, "pod_idx"].tolist() if "pod_idx" in pods_df.columns else []
            # Defensive stringify
            pod_list = [f"P{int(p)}" for p in pod_list if p is not None and not pd.isna(p)]
            return ", ".join(pod_list)

        def _exclusion_clause() -> str:
            if excluded <= 0:
                return ""
            pods = _excluded_pods_str()
            pods_part = f" ({pods})" if pods else ""
            return (
                f"; excluded {excluded} POD(s){pods_part} due to missing capture time(s) ({_excluded_breakdown()})"
            )

        if avg_dt is None:
            time_part = (
                "Average capture-time gap: N/A. "
                "This metric only averages time differences for PODs where BOTH expert and trainee have a capture time. "
                f"Here, expert has times for {ex_time_count}/{total_pods} PODs and trainee has times for {tr_time_count}/{total_pods} PODs, "
                "so there are no PODs with times on both sides to compare."
            )
        elif avg_dt < 0:
            time_part = (
                f"Average capture-time gap (T − E): {avg_dt:.2f}s (trainee faster). "
                f"Computed over {n_dt}/{total_pods} PODs where BOTH expert and trainee recorded a capture time"
                f"{_exclusion_clause()}."
            )
        elif avg_dt > 0:
            time_part = (
                f"Average capture-time gap (T − E): {avg_dt:.2f}s (trainee slower). "
                f"Computed over {n_dt}/{total_pods} PODs where BOTH expert and trainee recorded a capture time"
                f"{_exclusion_clause()}."
            )
        else:
            time_part = (
                "Average capture-time gap (T − E): 0.00s (match). "
                f"Computed over {n_dt}/{total_pods} PODs where BOTH expert and trainee recorded a capture time"
                f"{_exclusion_clause()}."
            )

        if avg_ds is None:
            score_part = "I couldn't compute an average score difference."
        elif avg_ds > 0:
            score_part = f"On score, the trainee came in about {abs(avg_ds):.3f} higher than the expert on average."
        elif avg_ds < 0:
            score_part = f"On score, the trainee came in about {abs(avg_ds):.3f} lower than the expert on average."
        else:
            score_part = "On score, the trainee matched the expert on average."

        # Thresholds (time limits) used in this comparison.
        thr_parts = []
        for pid in pod_ids:
            lim = float(limits[pid]) if pid < len(limits) else float(limits[-1])
            thr_parts.append(f"P{int(pid)}={lim:.2f}s")
        thresholds_part = "Time limits used (sec): " + ", ".join(thr_parts) if thr_parts else "Time limits used (sec): N/A"

        # Per-POD details
        lines = [
            "POD, Expert Entrant#, Expert ID, Trainee Entrant#, Trainee ID, Time Δ (T−E), Score Δ (T−E), Performance"
        ]

        for r in rows:
            pod_idx = r.get("pod_idx")
            e_ent = r.get("expert_entry_number")
            e_id = r.get("expert_id")
            t_ent = r.get("trainee_entry_number")
            t_id = r.get("trainee_id")
            dt = r.get("delta_time_sec_trainee_minus_expert")
            ds = r.get("delta_score_trainee_minus_expert")

            dt_str = "N/A" if dt in (None, "") else f"{float(dt):+.2f}s"
            ds_str = "N/A" if ds in (None, "") else f"{float(ds):+.3f}"

            if ds in (None, ""):
                perf = "N/A"
            else:
                ds_f = float(ds)
                if ds_f > 0:
                    perf = "BETTER"
                elif ds_f < 0:
                    perf = "WORSE"
                else:
                    perf = "SIMILAR"

            lines.append(
                f"P{int(pod_idx) if pod_idx is not None else 'N/A'}, "
                f"{e_ent if e_ent != '' else 'N/A'}, {e_id if e_id != '' else 'N/A'}, "
                f"{t_ent if t_ent != '' else 'N/A'}, {t_id if t_id != '' else 'N/A'}, "
                f"{dt_str}, {ds_str}, {perf}"
            )

        # Returned payload remains comma-separated (easy to parse)
        details_csv = "\n".join(lines)
        text = time_part + " " + score_part + "\n" + thresholds_part + "\n" + details_csv

        # Saved TXT uses a readable broken-line table
        def _broken_table(headers: List[str], data_rows: List[List[str]]) -> str:
            if not data_rows:
                return "(no rows)"

            widths = []
            for j, h in enumerate(headers):
                max_cell = max([len(h)] + [len(r[j]) for r in data_rows])
                widths.append(max_cell)

            sep = " | "

            def _fmt(values: List[str]) -> str:
                return sep.join(values[i].ljust(widths[i]) for i in range(len(headers)))

            line_parts = ["-" * w for w in widths]
            broken_line = sep.join(line_parts)

            out = [_fmt(headers), broken_line]
            out.extend(_fmt(r) for r in data_rows)
            return "\n".join(out)

        header = lines[0] if lines else ""
        pretty_headers = [h.strip() for h in header.split(",")] if header else []
        pretty_rows: List[List[str]] = []
        for ln in lines[1:]:
            parts = [p.strip() for p in ln.split(",")]
            while len(parts) < len(pretty_headers):
                parts.append("N/A")
            pretty_rows.append(parts[: len(pretty_headers)])

        details_pretty = _broken_table(pretty_headers, pretty_rows) if pretty_headers else "(no rows)"
        saved_text = time_part + " " + score_part + "\n" + thresholds_part + "\n\n" + details_pretty + "\n"

        # Save the readable table to TXT (but keep comma-separated text in the returned payload)
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(saved_text)
        except Exception:
            # Non-fatal: still return the summary even if file write fails
            pass

        return {
            "Name": "POD_CAPTURE_TIME",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }

    @staticmethod
    def __generateExpertCompareGraphic(output_path: str, pods_df: pd.DataFrame) -> None:
        """Create a simple grouped-bar chart comparing expert vs trainee capture time per POD.

        The chart is intentionally minimal and readable:
          • X-axis: POD index
          • Bars: expert_capture_time_sec and trainee_capture_time_sec
          • Markers: time_limit_sec per POD

        Saved to `output_path` (inside the session folder).
        """
        if pods_df is None or pods_df.empty:
            return

        # Ensure numeric
        df = pods_df.copy()
        df["pod_idx"] = pd.to_numeric(df["pod_idx"], errors="coerce")
        df = df.dropna(subset=["pod_idx"])
        if df.empty:
            return

        x = df["pod_idx"].astype(int).tolist()

        def _to_float_series(col: str) -> np.ndarray:
            s = pd.to_numeric(df[col], errors="coerce")
            return s.to_numpy(dtype=float)

        ex_t = _to_float_series("expert_capture_time_sec")
        tr_t = _to_float_series("trainee_capture_time_sec")
        lim = _to_float_series("time_limit_sec")
        lim2 = lim * 2.0

        # Replace NaNs with 0 for plotting bars (missing capture will be 0-height)
        ex_t_plot = np.nan_to_num(ex_t, nan=0.0)
        tr_t_plot = np.nan_to_num(tr_t, nan=0.0)

        width = 0.35
        idx = np.arange(len(x), dtype=float)

        fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)

        ax.bar(idx - width / 2.0, ex_t_plot, width, label="Expert")
        ax.bar(idx + width / 2.0, tr_t_plot, width, label="Trainee")

        # Time-limit markers (underscore marker)
        ax.scatter(idx, lim, marker="_", s=800, linewidths=3, label="Time limit")

        # 2× time-limit markers: score reaches 0 at/after this point
        mask2 = ~np.isnan(lim2)
        if np.any(mask2):
            ax.scatter(idx[mask2], lim2[mask2], marker="_", s=600, linewidths=2,
                       label="2× time limit (score = 0 from here)")

        ax.set_xticks(idx)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_xlabel("POD index")
        ax.set_ylabel("Capture time (sec)")
        ax.set_title("POD capture time: Expert vs Trainee (0 = not captured)")
        ax.legend(loc="upper right")

        # Make y-limits readable
        ymax = max(float(np.nanmax([ex_t_plot.max(), tr_t_plot.max(), np.nanmax(lim), np.nanmax(lim2)])), 1.0)
        ax.set_ylim(0, ymax * 1.15)

        # Light grid for readability
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        plt.savefig(output_path, dpi=150)
        plt.close(fig)