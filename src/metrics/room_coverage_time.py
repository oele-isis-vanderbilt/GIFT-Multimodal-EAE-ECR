import os
import glob
import io
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metric import AbstractMetric

class TotalRoomCoverageTime_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "TOTAL_FLOOR_COVERAGE_TIME"
        # Coverage time threshold in seconds (full score at/below this).
        self.threshold = config.get("coverage_time_threshold", 3)
        self.time_to_full = None

    def process(self, ctx):
        """
        Check if the room is completely covered. If so, record the time_to_full.
        Otherwise, record None to indicate no complete coverage.
        """
        room_cov = ctx.room_coverage
        if room_cov is None:
            # No room_coverage data available
            self.time_to_full = None
        else:
            # time_to_full is in seconds or None if full coverage never occurred
            self.time_to_full = room_cov.get("time_to_full")

    def getFinalScore(self) -> float:
        # If the room never reached full coverage, score is 0
        if self.time_to_full is None:
            return 0.0

        # Full score up to the threshold
        if self.time_to_full <= self.threshold:
            return 1.0

        # Compute overrun beyond the threshold
        overrun = self.time_to_full - self.threshold
        # Apply smooth exponential penalty that reaches zero at double the threshold
        return round(self._exp_penalty(overrun, self.threshold), 2)

    def _exp_penalty(self, overrun: float, limit: float) -> float:
        """Exponential decay from 1→0 as lateness increases."""
        if overrun >= limit:
            return 0.0
        return float(np.exp(-(overrun) / (limit - overrun)))

    @staticmethod
    def expertCompare(session_folder: str, expert_folder: str, map_image=None, config=None, **kwargs):
        """Compare trainee vs expert room visual coverage over time.

        Reads the latest `*_RoomCoverageCache.txt` (and `*_PositionCache.txt` for entry alignment),
        writes `TOTAL_FLOOR_COVERAGE_TIME_Comparison.jpg`, and returns a clear text summary.
        """

        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _parse_room_coverage_cache(path: str) -> Tuple[pd.DataFrame, Dict[str, Optional[float]]]:
            """Parse RoomCoverageCache into (per-frame df, summary dict).

            The file contains a CSV table, then an optional blank line, then key,value summary rows.
            """
            with open(path, "r") as f:
                lines = [ln.rstrip("\n") for ln in f.readlines()]

            # Split into "table block" + "summary block" by first blank line
            blank_idx = None
            for i, ln in enumerate(lines):
                if ln.strip() == "":
                    blank_idx = i
                    break

            table_lines = lines if blank_idx is None else lines[:blank_idx]
            summary_lines = [] if blank_idx is None else lines[blank_idx + 1 :]

            # --- parse per-frame table ---
            df = pd.DataFrame(columns=["frame", "coverage_fraction"])
            if len(table_lines) >= 2:
                try:
                    df = pd.read_csv(io.StringIO("\n".join(table_lines)))
                except Exception:
                    df = pd.DataFrame(columns=["frame", "coverage_fraction"])

            # normalize columns
            if not df.empty:
                cols = {c.lower(): c for c in df.columns}
                fcol = cols.get("frame")
                ccol = cols.get("coverage_fraction") or cols.get("coverage") or cols.get("fraction")

                if fcol is None or ccol is None:
                    df = pd.DataFrame(columns=["frame", "coverage_fraction"])
                else:
                    out = pd.DataFrame({
                        "frame": pd.to_numeric(df[fcol], errors="coerce"),
                        "coverage_fraction": pd.to_numeric(df[ccol], errors="coerce"),
                    }).dropna(subset=["frame"]).copy()
                    out["frame"] = out["frame"].astype(int)
                    out = out.sort_values("frame")
                    # clamp fraction into [0,1] if weird values appear
                    out["coverage_fraction"] = out["coverage_fraction"].clip(lower=0.0, upper=1.0)
                    df = out

            # --- parse summary block ---
            summary: Dict[str, Optional[float]] = {
                "first_non_enemy_frame": None,
                "time_to_full_seconds": None,
                "final_fraction": None,
            }

            for ln in summary_lines:
                if not ln.strip():
                    continue
                parts = [p.strip() for p in ln.split(",", 1)]
                if len(parts) != 2:
                    continue
                k, v = parts[0], parts[1]
                if k in summary:
                    if v == "":
                        summary[k] = None
                    else:
                        # first_non_enemy_frame is integer, others are floats
                        try:
                            if k == "first_non_enemy_frame":
                                summary[k] = float(int(float(v)))
                            else:
                                summary[k] = float(v)
                        except Exception:
                            summary[k] = None

            # If final_fraction missing but we have frames, infer it from last row
            if summary.get("final_fraction") is None and not df.empty:
                summary["final_fraction"] = float(df["coverage_fraction"].iloc[-1])

            return df, summary


        def _generate_plot(out_path: str, ex_post: pd.DataFrame, tr_post: pd.DataFrame, threshold_sec: Optional[float] = None) -> None:
            """Plot coverage fraction vs seconds since team entry."""
            if (ex_post is None or ex_post.empty) and (tr_post is None or tr_post.empty):
                return

            fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)

            if ex_post is not None and not ex_post.empty:
                ax.plot(ex_post["time_sec"], ex_post["coverage_fraction"], label="Expert")
            if tr_post is not None and not tr_post.empty:
                ax.plot(tr_post["time_sec"], tr_post["coverage_fraction"], label="Trainee")

            # If both runs reach full coverage, truncate x-axis to the later completion time.
            eps = 1e-6
            ex_full = None
            tr_full = None
            if ex_post is not None and not ex_post.empty:
                hit = ex_post[ex_post["coverage_fraction"] >= (1.0 - eps)]
                if not hit.empty:
                    ex_full = float(hit["time_sec"].iloc[0])
            if tr_post is not None and not tr_post.empty:
                hit = tr_post[tr_post["coverage_fraction"] >= (1.0 - eps)]
                if not hit.empty:
                    tr_full = float(hit["time_sec"].iloc[0])

            if ex_full is not None and tr_full is not None:
                ax.set_xlim(0.0, max(ex_full, tr_full) + 0.25)

            # Draw the scoring threshold as a reference line (same seconds axis as the plot).
            if threshold_sec is not None and np.isfinite(float(threshold_sec)):
                ax.axvline(
                    float(threshold_sec),
                    linestyle="--",
                    linewidth=1.2,
                    color="black",
                    alpha=0.8,
                    label=f"Threshold ({float(threshold_sec):.2f}s)",
                )

            ax.set_xlabel("Seconds since first non-enemy gaze")
            ax.set_ylabel("Coverage fraction")
            ax.set_title("Room visual coverage over time (expert vs trainee)")
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", linestyle="--", alpha=0.4)
            ax.legend(loc="lower right")

            plt.savefig(out_path, dpi=150)
            plt.close(fig)

        # ------------------------- main flow ------------------------- #

        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "TOTAL_FLOOR_COVERAGE_TIME_Comparison.jpg")
        txt_path = os.path.join(session_folder, "TOTAL_FLOOR_COVERAGE_TIME_Comparison.txt")

        expert_path = _pick_latest(expert_folder, "*_RoomCoverageCache.txt")
        trainee_path = _pick_latest(session_folder, "*_RoomCoverageCache.txt")

        if expert_path is None or trainee_path is None:
            err_text = "There was an error while processing this comparison. Missing RoomCoverageCache in expert and/or trainee folder."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "TOTAL_FLOOR_COVERAGE_TIME",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        try:
            ex_df, ex_sum = _parse_room_coverage_cache(expert_path)
            tr_df, tr_sum = _parse_room_coverage_cache(trainee_path)
        except Exception:
            err_text = "There was an error while processing this comparison. RoomCoverageCache could not be parsed."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "TOTAL_FLOOR_COVERAGE_TIME",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        if ex_df.empty and tr_df.empty:
            err_text = "There was an error while processing this comparison. No per-frame coverage data found."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "TOTAL_FLOOR_COVERAGE_TIME",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Seconds since team entry.
        # Prefer kwargs/config for fps; fall back to 30.
        fps = 30.0
        if config is not None:
            try:
                fps = float(config.get("frame_rate") or config.get("fps") or fps)
            except Exception:
                pass
        try:
            fps = float(kwargs.get("frame_rate") or kwargs.get("fps") or fps)
        except Exception:
            pass
        if fps <= 0:
            fps = 30.0

        def _load_position_cache_min_frame(folder: str) -> Optional[int]:
            """Return first frame where any non-enemy track appears (team entry)."""
            path = _pick_latest(folder, "*_PositionCache.txt")
            if path is None:
                return None
            try:
                df = pd.read_csv(path)
            except Exception:
                return None
            if df is None or df.empty:
                return None

            cols = {c.lower(): c for c in df.columns}
            f_col = cols.get("frame")
            id_col = cols.get("id") or cols.get("track_id") or cols.get("track")
            if f_col is None or id_col is None:
                return None

            out = pd.DataFrame({
                "frame": pd.to_numeric(df[f_col], errors="coerce"),
                "id": pd.to_numeric(df[id_col], errors="coerce"),
            }).dropna(subset=["frame", "id"]).copy()
            if out.empty:
                return None

            out["frame"] = out["frame"].astype(int)
            out["id"] = out["id"].astype(int)

            # Treat enemy IDs as non-team for entry purposes
            enemy_ids = [99]
            if config is not None:
                try:
                    enemy_ids = list(config.get("enemy_ids", enemy_ids))
                except Exception:
                    pass
            try:
                enemy_ids = list(kwargs.get("enemy_ids") or enemy_ids)
            except Exception:
                pass

            out = out[~out["id"].isin(enemy_ids)].copy()
            if out.empty:
                return None

            return int(out["frame"].min())

        def _start_frame_team_entry(folder: str, df: pd.DataFrame) -> int:
            # Prefer team entry from PositionCache
            entry = _load_position_cache_min_frame(folder)
            if entry is not None:
                return int(entry)
            # Fallback: use the earliest frame present in the coverage cache
            if df is not None and not df.empty:
                return int(df["frame"].min())
            return 1

        ex_start = _start_frame_team_entry(expert_folder, ex_df)
        tr_start = _start_frame_team_entry(session_folder, tr_df)

        def _post_entry_df(df: pd.DataFrame, start_frame: int) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame(columns=["frame", "coverage_fraction", "time_sec"])
            d = df[df["frame"] >= int(start_frame)].copy()
            if d.empty:
                return pd.DataFrame(columns=["frame", "coverage_fraction", "time_sec"])
            d["time_sec"] = (d["frame"] - int(start_frame)) / float(fps)
            d = d.sort_values("time_sec")
            return d[["frame", "coverage_fraction", "time_sec"]]

        ex_post = _post_entry_df(ex_df, ex_start)
        tr_post = _post_entry_df(tr_df, tr_start)


        # Score comparison using the same scoring logic as getFinalScore()
        threshold = 3.0
        if config is not None:
            try:
                threshold = float(config.get("coverage_time_threshold", threshold))
            except Exception:
                pass

        def _exp_penalty(overrun: float, limit: float) -> float:
            if overrun >= limit:
                return 0.0
            return float(np.exp(- (overrun) / (limit - overrun)))

        def _score_time_to_full(t: Optional[float], limit: float) -> float:
            if t is None or (isinstance(t, float) and np.isnan(t)):
                return 0.0
            tt = float(t)
            if tt <= limit:
                return 1.0
            return float(_exp_penalty(tt - limit, limit))

        expert_score = _score_time_to_full(ex_sum.get("time_to_full_seconds"), threshold)
        trainee_score = _score_time_to_full(tr_sum.get("time_to_full_seconds"), threshold)


        # Plot (always try; if it fails we still have summary)
        try:
            _generate_plot(img_path, ex_post, tr_post, threshold_sec=threshold)
        except Exception:
            pass

        # Text summary
        thr = float(threshold)
        zero_at = 2.0 * thr

        ex_time_full = ex_sum.get("time_to_full_seconds")
        tr_time_full = tr_sum.get("time_to_full_seconds")
        ex_final = ex_sum.get("final_fraction")
        tr_final = tr_sum.get("final_fraction")

        # Time-to-full comparison
        if ex_time_full is None and tr_time_full is None:
            time_part = "Neither run reached full coverage."
        elif ex_time_full is None and tr_time_full is not None:
            time_part = f"Trainee reached full coverage in {float(tr_time_full):.2f}s; expert did not reach full coverage."
        elif ex_time_full is not None and tr_time_full is None:
            time_part = f"Expert reached full coverage in {float(ex_time_full):.2f}s; trainee did not reach full coverage."
        else:
            dt = float(tr_time_full) - float(ex_time_full)
            if abs(dt) <= 0.05:
                time_part = "Trainee and expert were about the same on time to full coverage."
            elif dt < 0:
                time_part = f"Trainee reached full coverage about {abs(dt):.2f}s faster than the expert."
            else:
                time_part = f"Trainee reached full coverage about {abs(dt):.2f}s slower than the expert."

        # Score comparison
        ds = float(trainee_score - expert_score)
        if abs(ds) <= 0.01:
            score_part = f"Scores were basically the same (Trainee {trainee_score:.2f}, Expert {expert_score:.2f})."
        elif ds > 0:
            score_part = f"On score, the trainee came in higher (Trainee {trainee_score:.2f} vs Expert {expert_score:.2f}, +{abs(ds):.2f})."
        else:
            score_part = f"On score, the trainee came in lower (Trainee {trainee_score:.2f} vs Expert {expert_score:.2f}, -{abs(ds):.2f})."

        thresholds_part = (
            f"Thresholds: full score when time_to_full ≤ {thr:.2f}s"
        )

        parts = [time_part, score_part, thresholds_part]

        text = "\n".join(parts)

        # Also save the returned summary text alongside the image
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            # Non-fatal: still return the summary even if file write fails
            pass

        return {
            "Name": "TOTAL_FLOOR_COVERAGE_TIME",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }