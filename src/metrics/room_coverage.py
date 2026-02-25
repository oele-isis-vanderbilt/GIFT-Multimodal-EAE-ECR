import os
import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metric import AbstractMetric


class RoomCoverage_Metric(AbstractMetric):
    """Final room visual coverage fraction (0–1) from context.room_coverage."""

    def __init__(self, config):
        super().__init__(config)
        self.metricName = "FLOOR_COVERAGE"

    def process(self, context):
        """Cache final coverage score and related coverage metadata."""
        coverage_data = getattr(context, "room_coverage", None)
        if coverage_data is None:
            self._final_score = 0.0
            return

        self._final_score = float(coverage_data.get("final_fraction", 0.0) or 0.0)

    def getFinalScore(self):
        """Return final coverage fraction."""
        return float(getattr(self, "_final_score", 0.0) or 0.0)

    # --------------------------------------------------------------------- #
    # Expert comparison (folder-based)
    # --------------------------------------------------------------------- #
    @staticmethod
    def expertCompare(session_folder: str, expert_folder: str, map_image=None, config=None):
        """Compare trainee vs expert final coverage fraction.

        Reads the latest `*_RoomCoverageCache.txt` (and `*_PositionCache.txt` for time alignment),
        writes `FLOOR_COVERAGE_Comparison.jpg`, and returns a short score summary.
        """

        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_position_cache_first_entry(folder: str, enemy_ids: List[int]) -> Optional[int]:
            """Return earliest frame any non-enemy ID appears in PositionCache."""
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
            out = out[~out["id"].isin(set(enemy_ids))].copy()
            if out.empty:
                return None
            return int(out["frame"].min())

        def _load_room_coverage_cache(folder: str) -> Tuple[pd.DataFrame, dict]:
            """Parse the RoomCoverageCache.txt format written by save_room_coverage_cache()."""
            path = _pick_latest(folder, "*_RoomCoverageCache.txt")
            if path is None:
                raise FileNotFoundError(f"No RoomCoverageCache found in {folder}")

            # Read file manually to handle the blank line + summary block
            with open(path, "r") as f:
                lines = [ln.strip() for ln in f.readlines()]

            # Find blank separator
            sep_idx = None
            for i, ln in enumerate(lines):
                if ln == "":
                    sep_idx = i
                    break

            curve_lines = lines
            summary_lines: List[str] = []
            if sep_idx is not None:
                curve_lines = lines[:sep_idx]
                summary_lines = lines[sep_idx + 1:]

            # Curve CSV part
            curve_df = pd.DataFrame(columns=["frame", "coverage_fraction"])
            if len(curve_lines) >= 2:
                # first line is header
                rows = []
                for ln in curve_lines[1:]:
                    if not ln:
                        continue
                    parts = [p.strip() for p in ln.split(",")]
                    if len(parts) < 2:
                        continue
                    rows.append({"frame": parts[0], "coverage_fraction": parts[1]})
                if rows:
                    curve_df = pd.DataFrame(rows)
                    curve_df["frame"] = pd.to_numeric(curve_df["frame"], errors="coerce")
                    curve_df["coverage_fraction"] = pd.to_numeric(curve_df["coverage_fraction"], errors="coerce")
                    curve_df = curve_df.dropna(subset=["frame", "coverage_fraction"]).copy()
                    curve_df["frame"] = curve_df["frame"].astype(int)
                    curve_df["coverage_fraction"] = curve_df["coverage_fraction"].astype(float)
                    curve_df = curve_df.sort_values("frame")

            # Summary block
            summary: dict = {
                "first_non_enemy_frame": None,
                "time_to_full_seconds": None,
                "final_fraction": None,
            }
            for ln in summary_lines:
                if not ln:
                    continue
                parts = [p.strip() for p in ln.split(",")]
                if len(parts) < 2:
                    continue
                k = parts[0]
                v = parts[1]
                if v == "":
                    val = None
                else:
                    try:
                        val = float(v)
                    except Exception:
                        val = None
                if k in summary:
                    summary[k] = val

            return curve_df, summary

        os.makedirs(session_folder, exist_ok=True)

        fig_path = os.path.join(session_folder, "FLOOR_COVERAGE_Comparison.jpg")
        txt_path = os.path.join(session_folder, "FLOOR_COVERAGE_Comparison.txt")

        # FPS selection
        fps = 30.0
        if isinstance(config, dict):
            try:
                fps = float(config.get("frame_rate", fps))
            except Exception:
                fps = 30.0

        enemy_ids = [99]
        if isinstance(config, dict) and isinstance(config.get("enemy_ids"), list) and len(config.get("enemy_ids")) > 0:
            enemy_ids = list(config.get("enemy_ids"))

        # Load caches
        try:
            expert_curve, expert_summary = _load_room_coverage_cache(expert_folder)
            trainee_curve, trainee_summary = _load_room_coverage_cache(session_folder)
        except Exception:
            err_text = "There was an error while processing this comparison. Missing or invalid RoomCoverageCache."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "FLOOR_COVERAGE",
                "Type": "Single",
                "ImgLocation": fig_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Removed unused first entry frames and series building

        expert_final = expert_summary.get("final_fraction", None)
        if expert_final is None:
            expert_final = float(expert_curve["coverage_fraction"].iloc[-1]) if not expert_curve.empty else 0.0

        trainee_final = trainee_summary.get("final_fraction", None)
        if trainee_final is None:
            trainee_final = float(trainee_curve["coverage_fraction"].iloc[-1]) if not trainee_curve.empty else 0.0

        # Final coverage comparison label (trainee vs expert)
        delta_final = float(trainee_final) - float(expert_final)

        # Plot: final coverage comparison (bar chart). Time-curve is handled by the coverage-time metric.
        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)

            # Explicit, consistent colors (Expert = blue, Trainee = orange)
            expert_color = "tab:blue"
            trainee_color = "tab:orange"
            threshold_color = "tab:green"

            x = np.arange(2, dtype=float)

            bars_ex = ax.bar(x[0], float(expert_final), width=0.6, color=expert_color, label="Expert")
            bars_tr = ax.bar(x[1], float(trainee_final), width=0.6, color=trainee_color, label="Trainee")

            # Full coverage reference line
            ax.axhline(1.0, linestyle="--", linewidth=2, color=threshold_color, label="Full coverage (1.0)")

            ax.set_xticks(x)
            ax.set_xticklabels(["Expert", "Trainee"])

            ax.set_ylabel("Final coverage fraction")
            ax.set_title("Final room visual coverage: Expert vs Trainee")
            ax.set_ylim(0.0, 1.05)
            ax.grid(True, axis="y", linestyle="--", alpha=0.35)

            # Bar value labels
            for rect in list(bars_ex) + list(bars_tr):
                h = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    h + 0.02,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

            # Legend outside the axes so it doesn't cover the plot
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0)

            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
        except Exception:
            # Plot is optional; ignore failures
            pass

        # Text summary
        d = float(trainee_final) - float(expert_final)
        if abs(d) <= 0.02:
            comp = "about the same as"
        elif d > 0:
            comp = "higher than"
        else:
            comp = "lower than"

        text = (
            f"Final floor coverage (score = final coverage fraction, 0–1): "
            f"Trainee {float(trainee_final):.3f} vs Expert {float(expert_final):.3f} "
            f"(Δ T−E = {d:+.3f}). The trainee is {comp} the expert."
        )

        # Also save the returned summary text alongside the image
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            # Non-fatal: still return the summary even if file write fails
            pass

        return {
            "Name": "FLOOR_COVERAGE",
            "Type": "Single",
            "ImgLocation": fig_path,
            "TxtLocation": txt_path,
            "Text": text,
        }