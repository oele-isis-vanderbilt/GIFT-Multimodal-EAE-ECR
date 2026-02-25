import os
import glob
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metric import AbstractMetric
from .utils import arg_first_non_null


class TotalEntryTime_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "TOTAL_TIME_OF_ENTRY"
        self.num_tracks = len(config["POD"])
        self.entry_starts = []
        # Configurable threshold (seconds) for full score
        self.entry_time_threshold_sec = float(config.get("entry_time_threshold_sec", 2.0))

    def process(self, ctx):
        """Compute team entry span inputs (first frames of the main N tracks)."""
        enemy_ids = self.config.get("enemy_ids", [99])
        tracks = {
            tid: traj
            for tid, traj in ctx.tracks_by_id.items()
            if tid not in enemy_ids
        }
        longest = sorted(tracks.values(), key=lambda t: len(t), reverse=True)[:self.num_tracks]
        sorted_tracks = sorted(longest, key=lambda t: arg_first_non_null(t))
        self.entry_starts = [arg_first_non_null(trk) for trk in sorted_tracks]
    
    def getFinalScore(self) -> float:
        if len(self.entry_starts) < 2:
            return -1
        delta_frames = self.entry_starts[-1] - self.entry_starts[0]
        delta_secs = delta_frames / float(self.config.get("frame_rate", 30.0) or 30.0)
        threshold = float(self.entry_time_threshold_sec)
        if delta_secs <= threshold:
            return 1.0
        overrun = delta_secs - threshold
        return round(self._exp_penalty(overrun, threshold), 2)

    def _exp_penalty(self, overrun: float, limit: float) -> float:
        """Exponential decay from 1→0 as lateness increases."""
        if overrun >= limit:
            return 0.0
        return float(np.exp(-(overrun) / (limit - overrun)))

    # --------------------------------------------------------------------- #
    # Expert comparison (folder-based)
    # --------------------------------------------------------------------- #
    @staticmethod
    def expertCompare(session_folder: str, expert_folder: str, map_image=None, config=None):
        """Compare trainee vs expert team entry span.

        Reads the latest `*_PositionCache.txt` from each folder, writes
        `TOTAL_TIME_OF_ENTRY_Comparison.jpg`, and returns a short summary.
        """

        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_position_cache(folder: str, enemy_ids: List[int]) -> pd.DataFrame:
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

            # Filter out enemies
            out = out[~out["id"].isin([int(x) for x in enemy_ids])].copy()
            return out

        def _select_entry_rows(pos: pd.DataFrame, n_tracks: int) -> List[Dict[str, int]]:
            """Return ordered entrant rows for the longest N tracks.

            Each row:
              {"entry_number": i (1-based), "id": track_id, "start_frame": first_frame, "samples": count}

            Selection logic mirrors the metric:
              • rank by track length (samples) desc
              • take top N
              • order the selected tracks by first appearance (start_frame) asc
            """
            if pos is None or pos.empty:
                return []

            counts = pos.groupby("id").size().sort_values(ascending=False)
            starts = pos.groupby("id")["frame"].min()

            top_ids = counts.index.tolist()[: max(0, int(n_tracks))] if n_tracks > 0 else counts.index.tolist()
            if not top_ids:
                return []

            sub = pd.DataFrame({
                "id": [int(t) for t in top_ids],
                "samples": [int(counts.loc[t]) for t in top_ids],
                "start_frame": [int(starts.loc[t]) for t in top_ids],
            })
            sub = sub.sort_values("start_frame", ascending=True)

            rows: List[Dict[str, int]] = []
            for i, r in enumerate(sub.itertuples(index=False), start=1):
                rows.append({
                    "entry_number": int(i),
                    "id": int(r.id),
                    "start_frame": int(r.start_frame),
                    "samples": int(r.samples),
                })
            return rows

        def _exp_penalty(overrun: float, limit: float) -> float:
            if overrun >= limit:
                return 0.0
            return float(np.exp(-(overrun) / (limit - overrun)))

        def _score(delta_secs: float, threshold: float = 2.0) -> float:
            if delta_secs <= threshold:
                return 1.0
            overrun = delta_secs - threshold
            if overrun >= threshold:
                return 0.0
            return round(_exp_penalty(overrun, threshold), 2)

        def _trend(delta: Optional[float], eps: float = 0.05) -> str:
            """Human-friendly comparison label."""
            if delta is None:
                return "N/A"
            if abs(delta) <= eps:
                return "SIMILAR"
            return "SLOWER" if delta > 0 else "FASTER"

        # Output artifacts
        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "TOTAL_TIME_OF_ENTRY_Comparison.jpg")
        txt_path = os.path.join(session_folder, "TOTAL_TIME_OF_ENTRY_Comparison.txt")

        # Config-derived parameters
        fps = 30.0
        enemy_ids = [99]
        n_tracks_cfg: Optional[int] = None
        threshold = 2.0
        if isinstance(config, dict):
            fps = float(config.get("frame_rate", fps) or fps)
            enemy_ids = config.get("enemy_ids", enemy_ids) or enemy_ids
            threshold = float(config.get("entry_time_threshold_sec", threshold) or threshold)
            if "POD" in config and config.get("POD") is not None:
                try:
                    n_tracks_cfg = int(len(config.get("POD")))
                except Exception:
                    n_tracks_cfg = None

        # Load caches
        pos_ex = _load_position_cache(expert_folder, enemy_ids)
        pos_tr = _load_position_cache(session_folder, enemy_ids)

        if (pos_ex is None or pos_ex.empty) and (pos_tr is None or pos_tr.empty):
            err_text = "There was an error while processing this comparison. Missing or invalid PositionCache."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "TOTAL_TIME_OF_ENTRY",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Decide N (tracks to consider)
        if n_tracks_cfg is not None and n_tracks_cfg > 0:
            n_tracks = n_tracks_cfg
        else:
            n_ex = int(pos_ex["id"].nunique()) if pos_ex is not None and not pos_ex.empty else 0
            n_tr = int(pos_tr["id"].nunique()) if pos_tr is not None and not pos_tr.empty else 0
            n_tracks = max(0, min(n_ex, n_tr))

        # If one side has no tracks, fall back to the other side's count
        if n_tracks == 0:
            n_tracks = int(pos_tr["id"].nunique()) if pos_tr is not None and not pos_tr.empty else int(pos_ex["id"].nunique())

        # Compute ordered entrant rows
        ex_entries = _select_entry_rows(pos_ex, n_tracks) if pos_ex is not None else []
        tr_entries = _select_entry_rows(pos_tr, n_tracks) if pos_tr is not None else []

        ex_entry_frames = [r["start_frame"] for r in ex_entries]
        tr_entry_frames = [r["start_frame"] for r in tr_entries]

        def _span_seconds(frames: List[int]) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[float]]:
            if not frames or len(frames) < 2:
                return None, None, None, None
            first_f = int(frames[0])
            last_f = int(frames[-1])
            delta_f = int(last_f - first_f)
            delta_s = float(delta_f / fps) if fps > 0 else None
            return first_f, last_f, delta_f, delta_s

        _, _, _, ex_delta_secs = _span_seconds(ex_entry_frames)
        _, _, _, tr_delta_secs = _span_seconds(tr_entry_frames)

        ex_score = _score(ex_delta_secs, threshold) if ex_delta_secs is not None else None
        tr_score = _score(tr_delta_secs, threshold) if tr_delta_secs is not None else None

        # Delta (trainee - expert)
        dt = None
        if ex_delta_secs is not None and tr_delta_secs is not None:
            dt = float(tr_delta_secs - ex_delta_secs)


        # Generate a simple figure (timeline)
        try:
            fig, ax = plt.subplots(figsize=(11.5, 3.6), constrained_layout=True)

            def _to_rel_seconds(frames: List[int]) -> List[float]:
                if not frames:
                    return []
                f0 = frames[0]
                return [float((f - f0) / fps) for f in frames]

            ex_t = _to_rel_seconds(ex_entry_frames)
            tr_t = _to_rel_seconds(tr_entry_frames)

            expert_color = "tab:blue"
            trainee_color = "tab:orange"
            threshold_color = "tab:gray"

            # Plot expert (y=1) and trainee (y=0)
            if ex_t:
                ax.scatter(ex_t, [1.0] * len(ex_t), label="Expert", color=expert_color)
                ax.hlines(1.0, min(ex_t), max(ex_t), linewidth=2, colors=expert_color)
                ax.text(max(ex_t) + 0.05, 1.0, f"{(ex_delta_secs or 0.0):.2f}s", va="center", color=expert_color)

            if tr_t:
                ax.scatter(tr_t, [0.0] * len(tr_t), label="Trainee", color=trainee_color)
                ax.hlines(0.0, min(tr_t), max(tr_t), linewidth=2, colors=trainee_color)
                ax.text(max(tr_t) + 0.05, 0.0, f"{(tr_delta_secs or 0.0):.2f}s", va="center", color=trainee_color)

            ax.axvline(threshold, linestyle="--", linewidth=1.5, label=f"{threshold:.2f}s threshold", color=threshold_color)

            ax.set_yticks([0.0, 1.0])
            ax.set_yticklabels(["Trainee", "Expert"])
            ax.set_xlabel("Seconds since first team entry")
            ax.set_title("Total time of entry: Expert vs Trainee")
            ax.grid(True, axis="x", linestyle="--", alpha=0.4)
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, framealpha=0.9)
            fig.subplots_adjust(right=0.78)

            plt.savefig(img_path, dpi=150)
            plt.close(fig)
        except Exception:
            pass

        # Build response text (no per-entry details)
        zero_at = 2.0 * threshold

        ex_span_str = "N/A" if ex_delta_secs is None else f"{float(ex_delta_secs):.2f}s"
        tr_span_str = "N/A" if tr_delta_secs is None else f"{float(tr_delta_secs):.2f}s"

        ex_score_str = "N/A" if ex_score is None else f"{float(ex_score):.2f}"
        tr_score_str = "N/A" if tr_score is None else f"{float(tr_score):.2f}"

        if dt is None:
            time_part = "I couldn't compare total entry time (missing enough entrants on either the expert or trainee side)."
        else:
            if abs(dt) <= 0.05:
                time_part = "Overall entry timing looks about the same as the expert."
            elif dt < 0:
                time_part = f"Overall, the trainee team got everyone in about {abs(dt):.2f}s faster than the expert."
            else:
                time_part = f"Overall, the trainee team was about {abs(dt):.2f}s slower than the expert to get everyone in."

        scores_part = (
            f"Spans / scores → Trainee: {tr_span_str} (score {tr_score_str}), "
            f"Expert: {ex_span_str} (score {ex_score_str})."
        )

        thresholds_part = (
            f"Thresholds: full score when team entry span ≤ {threshold:.2f}s; "
        )

        text = time_part + "\n" + scores_part + "\n" + thresholds_part

        # Also save the returned summary text alongside the image
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception:
            # Non-fatal: still return the summary even if file write fails
            pass

        return {
            "Name": "TOTAL_TIME_OF_ENTRY",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }
