from functools import cmp_to_key
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from typing import Optional, List, Dict, Tuple, Any

from .metric import AbstractMetric
from .utils import len_comparator, arg_first_comparator, arg_first_non_null


class EntranceHesitation_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "ENTRANCE_HESITATION"
        self.num_tracks = len(config["POD"])
        self.tracks = {}
        self.hesitation_threshold = config.get("HESITATION_THRESHOLD", 1)
        self.hesitation_threshold_second = config.get(
            "HESITATION_THRESHOLD_SECOND", self.hesitation_threshold * 2
        )

    def process(self, ctx):
        """
        Store filtered tracks from the unified context.
        """
        # Exclude all enemy tracks (supports multiple enemy IDs)
        enemy_ids = self.config.get("enemy_ids", [99])
        self.tracks = {
            idx: traj
            for idx, traj in ctx.tracks_by_id.items()
            if idx not in enemy_ids
        }
    
    def getFinalScore(self) -> float:
        # Get the n longest tracks and sort them by their start frames
        tracks = list(self.tracks.values())
        tracks.sort(key=cmp_to_key(len_comparator), reverse=True)
        tracks = tracks[:self.num_tracks]
        tracks.sort(key=cmp_to_key(arg_first_comparator))
        frame_starts = [arg_first_non_null(trk) for trk in tracks]

        if len(tracks) < 2:
            return -1

        time_diffs = []
        for i, start in enumerate(frame_starts[1:]):
            start_prev = frame_starts[i]
            difference = start - start_prev
            time_diff = difference / 30.
            if i == 1:
                bottom_clamp = max(0, time_diff - self.hesitation_threshold_second)
            else:
                bottom_clamp = max(0, time_diff - self.hesitation_threshold)
            score = np.exp(-np.power(bottom_clamp, 2)/0.5)
            time_diffs.append(score)
        
        return round(np.mean(time_diffs), 2)

    def expertCompare(self, session_folder: str, expert_folder: str, map_image=None):
        """Compare trainee vs expert entry timing gaps using the same hesitation scoring logic.

        Returns a short summary and a per-entry table, and writes `ENTRANCE_HESITATION_Comparison.jpg`.
        """
        os.makedirs(session_folder, exist_ok=True)
        img_path = os.path.join(session_folder, "ENTRANCE_HESITATION_Comparison.jpg")
        txt_path = os.path.join(session_folder, "ENTRANCE_HESITATION_Comparison.txt")

        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_position_cache(folder: str) -> Dict[int, List[Any]]:
            cache_path = _pick_latest(folder, "*_PositionCache.txt")
            if cache_path is None:
                raise FileNotFoundError(f"No PositionCache found in {folder}")

            data = np.genfromtxt(
                cache_path, delimiter=",", names=True, dtype=None, encoding=None
            )
            if data is None or data.size == 0:
                return {}

            if data.shape == ():
                data = np.array([data])

            if not all(
                name in data.dtype.names for name in ("frame", "id", "mapX", "mapY")
            ):
                raise ValueError(f"Unexpected PositionCache format: {cache_path}")

            frames = data["frame"].astype(int)
            ids = data["id"].astype(int)
            xs = data["mapX"].astype(float)
            ys = data["mapY"].astype(float)

            max_frame = int(frames.max()) if frames.size else 0
            tracks = {}
            for tid in np.unique(ids):
                if int(tid) in set(self.config.get("enemy_ids", [99])):
                    continue
                traj = [None] * max_frame
                mask = ids == tid
                for f, x, y in zip(frames[mask], xs[mask], ys[mask]):
                    if 1 <= int(f) <= max_frame:
                        traj[int(f) - 1] = (float(x), float(y))
                tracks[int(tid)] = traj

            return tracks

        try:
            expert_track = _load_position_cache(expert_folder)
            trainee_track = _load_position_cache(session_folder)
        except Exception:
            err_text = "There was an error while processing this comparison. Missing or invalid PositionCache."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "ENTRANCE_HESITATION",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        if not expert_track or not trainee_track:
            err_text = "There was an error while processing this comparison. No valid tracks found."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "ENTRANCE_HESITATION",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        def _select_for_scoring(tracks_by_id: Dict[int, List[Any]]) -> Tuple[List[int], List[int]]:
            """Mirror getFinalScore: pick N longest tracks, then sort by first appearance."""
            items = [(int(tid), trk) for tid, trk in (tracks_by_id or {}).items()]
            items.sort(key=cmp_to_key(lambda a, b: len_comparator(a[1], b[1])), reverse=True)
            items = items[: self.num_tracks]
            items.sort(key=cmp_to_key(lambda a, b: arg_first_comparator(a[1], b[1])))
            ids_ordered = [tid for tid, _ in items]
            starts_ordered = [arg_first_non_null(trk) for _, trk in items]
            return ids_ordered, starts_ordered

        expert_ids_ordered, frame_starts_expert = _select_for_scoring(expert_track)
        trainee_ids_ordered, frame_starts_trainee = _select_for_scoring(trainee_track)

        if len(frame_starts_expert) < 2 and len(frame_starts_trainee) < 2:
            err_text = "There was an error while processing this comparison. Not enough entrants to compute hesitation."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "ENTRANCE_HESITATION",
                "Type": "Single",
                "ImgLocation": img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Compare by entry index; allow mismatched counts.
        max_entries = max(len(frame_starts_expert), len(frame_starts_trainee))

        def _gap_seconds(frames: List[int], entry_idx: int) -> Optional[float]:
            """Gap (seconds) from previous entry to this entry_idx (1-based)."""
            if entry_idx <= 1 or entry_idx > len(frames):
                return None
            a = frames[entry_idx - 2]
            b = frames[entry_idx - 1]
            if a is None or b is None:
                return None
            return (int(b) - int(a)) / 30.0

        def _gap_score(gap_sec: Optional[float], gap_number: int) -> Optional[float]:
            """Per-gap score exactly as getFinalScore (gap_number starts at 1)."""
            if gap_sec is None:
                return None
            # Matches getFinalScore: the SECOND threshold applies to gap #2 (between entries 2→3).
            thr = self.hesitation_threshold_second if gap_number == 2 else self.hesitation_threshold
            bottom_clamp = max(0.0, float(gap_sec) - float(thr))
            return float(np.exp(-np.power(bottom_clamp, 2) / 0.5))

        def _expected_gap(gap_number: int) -> float:
            """No-penalty gap (seconds) used for this pair in scoring."""
            return float(self.hesitation_threshold_second if gap_number == 2 else self.hesitation_threshold)

        # Per-entry (gap-from-prev) comparisons. Entry 1 has no gap.
        rows = []
        valid_gap_diffs = []
        valid_score_diffs = []
        expert_scores = []
        trainee_scores = []

        for entry in range(1, max_entries + 1):
            expert_id = expert_ids_ordered[entry - 1] if entry <= len(expert_ids_ordered) else None
            trainee_id = trainee_ids_ordered[entry - 1] if entry <= len(trainee_ids_ordered) else None

            e_gap = _gap_seconds(frame_starts_expert, entry)
            t_gap = _gap_seconds(frame_starts_trainee, entry)

            # gap_number aligns with entry index: entry 2 => gap 1, entry 3 => gap 2, ...
            gap_number = max(0, entry - 1)
            e_score = _gap_score(e_gap, gap_number) if gap_number > 0 else None
            t_score = _gap_score(t_gap, gap_number) if gap_number > 0 else None

            if e_score is not None:
                expert_scores.append(e_score)
            if t_score is not None:
                trainee_scores.append(t_score)

            time_diff = (t_gap - e_gap) if (t_gap is not None and e_gap is not None) else None
            score_diff = (t_score - e_score) if (t_score is not None and e_score is not None) else None

            if time_diff is not None:
                valid_gap_diffs.append(float(time_diff))
            if score_diff is not None:
                valid_score_diffs.append(float(score_diff))

            if time_diff is None:
                trend = "N/A"
            elif time_diff < 0:
                trend = "FASTER"
            elif time_diff > 0:
                trend = "SLOWER"
            else:
                trend = "MATCH"

            rows.append(
                (
                    entry,
                    expert_id if expert_id is not None else "N/A",
                    trainee_id if trainee_id is not None else "N/A",
                    "N/A" if gap_number == 0 else f"{_expected_gap(gap_number):.2f}s",
                    "N/A" if time_diff is None else f"{time_diff:+.2f}s",
                    "N/A" if score_diff is None else f"{score_diff:+.3f}",
                    trend,
                )
            )

        avg_time_diff = float(np.mean(valid_gap_diffs)) if valid_gap_diffs else None
        avg_score_diff = float(np.mean(valid_score_diffs)) if valid_score_diffs else None

        expert_final = float(np.mean(expert_scores)) if expert_scores else None
        trainee_final = float(np.mean(trainee_scores)) if trainee_scores else None
        final_score_diff = (trainee_final - expert_final) if (trainee_final is not None and expert_final is not None) else None

        if avg_time_diff is None:
            time_part = "I couldn't compute an average time-gap difference (not enough paired entries)."
        else:
            if avg_time_diff < 0:
                time_part = f"On timing, the trainee was about {abs(avg_time_diff):.2f}s faster between entries on average."
            elif avg_time_diff > 0:
                time_part = f"On timing, the trainee was about {abs(avg_time_diff):.2f}s slower between entries on average."
            else:
                time_part = "On timing, the trainee was about the same as the expert between entries on average."

        # Explain expected gaps (no-penalty thresholds) used by the score.
        thr = float(self.hesitation_threshold)
        thr2 = float(self.hesitation_threshold_second)
        expected_part = f"Expected (no-penalty) gaps are about {thr:.2f}s for most pairs, and about {thr2:.2f}s between entries 2→3." 

        if final_score_diff is None:
            score_part = "I couldn't compute an average hesitation score difference." + " " + expected_part
        else:
            if final_score_diff > 0:
                score_part = f"On the hesitation score side, the trainee came in about {abs(final_score_diff):.3f} higher than the expert." + " " + expected_part
            elif final_score_diff < 0:
                score_part = f"On the hesitation score side, the trainee came in about {abs(final_score_diff):.3f} lower than the expert." + " " + expected_part
            else:
                score_part = "On the hesitation score side, the trainee matched the expert." + " " + expected_part

        # Comma-separated details kept for return payload (easy to parse)
        header = "Entry, Expert ID, Trainee ID, Expected gap (thr), Time Δ (T−E) from prev entry, Score Δ (T−E), Trainee vs Expert"
        detail_lines = [header] + [f"{r[0]}, {r[1]}, {r[2]}, {r[3]}, {r[4]}, {r[5]}, {r[6]}" for r in rows]
        details_csv = "\n".join(detail_lines)

        # Pretty table for saved TXT (broken-line separators between columns)
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

        pretty_headers = [h.strip() for h in header.split(",")]
        pretty_rows = [[str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5]), str(r[6])] for r in rows]
        details_pretty = _broken_table(pretty_headers, pretty_rows)

        EntranceHesitation_Metric.__generateExpertCompareGraphic(
            session_folder, frame_starts_expert, frame_starts_trainee
        )

        text = score_part + " " + time_part + "\n" + details_csv

        # Save a readable table to TXT (but keep comma-separated text in the returned payload)
        saved_text = score_part + " " + time_part + "\n\n" + details_pretty + "\n"
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(saved_text)
        except Exception:
            # Non-fatal: still return the summary even if file write fails
            pass

        return {
            "Name": "ENTRANCE_HESITATION",
            "Type": "Single",
            "ImgLocation": img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }

    @staticmethod
    def __generateExpertCompareGraphic(output_folder, frame_start_expert, frame_start_trainee):
        # Handle empty inputs.
        frame_start_expert = list(frame_start_expert) if frame_start_expert is not None else []
        frame_start_trainee = list(frame_start_trainee) if frame_start_trainee is not None else []
        if not frame_start_expert and not frame_start_trainee:
            return

        # Plot entry times as seconds since first entry (30 FPS).
        def _to_relative_seconds(frames: List[float]) -> List[float]:
            if not frames:
                return []
            base = float(min(frames))
            return [(float(f) - base) / 30.0 for f in frames]

        x_expert = _to_relative_seconds(frame_start_expert)
        x_trainee = _to_relative_seconds(frame_start_trainee)

        max_val = max(
            max(x_expert) if x_expert else 0.0,
            max(x_trainee) if x_trainee else 0.0,
        )

        fig, ax = plt.subplots(figsize=(15, 4), constrained_layout=True)

        x_max = max_val + int(0.25 * max_val) if max_val > 0 else 1
        ax.set_xlim(0, x_max)

        # Greedy lane assignment so labels on the same lane do not overlap horizontally.
        def _assign_lanes(xs: List[float], min_dx: float) -> Tuple[List[int], int]:
            if not xs:
                return [], 0
            order = np.argsort(xs)
            lanes = [0] * len(xs)
            lane_last: List[float] = []
            for idx in order:
                x = float(xs[int(idx)])
                placed = False
                for li, last in enumerate(lane_last):
                    if x - last >= min_dx:
                        lanes[int(idx)] = li
                        lane_last[li] = x
                        placed = True
                        break
                if not placed:
                    lane_last.append(x)
                    lanes[int(idx)] = len(lane_last) - 1
            return lanes, len(lane_last)

        def _font_size(n: int) -> int:
            # Adapt font size to entrant count
            if n <= 6:
                return 12
            if n <= 10:
                return 10
            return 8

        # Compute laneing parameters in data units
        x_range = float(max(1, x_max))
        lane_step = 0.42
        base_expert_y = 0.85
        base_trainee_y = -0.85

        # Minimum horizontal spacing scales with average spacing (now in seconds)
        min_dx_expert = max(0.25, 0.75 * (x_range / max(1, len(x_expert))))
        min_dx_trainee = max(0.25, 0.75 * (x_range / max(1, len(x_trainee))))

        expert_lanes, expert_lane_count = _assign_lanes(x_expert, min_dx_expert)
        trainee_lanes, trainee_lane_count = _assign_lanes(x_trainee, min_dx_trainee)

        # Dynamic ylim based on lane counts
        top = base_expert_y + lane_step * max(0, expert_lane_count - 1) + 0.55
        bottom = base_trainee_y - lane_step * max(0, trainee_lane_count - 1) - 0.55
        ax.set_ylim(bottom, top)

        # Baseline (full width, neutral color so it doesn't compete with point colors)
        ax.hlines(y=0, xmin=0, xmax=x_max, colors="gray", linewidth=2.0, zorder=1)

        # Points
        ax.scatter(x_expert, np.zeros(len(x_expert)), s=120, c="palevioletred", zorder=3)
        ax.scatter(x_expert, np.zeros(len(x_expert)), s=30, c="darkmagenta", zorder=4)
        ax.scatter(x_trainee, np.zeros(len(x_trainee)), s=120, c="lightskyblue", zorder=3)
        ax.scatter(x_trainee, np.zeros(len(x_trainee)), s=30, c="blue", zorder=4)

        # Legend (top-left): map colors to Expert/Trainee
        from matplotlib.lines import Line2D

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                label="Expert",
                markerfacecolor="palevioletred",
                markeredgecolor="darkmagenta",
                markersize=10,
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                label="Trainee",
                markerfacecolor="lightskyblue",
                markeredgecolor="blue",
                markersize=10,
            ),
        ]

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.99, 0.99),
            frameon=True,
            framealpha=0.85,
            facecolor="white",
            edgecolor="none",
        )

        fs_e = _font_size(len(x_expert))
        fs_t = _font_size(len(x_trainee))

        # Labels: shorter + boxed + leader lines to reduce clutter.
        for i, x in enumerate(x_expert):
            lane = expert_lanes[i] if i < len(expert_lanes) else 0
            y = base_expert_y + lane_step * lane
            label = f"E#{i + 1}"
            ax.annotate(
                label,
                xy=(x, 0),
                xytext=(x, y),
                textcoords="data",
                ha="center",
                va="center",
                fontfamily="serif",
                fontweight="bold",
                color="darkmagenta",
                fontsize=fs_e,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
                arrowprops=dict(arrowstyle="-", color="darkmagenta", lw=1.2),
                zorder=5,
            )

        for i, x in enumerate(x_trainee):
            lane = trainee_lanes[i] if i < len(trainee_lanes) else 0
            y = base_trainee_y - lane_step * lane
            label = f"T#{i + 1}"
            ax.annotate(
                label,
                xy=(x, 0),
                xytext=(x, y),
                textcoords="data",
                ha="center",
                va="center",
                fontfamily="serif",
                fontweight="bold",
                color="blue",
                fontsize=fs_t,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.75),
                arrowprops=dict(arrowstyle="-", color="blue", lw=1.2),
                zorder=5,
            )

        # Hide chart chrome
        for spine in ["left", "top", "right", "bottom"]:
            ax.spines[spine].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.savefig(os.path.join(output_folder, "ENTRANCE_HESITATION_Comparison.jpg"))
        plt.close(fig)
