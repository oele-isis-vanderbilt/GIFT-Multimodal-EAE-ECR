from functools import cmp_to_key
import pandas as pd
import numpy as np
import cv2
import os
import glob
import math
from .metric import AbstractMetric
from .utils import len_comparator, arg_first_comparator
from typing import Optional, Union, List, Dict, Tuple, Any


class EntranceVectors_Metric(AbstractMetric):
    def __init__(self, config) -> None:
        super().__init__(config)
        self.metricName = "ENTRANCE_VECTORS"
        self.num_tracks = len(config["POD"])
        self.map=config.get("Map Image", None)
        self.tracks = {}

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
        # Analyze the first k valid points per track to infer entrance side (+1 / -1)
        k_frames = 60

        # Get the n longest tracks and sort them by their start frames
        tracks = list(self.tracks.values())
        tracks.sort(key=cmp_to_key(len_comparator), reverse=True)
        tracks = tracks[:self.num_tracks]
        tracks.sort(key=cmp_to_key(arg_first_comparator))

        if len(tracks) < 2:
            return -1

        entrance_vectors = []
        for trk in tracks:
            # Find the first non-None frame
            first_non_none = next((i for i, v in enumerate(trk) if v is not None), None)
            if first_non_none is None:
                continue

            # Collect the first k non-None frames
            non_none_points = [v for v in trk[first_non_none:] if v is not None]
            if len(non_none_points) < k_frames:
                continue
            selected_points = non_none_points[:k_frames]

            # Determine the side (+1 / -1) of this entrance relative to room centroid
            entry_pt = np.array(selected_points[0])
            movement_vec = np.array(selected_points[-1]) - entry_pt

            centroid = np.mean(np.array(self.config["POD"]), axis=0)
            v_centre = centroid - entry_pt

            # 2‑D cross‑product → scalar; sign tells which side of the divider
            z_cross = np.cross(v_centre, movement_vec)
            direction_sign = 1 if z_cross >= 0 else -1
            entrance_vectors.append(direction_sign)

        # Calculate final scores based on percentage of correct alternations
        score = 0
        for i in range(1, len(entrance_vectors)):
            if np.sign(entrance_vectors[i]) != np.sign(entrance_vectors[i - 1]):
                score += 1

        return round(score / max(1, len(entrance_vectors) - 1), 2)
    
    
    @staticmethod
    def _first_valid_index(trk: List[Any]) -> Optional[int]:
        for i, v in enumerate(trk):
            if v is not None:
                return i
        return None

    @staticmethod
    def _centroid_from_pod(pod) -> np.ndarray:
        return np.mean(np.array(pod, dtype=float), axis=0)

    @staticmethod
    def _entrance_sign_cross(
        trk: List[Any],
        centroid: np.ndarray,
        *,
        k_frames: int = 60,
        min_move_norm: float = 1e-6,
    ) -> Dict[str, Any]:
        """Return entrance direction metadata using the metric's cross-product sign.

        Uses the first `k_frames` non-None points:
          entry_pt = first point
          end_pt   = k_frames-th point
          sign = sign(cross(centroid-entry_pt, end_pt-entry_pt))
        """
        first = EntranceVectors_Metric._first_valid_index(trk)
        if first is None:
            return {
                "start_frame": None,
                "start_xy": None,
                "end_xy": None,
                "dx": None,
                "dy": None,
                "z_cross": None,
                "sign": 0,
                "side": "UNKNOWN",
            }

        non_none_points = [v for v in trk[first:] if v is not None]
        if len(non_none_points) < k_frames:
            return {
                "start_frame": int(first) + 1,
                "start_xy": non_none_points[0] if len(non_none_points) > 0 else None,
                "end_xy": None,
                "dx": None,
                "dy": None,
                "z_cross": None,
                "sign": 0,
                "side": "UNKNOWN",
            }

        selected_points = non_none_points[:k_frames]
        entry_pt = np.array(selected_points[0], dtype=float)
        end_pt = np.array(selected_points[-1], dtype=float)
        movement_vec = end_pt - entry_pt

        v_centre = np.array(centroid, dtype=float) - entry_pt

        mv_norm = float(np.linalg.norm(movement_vec))
        vc_norm = float(np.linalg.norm(v_centre))
        if mv_norm < min_move_norm or vc_norm < min_move_norm:
            sign = 0
            side = "UNKNOWN"
            z_cross = 0.0
        else:
            z_cross = float(np.cross(v_centre, movement_vec))
            sign = 1 if z_cross >= 0 else -1
            side = "POS" if sign == 1 else "NEG"

        return {
            "start_frame": int(first) + 1,
            "start_xy": (float(entry_pt[0]), float(entry_pt[1])),
            "end_xy": (float(end_pt[0]), float(end_pt[1])),
            "dx": float(movement_vec[0]),
            "dy": float(movement_vec[1]),
            "z_cross": float(z_cross),
            "sign": int(sign),
            "side": side,
        }

    def expertCompare(self, session_folder: str, expert_folder: str, map_image=None, pod=None):
        """Compare trainee vs expert entrance-side (sign) using the same logic as scoring.

        Returns a short summary plus a per-entry list: Entry, Trainee ID, Expert ID, Match.
        Also writes two map images with entrance arrows.
        """
        enemy_ids = set(self.config.get("enemy_ids", [99]))

        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_position_cache(folder: str) -> List[Dict[str, Any]]:
            cache_path = _pick_latest(folder, "*_PositionCache.txt")
            if cache_path is None:
                raise FileNotFoundError(f"No PositionCache found in {folder}")

            df = pd.read_csv(cache_path)
            cols = {c.lower(): c for c in df.columns}
            frame_col = cols.get("frame")
            id_col = cols.get("id")
            x_col = cols.get("mapx")
            y_col = cols.get("mapy")
            if frame_col is None or id_col is None or x_col is None or y_col is None:
                raise ValueError(f"Unexpected PositionCache format: {cache_path}")

            df = df[[frame_col, id_col, x_col, y_col]].dropna()
            df[frame_col] = df[frame_col].astype(int)
            df[id_col] = df[id_col].astype(int)
            df[x_col] = df[x_col].astype(float)
            df[y_col] = df[y_col].astype(float)

            max_frame = int(df[frame_col].max()) if len(df) else 0
            tracks = {}
            for tid, g in df.groupby(id_col):
                # Mirror metric scoring: exclude configured enemy IDs
                if int(tid) in enemy_ids:
                    continue
                traj = [None] * max_frame
                for _, row in g.iterrows():
                    fidx = int(row[frame_col])
                    if 1 <= fidx <= max_frame:
                        traj[fidx - 1] = (float(row[x_col]), float(row[y_col]))
                tracks[int(tid)] = traj

            # Return list of dicts so we preserve the original track/person ID
            return [{"id": int(tid), "traj": traj} for tid, traj in tracks.items()]

        def _entrance_info(track: Dict[str, Any], centroid: np.ndarray, *, k_frames: int) -> Dict[str, Any]:
            """Compute per-track entrance direction and side using getFinalScore logic.

            `track` is a dict: {"id": <person_id>, "traj": <list of (x,y)/None>}.
            """
            info = EntranceVectors_Metric._entrance_sign_cross(track["traj"], centroid, k_frames=k_frames)
            info["track_id"] = int(track["id"])
            return info


        # Load map image (prefer provided map_image; fallback to metric's configured map)
        if map_image is None:
            map_image = self.map

            # If config provided a file path instead of an image array, load it
            if isinstance(map_image, str):
                map_image = cv2.imread(map_image) if os.path.exists(map_image) else None

        # Setup output paths for reuse
        os.makedirs(session_folder, exist_ok=True)
        expert_img_path = os.path.join(session_folder, "ENTRANCE_VECTORS_Expert.jpg")
        trainee_img_path = os.path.join(session_folder, "ENTRANCE_VECTORS_Trainee.jpg")
        txt_path = os.path.join(session_folder, "ENTRANCE_VECTORS_Comparison.txt")

        if map_image is None:
            err_text = "There was an error while processing this comparison. Missing map image (config['Map Image'] / self.map)."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "ENTRANCE_VECTORS",
                "Type": "SideBySide",
                "ExpertImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Centroid for cross-product logic: prefer provided POD, else config POD, else map center.
        pod_poly = pod if pod is not None else self.config.get("POD")
        if pod_poly is not None:
            centroid = EntranceVectors_Metric._centroid_from_pod(pod_poly)
        else:
            h, w = map_image.shape[:2]
            centroid = np.array([w / 2.0, h / 2.0], dtype=float)

        k_frames = 60

        try:
            expert_tracks = _load_position_cache(expert_folder)
            trainee_tracks = _load_position_cache(session_folder)
        except Exception:
            err_text = "There was an error while processing this comparison. Missing or invalid PositionCache."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "ENTRANCE_VECTORS",
                "Type": "SideBySide",
                "ExpertImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        def _start_idx(track: Dict[str, Any]) -> int:
            idx = EntranceVectors_Metric._first_valid_index(track["traj"])
            return int(idx) if idx is not None else 10**12

        def _valid_len(track: Dict[str, Any]) -> int:
            return sum(1 for v in track["traj"] if v is not None)

        def _select_for_scoring(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            # Mirror getFinalScore: pick the N longest, then sort by entry (first valid frame).
            if self.num_tracks and self.num_tracks > 0:
                tracks = sorted(tracks, key=_valid_len, reverse=True)[: self.num_tracks]
            return sorted(tracks, key=_start_idx)

        expert_tracks = _select_for_scoring(expert_tracks)
        trainee_tracks = _select_for_scoring(trainee_tracks)

        if len(expert_tracks) == 0 or len(trainee_tracks) == 0:
            err_text = "There was an error while processing this comparison. No valid tracks found."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "ENTRANCE_VECTORS",
                "Type": "SideBySide",
                "ExpertImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Compute per-entrant direction info
        expert_infos = [_entrance_info(t, centroid, k_frames=k_frames) for t in expert_tracks]
        trainee_infos = [_entrance_info(t, centroid, k_frames=k_frames) for t in trainee_tracks]

        def _alternation_score_and_by_entry(signs: List[int]) -> Tuple[float, List[str]]:
            """Mirror getFinalScore alternation: drop sign==0, then score adjacent alternations.

            Returns:
              - score in [0,1]
              - per-entry contribution labels aligned to the original entry list:
                  first valid entrant => 'N/A'
                  later valid entrant => '1' if alternated vs previous valid, else '0'
                  unknown entrant => 'N/A'
            """
            idxs = [i for i, s in enumerate(signs) if s != 0]
            by_entry = ["N/A"] * len(signs)
            if len(idxs) >= 1:
                by_entry[idxs[0]] = "N/A"
            alt = 0
            for j in range(1, len(idxs)):
                prev_i = idxs[j - 1]
                cur_i = idxs[j]
                is_alt = (np.sign(signs[cur_i]) != np.sign(signs[prev_i]))
                if is_alt:
                    alt += 1
                    by_entry[cur_i] = "1"
                else:
                    by_entry[cur_i] = "0"
            denom = max(1, len(idxs) - 1)
            return float(alt) / float(denom), by_entry

        expert_signs = [int(info.get("sign", 0)) for info in expert_infos]
        trainee_signs = [int(info.get("sign", 0)) for info in trainee_infos]
        expert_alt_score, expert_alt_by_entry = _alternation_score_and_by_entry(expert_signs)
        trainee_alt_score, trainee_alt_by_entry = _alternation_score_and_by_entry(trainee_signs)

        # Generate arrow-only comparison graphics into the session folder
        EntranceVectors_Metric.__generateExpertCompareGraphic(
            output_folder=session_folder,
            expert_infos=expert_infos,
            trainee_infos=trainee_infos,
            map_view=map_image
        )

        # Compare entry-by-entry (by entry order); allow mismatched counts.
        max_n = max(len(expert_infos), len(trainee_infos))

        valid = 0
        match = 0
        per_entry_lines = []
        for i in range(max_n):
            einfo = expert_infos[i] if i < len(expert_infos) else None
            tinfo = trainee_infos[i] if i < len(trainee_infos) else None

            expert_id = einfo.get("track_id") if einfo is not None else None
            trainee_id = tinfo.get("track_id") if tinfo is not None else None

            e_sign = einfo.get("sign") if einfo is not None else 0
            t_sign = tinfo.get("sign") if tinfo is not None else 0

            if einfo is not None and tinfo is not None and e_sign != 0 and t_sign != 0:
                is_match = (e_sign == t_sign)
                valid += 1
                match += 1 if is_match else 0
                match_str = "YES" if is_match else "NO"
            else:
                match_str = "N/A"

            t_alt = trainee_alt_by_entry[i] if i < len(trainee_alt_by_entry) else "N/A"
            e_alt = expert_alt_by_entry[i] if i < len(expert_alt_by_entry) else "N/A"

            def _alt_num(v: str) -> Optional[int]:
                return int(v) if v in ("0", "1") else None

            t_num = _alt_num(t_alt)
            e_num = _alt_num(e_alt)

            if t_num is not None and e_num is not None:
                pair_diff = t_num - e_num  # Δ = trainee - expert
                pair_diff_str = f"{pair_diff:+d}"
                if pair_diff > 0:
                    pair_cmp = "BETTER"
                elif pair_diff < 0:
                    pair_cmp = "WORSE"
                else:
                    pair_cmp = "SIMILAR"
            else:
                pair_diff_str = "N/A"
                pair_cmp = "N/A"

            per_entry_lines.append(
                f"{i+1}, {trainee_id if trainee_id is not None else 'N/A'}, {expert_id if expert_id is not None else 'N/A'}, {match_str}, {pair_diff_str}, {pair_cmp}"
            )

        alt_summary = (
            f"Alternation score: trainee {trainee_alt_score * 100:.1f}% vs expert {expert_alt_score * 100:.1f}%."
        )

        if valid > 0:
            sign_match_pct = (match / valid) * 100.0
            summary_line = (
                f"The trainee matches the expert about {sign_match_pct:.1f}% of the time on entrance side. "
                + alt_summary
            )
        else:
            sign_match_pct = None
            summary_line = "Couldn't compute a match percent (not enough paired entries with a clear entrance side). " + alt_summary

        # --- Build a clean, aligned table for the saved TXT (keep CSV-ish text for return payload) ---
        rows = []
        for i, line in enumerate(per_entry_lines, start=1):
            # per_entry_lines are: "Entry, Trainee ID, Expert ID, Match, PairScoreΔ (T−E), Trainee vs Expert"
            parts = [p.strip() for p in line.split(",")]
            # Defensive: ensure we always have 6 columns
            while len(parts) < 6:
                parts.append("N/A")
            entry, trainee_id, expert_id, match_str, pair_diff_str, pair_cmp = parts[:6]
            rows.append(
                {
                    "Entry": entry,
                    "Trainee ID": trainee_id,
                    "Expert ID": expert_id,
                    "Match": match_str,
                    "PairScoreΔ (T−E)": pair_diff_str,
                    "Trainee vs Expert": pair_cmp,
                }
            )

        df_table = pd.DataFrame(rows)

        # Text returned to callers can remain comma-separated (easy to parse)
        details_header = "Entry, Trainee ID, Expert ID, Match, PairScoreΔ (T−E), Trainee vs Expert\n"
        details_csv = details_header + "\n".join(per_entry_lines)
        text = summary_line + "\n\n" + details_csv

        # Text saved to TXT is formatted as a readable table (with dotted separators between columns)
        def _dotted_table(df: pd.DataFrame) -> str:
            if df is None or df.empty:
                return "(no rows)"

            cols = [str(c) for c in df.columns]
            # Stringify cell values (keep as-is for readability)
            data_rows = [["" if v is None else str(v) for v in row] for row in df.to_numpy().tolist()]

            # Compute column widths
            widths = []
            for j, c in enumerate(cols):
                max_cell = max([len(c)] + [len(r[j]) for r in data_rows])
                widths.append(max_cell)

            sep = " | "  # broken vertical separator between columns

            def _fmt_row(values: List[str]) -> str:
                return sep.join(v.ljust(widths[i]) for i, v in enumerate(values))

            # Broken horizontal line (matches the vertical separator style)
            line_parts = ["-" * w for w in widths]
            broken_line = sep.join(line_parts)

            out = []
            out.append(_fmt_row(cols))
            out.append(broken_line)
            for r in data_rows:
                out.append(_fmt_row(r))
            return "\n".join(out)

        table_pretty = _dotted_table(df_table)
        saved_text = summary_line + "\n\n" + table_pretty + "\n"

        # Also save the returned summary text alongside the images
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(saved_text)
        except Exception:
            # Non-fatal: still return the summary even if file write fails
            pass

        return {
            "Name": "ENTRANCE_VECTORS",
            "Type": "SideBySide",
            "ExpertImageLocation": expert_img_path,
            "TraineeImageLocation": trainee_img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }
    
    @staticmethod
    def __generateExpertCompareGraphic(output_folder, expert_infos, trainee_infos, map_view):
        """Write two arrow maps (expert + trainee) for entrance direction.

        Output files in `output_folder`:
          • ENTRANCE_VECTORS_Expert.jpg
          • ENTRANCE_VECTORS_Trainee.jpg

        Each entrant is drawn as a single direction arrow (start -> fixed length), plus a legend.
        """
        expert_view = map_view.copy()
        trainee_view = map_view.copy()

        # Match helper_functions.py palette (BGR tuples)
        predefined_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128)
        ]

        def _put_text(img, text, org, font, scale, color, thickness, *, shadow=True):
            """Draw readable text with a subtle outline for clarity."""
            x, y = int(org[0]), int(org[1])
            if shadow:
                # outline / shadow (dark)
                cv2.putText(img, text, (x + 1, y + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
            else:
                cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

        def _draw_legend(img, infos, title="Legend"):
            """Extend image to the right and draw a clean legend in the new margin."""
            h0, w0 = img.shape[:2]

            font = cv2.FONT_HERSHEY_SIMPLEX
            # Responsive sizing based on image diagonal
            diag = math.hypot(w0, h0)
            font_scale = max(0.55, min(0.9, diag / 1400.0))
            thickness = max(1, int(round(diag / 900.0)))

            line_h = int(round(26 * font_scale))
            swatch = int(round(16 * font_scale))
            gap = int(round(10 * font_scale))
            pad = int(round(12 * font_scale))

            lines = [title] + [f"Entrant #{i}" for i in range(1, len(infos) + 1)]

            # Measure text widths
            max_text_w = 0
            for t in lines:
                (tw, th), _b = cv2.getTextSize(t, font, font_scale, thickness)
                max_text_w = max(max_text_w, tw)

            panel_w = pad * 2 + swatch + gap + max_text_w
            panel_h = pad * 2 + line_h * len(lines)

            # Extend canvas (right margin for legend; bottom margin if needed)
            extra_right = panel_w + pad * 2
            extra_bottom = max(0, panel_h + pad * 2 - h0)

            bg = tuple(int(x) for x in img[0, 0].tolist())
            canvas = np.full((h0 + extra_bottom, w0 + extra_right, 3), bg, dtype=np.uint8)
            canvas[:h0, :w0] = img
            img = canvas

            # Panel anchor in new right margin
            x0 = w0 + pad
            y0 = pad

            # Alpha-blended panel for a modern look
            overlay = img.copy()
            panel_bg = (28, 28, 28)
            border = (220, 220, 220)
            cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), panel_bg, -1, cv2.LINE_AA)
            alpha_panel = 0.82
            cv2.addWeighted(overlay, alpha_panel, img, 1.0 - alpha_panel, 0, dst=img)
            cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), border, max(1, thickness), cv2.LINE_AA)

            # Title
            title_y = y0 + pad + int(round(18 * font_scale))
            _put_text(img, title, (x0 + pad, title_y), font, font_scale, (255, 255, 255), thickness)

            # Entries
            y = y0 + pad + line_h
            for entry_num, info in enumerate(infos, start=1):
                tid = info.get("track_id")
                tid = int(tid) if tid is not None else entry_num
                color = predefined_colors[tid % len(predefined_colors)]

                sx1 = x0 + pad
                sy1 = y + int(round(4 * font_scale))
                sx2 = sx1 + swatch
                sy2 = sy1 + swatch

                cv2.rectangle(img, (sx1, sy1), (sx2, sy2), color, -1, cv2.LINE_AA)
                cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (255, 255, 255), max(1, thickness), cv2.LINE_AA)

                label = f"Entrant #{entry_num}"
                _put_text(img, label, (sx2 + gap, y + int(round(18 * font_scale))), font, font_scale, (255, 255, 255), thickness)

                y += line_h

            return img

        def _draw_arrow(img, start_xy, end_xy, color, idx, total):
            if start_xy is None or end_xy is None:
                return

            sx, sy = float(start_xy[0]), float(start_xy[1])
            ex0, ey0 = float(end_xy[0]), float(end_xy[1])

            v = np.array([ex0 - sx, ey0 - sy], dtype=float)
            n = float(np.linalg.norm(v))
            if n == 0:
                return
            u = v / n

            h, w = img.shape[:2]
            diag = math.hypot(w, h)
            arrow_len = max(70.0, 0.14 * diag)
            ex = sx + u[0] * arrow_len
            ey = sy + u[1] * arrow_len

            # Perpendicular offset to reduce clutter around the entrance
            perp = np.array([-u[1], u[0]], dtype=float)
            spread = (idx - (total + 1) / 2.0)
            offset = perp * (14.0 * spread)

            p1 = (int(round(sx + offset[0])), int(round(sy + offset[1])))
            p2 = (int(round(ex + offset[0])), int(round(ey + offset[1])))

            # Thickness scales with image size
            thick = max(2, int(round(diag / 520.0)))

            # Draw a darker outline first, then the colored arrow on top (clean + readable)
            outline_color = (0, 0, 0)
            cv2.arrowedLine(img, p1, p2, outline_color, thick + 3, cv2.LINE_AA, tipLength=0.28)
            cv2.arrowedLine(img, p1, p2, color, thick, cv2.LINE_AA, tipLength=0.28)

            # Start dot for clarity
            cv2.circle(img, p1, max(4, thick + 1), outline_color, -1, cv2.LINE_AA)
            cv2.circle(img, p1, max(3, thick), color, -1, cv2.LINE_AA)

        # Expert arrows (color by track/person ID, legend label by entry number)
        total_e = len(expert_infos)
        for entry_num, info in enumerate(expert_infos, start=1):
            tid = info.get("track_id")
            tid = int(tid) if tid is not None else entry_num
            color = predefined_colors[tid % len(predefined_colors)]
            _draw_arrow(
                expert_view,
                info.get("start_xy"),
                info.get("end_xy"),
                color,
                idx=entry_num,
                total=total_e
            )

        # Trainee arrows (color by track/person ID, legend label by entry number)
        total_t = len(trainee_infos)
        for entry_num, info in enumerate(trainee_infos, start=1):
            tid = info.get("track_id")
            tid = int(tid) if tid is not None else entry_num
            color = predefined_colors[tid % len(predefined_colors)]
            _draw_arrow(
                trainee_view,
                info.get("start_xy"),
                info.get("end_xy"),
                color,
                idx=entry_num,
                total=total_t
            )

        # Add legends in the right margin (canvas extended)
        expert_view = _draw_legend(expert_view, expert_infos, title="Expert")
        trainee_view = _draw_legend(trainee_view, trainee_infos, title="Trainee")

        cv2.imwrite(os.path.join(output_folder, "ENTRANCE_VECTORS_Trainee.jpg"), trainee_view)
        cv2.imwrite(os.path.join(output_folder, "ENTRANCE_VECTORS_Expert.jpg"), expert_view)