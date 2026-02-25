import os
import glob
import math
import cv2
import numpy as np
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd

from .metric import AbstractMetric


class IdentifyAndCapturePods_Metric(AbstractMetric):
    def __init__(self, config):
        super().__init__(config)
        self.metricName = "IDENTIFY_AND_CAPTURE_POD"
        self.score = 0.0
        self.pod = config.get("POD", None)
        self.map=config.get("Map Image", None)
        
    def process(self, context):
        # Retrieve pod capture info from context
        pod_capture = context.pod_capture or {}
        total_pods = len(pod_capture)
        # Count correctly occupied pods
        occupied = sum(
            1
            for _, info in pod_capture.items()
            if info.get("assigned_id") is not None and info.get("capture_frame") is not None
        )
        # Compute ratio (0.0 if no pods)
        self.score = occupied / total_pods if total_pods > 0 else 1.0

    def getFinalScore(self):
        # Return ratio rounded to two decimals
        return round(float(self.score), 2)

    # ------------------------------------------------------------------
    # Expert compare
    # ------------------------------------------------------------------
    def expertCompare(
        self,
        session_folder: str,
        expert_folder: str,
        map_image=None,
        pod=None,
        _config: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Compare trainee vs expert POD assignment/capture.

        Reads the latest `*_PodCache.txt` (and `*_PositionCache.txt` for entry order) from each folder,
        writes `IDENTIFY_AND_CAPTURE_POD_Expert.jpg`/`IDENTIFY_AND_CAPTURE_POD_Trainee.jpg`, and returns a text summary + per-POD lines.
        """

        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _load_pod_cache(folder: str) -> pd.DataFrame:
            cache_path = _pick_latest(folder, "*_PodCache.txt")
            if cache_path is None:
                raise FileNotFoundError(f"No PodCache found in {folder}")

            df = pd.read_csv(cache_path)
            cols = {c.strip().lower(): c for c in df.columns}
            need = ["pod_idx", "assigned_id", "capture_time_sec", "capture_frame"]
            missing = [c for c in need if c not in cols]
            if missing:
                raise ValueError(f"Unexpected PodCache format: {cache_path} (missing {missing})")

            df = df[[cols["pod_idx"], cols["assigned_id"], cols["capture_time_sec"], cols["capture_frame"]]].copy()
            df.columns = ["pod_idx", "assigned_id", "capture_time_sec", "capture_frame"]

            df["pod_idx"] = pd.to_numeric(df["pod_idx"], errors="coerce").astype("Int64")
            df["assigned_id"] = pd.to_numeric(df["assigned_id"], errors="coerce").astype("Int64")
            df["capture_time_sec"] = pd.to_numeric(df["capture_time_sec"], errors="coerce")
            df["capture_frame"] = pd.to_numeric(df["capture_frame"], errors="coerce").astype("Int64")

            df = df.dropna(subset=["pod_idx"])
            df["pod_idx"] = df["pod_idx"].astype(int)

            # Captured if assigned_id and capture_frame are present
            df["captured"] = (~df["assigned_id"].isna()) & (~df["capture_frame"].isna())
            df["assigned"] = ~df["assigned_id"].isna()

            return df.sort_values("pod_idx").reset_index(drop=True)

        def _load_entry_map(folder: str) -> Dict[int, int]:
            """Return {track_id: entry_number} from first appearance in PositionCache."""
            pos_path = _pick_latest(folder, "*_PositionCache.txt")
            if pos_path is None:
                return {}

            dfp = pd.read_csv(pos_path)
            cols = {c.strip().lower(): c for c in dfp.columns}
            if not {"frame", "id"}.issubset(set(cols.keys())):
                return {}

            frame_col = cols["frame"]
            id_col = cols["id"]

            dfp = dfp[[frame_col, id_col]].dropna()
            dfp[frame_col] = pd.to_numeric(dfp[frame_col], errors="coerce")
            dfp[id_col] = pd.to_numeric(dfp[id_col], errors="coerce")
            dfp = dfp.dropna(subset=[frame_col, id_col])
            dfp[frame_col] = dfp[frame_col].astype(int)
            dfp[id_col] = dfp[id_col].astype(int)

            enemy_ids = set(self.config.get("enemy_ids", [99]))
            dfp = dfp[~dfp[id_col].isin([int(x) for x in enemy_ids])]
            if dfp.empty:
                return {}

            first_frame_by_id = dfp.groupby(id_col)[frame_col].min().sort_values()
            entry_map: Dict[int, int] = {}
            for i, tid in enumerate(first_frame_by_id.index.tolist(), start=1):
                entry_map[int(tid)] = int(i)
            return entry_map

        def _summary(df: pd.DataFrame) -> Tuple[int, int, float]:
            total = int(len(df))
            captured = int(df["captured"].sum()) if total > 0 else 0
            score = (captured / total) if total > 0 else 1.0
            return total, captured, float(score)

        def _coerce_map_image(img_or_path):
            if img_or_path is None:
                return None
            if isinstance(img_or_path, str):
                return cv2.imread(img_or_path) if os.path.exists(img_or_path) else None
            return img_or_path

        # Resolve map + POD from args or self
        if map_image is None:
            map_image = getattr(self, "map", None)
        map_image = _coerce_map_image(map_image)

        if pod is None:
            pod = getattr(self, "pod", None)

        # Palette matches helper_functions / EntranceVectors (BGR)
        predefined_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128)
        ]

        expert_img_path = os.path.join(session_folder, "IDENTIFY_AND_CAPTURE_POD_Expert.jpg")
        trainee_img_path = os.path.join(session_folder, "IDENTIFY_AND_CAPTURE_POD_Trainee.jpg")
        txt_path = os.path.join(session_folder, "IDENTIFY_AND_CAPTURE_POD_Comparison.txt")

        if map_image is None or pod is None:
            os.makedirs(session_folder, exist_ok=True)
            err_text = "There was an error while processing this comparison. Missing map image or POD coordinates."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "IDENTIFY_AND_CAPTURE_POD",
                "Type": "SideBySide",
                "ExpertImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Normalize POD coordinates to a list of (x, y)
        try:
            pods_list = pod.tolist() if isinstance(pod, np.ndarray) else list(pod)
        except Exception:
            pods_list = []
        # Ensure tuple-like
        pods_xy = []
        for p in pods_list:
            try:
                pods_xy.append((float(p[0]), float(p[1])))
            except Exception:
                continue

        # Load caches
        try:
            expert_df = _load_pod_cache(expert_folder)
            trainee_df = _load_pod_cache(session_folder)
            expert_entry_map = _load_entry_map(expert_folder)
            trainee_entry_map = _load_entry_map(session_folder)
        except Exception:
            os.makedirs(session_folder, exist_ok=True)
            err_text = "There was an error while processing this comparison. Missing or invalid PodCache in one or both folders."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "IDENTIFY_AND_CAPTURE_POD",
                "Type": "Text",
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Align pods by pod_idx (outer join so we can show mismatches)
        merged = pd.merge(
            expert_df,
            trainee_df,
            on="pod_idx",
            how="outer",
            suffixes=("_expert", "_trainee"),
        ).sort_values("pod_idx").reset_index(drop=True)

        # Summary scores per side
        e_total, e_captured, e_score = _summary(expert_df)
        t_total, t_captured, t_score = _summary(trainee_df)

        # Entry number per side
        expert_entry_num = (
            merged.get("assigned_id_expert").apply(
                lambda x: (expert_entry_map.get(int(x)) if pd.notna(x) else pd.NA)
            )
            if merged.get("assigned_id_expert") is not None
            else pd.Series([pd.NA] * len(merged))
        )
        trainee_entry_num = (
            merged.get("assigned_id_trainee").apply(
                lambda x: (trainee_entry_map.get(int(x)) if pd.notna(x) else pd.NA)
            )
            if merged.get("assigned_id_trainee") is not None
            else pd.Series([pd.NA] * len(merged))
        )

        entry_match = (
            expert_entry_num.notna()
            & trainee_entry_num.notna()
            & (expert_entry_num.astype("Int64") == trainee_entry_num.astype("Int64"))
        )
        entry_match_count = int(entry_match.sum())
        total_pods_compared = int(len(merged))

        expert_assigned_id = (
            merged.get("assigned_id_expert")
            if merged.get("assigned_id_expert") is not None
            else pd.Series([pd.NA] * len(merged))
        )
        trainee_assigned_id = (
            merged.get("assigned_id_trainee")
            if merged.get("assigned_id_trainee") is not None
            else pd.Series([pd.NA] * len(merged))
        )

        # Ensure output folder exists before drawing
        os.makedirs(session_folder, exist_ok=True)

        # Build per-pod status dicts for drawing
        def _status_map(df_side: pd.DataFrame, entry_map: Dict[int, int]) -> Dict[int, Dict[str, Any]]:
            out: Dict[int, Dict[str, Any]] = {}
            for _, r in df_side.iterrows():
                pid = int(r["pod_idx"])
                aid = r.get("assigned_id")
                captured = bool(r.get("captured", False))

                tid = None
                ent = None
                if pd.notna(aid):
                    try:
                        tid = int(aid)
                        ent = entry_map.get(tid)
                    except Exception:
                        tid = None
                        ent = None

                out[pid] = {"entry": ent, "id": tid, "captured": captured}
            return out

        expert_status = _status_map(expert_df, expert_entry_map)
        trainee_status = _status_map(trainee_df, trainee_entry_map)

        # Draw the two panels
        self.__generateExpertCompareGraphic(
            output_folder=session_folder,
            map_view=map_image,
            pods_xy=pods_xy,
            status=expert_status,
            out_name="IDENTIFY_AND_CAPTURE_POD_Expert.jpg",
            title="Expert",
            predefined_colors=predefined_colors,
        )
        self.__generateExpertCompareGraphic(
            output_folder=session_folder,
            map_view=map_image,
            pods_xy=pods_xy,
            status=trainee_status,
            out_name="IDENTIFY_AND_CAPTURE_POD_Trainee.jpg",
            title="Trainee",
            predefined_colors=predefined_colors,
        )

        # Summary + per-POD details
        e_pct = e_score * 100.0
        t_pct = t_score * 100.0
        diff_pp = t_pct - e_pct

        if abs(diff_pp) <= 1.0:
            perf = "about the same as"
        elif diff_pp > 1.0:
            perf = "better than"
        else:
            perf = "worse than"

        summary = (
            f"Entry assignment matched on {entry_match_count}/{total_pods_compared} PODs. "
            f"Overall, the trainee looks {perf} the expert on POD capture "
            f"(Trainee {t_pct:.1f}%, Expert {e_pct:.1f}%)."
        )

        lines = [
            "POD, Expert Entrant#, Expert ID, Trainee Entrant#, Trainee ID, Entrant# Match, Captured, Performance",
        ]

        for i, r in merged.iterrows():
            pod_idx = r.get("pod_idx")

            e_ent = expert_entry_num.iloc[i] if i < len(expert_entry_num) else pd.NA
            t_ent = trainee_entry_num.iloc[i] if i < len(trainee_entry_num) else pd.NA

            e_id = expert_assigned_id.iloc[i] if i < len(expert_assigned_id) else pd.NA
            t_id = trainee_assigned_id.iloc[i] if i < len(trainee_assigned_id) else pd.NA

            ent_match = bool(entry_match.iloc[i]) if i < len(entry_match) else False

            e_cap = bool(r.get("captured_expert")) if pd.notna(r.get("captured_expert")) else False
            t_cap = bool(r.get("captured_trainee")) if pd.notna(r.get("captured_trainee")) else False

            if e_cap == t_cap:
                pod_perf = "SIMILAR"
            elif t_cap and not e_cap:
                pod_perf = "BETTER"
            else:
                pod_perf = "WORSE"

            lines.append(
                f"P{int(pod_idx) if pd.notna(pod_idx) else 'N/A'}, "
                f"{int(e_ent) if pd.notna(e_ent) else 'N/A'}, {int(e_id) if pd.notna(e_id) else 'N/A'}, "
                f"{int(t_ent) if pd.notna(t_ent) else 'N/A'}, {int(t_id) if pd.notna(t_id) else 'N/A'}, "
                f"{'YES' if ent_match else 'NO'}, {'YES' if t_cap else 'NO'}, {pod_perf}"
            )

        # Returned payload remains comma-separated (easy to parse)
        details_csv = "\n".join(lines)
        text = summary + "\n" + details_csv

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

        # Convert the comma-separated lines into rows for pretty printing
        header = lines[0]
        pretty_headers = [h.strip() for h in header.split(",")]
        pretty_rows: List[List[str]] = []
        for ln in lines[1:]:
            parts = [p.strip() for p in ln.split(",")]
            while len(parts) < len(pretty_headers):
                parts.append("N/A")
            pretty_rows.append(parts[: len(pretty_headers)])

        details_pretty = _broken_table(pretty_headers, pretty_rows)
        saved_text = summary + "\n\n" + details_pretty + "\n"

        # Also save the returned summary text alongside the images
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(saved_text)
        except Exception:
            # Non-fatal: still return the summary even if file write fails
            pass

        return {
            "Name": "IDENTIFY_AND_CAPTURE_POD",
            "Type": "SideBySide",
            "ExpertImageLocation": expert_img_path,
            "TraineeImageLocation": trainee_img_path,
            "TxtLocation": txt_path,
            "Text": text,
        }

    @staticmethod
    def __generateExpertCompareGraphic(
        output_folder: str,
        map_view,
        pods_xy: list,
        status: Dict[int, Dict[str, Any]],
        out_name: str,
        title: str,
        predefined_colors: list,
    ) -> None:
        os.makedirs(output_folder, exist_ok=True)
        img = map_view.copy()
        h, w = img.shape[:2]
        w0 = w

        # Determine a reasonable POD marker radius from image diagonal
        diag = math.hypot(w, h)
        r = int(max(10, min(18, 0.018 * diag)))

        # Collect (entry_number, id) pairs for legend; label by entry_number but color by id
        entrant_items = sorted(
            {
                (v.get("entry"), v.get("id"))
                for v in status.values()
                if v.get("entry") is not None and v.get("id") is not None
            },
            key=lambda t: int(t[0]),
        )

        def _color_for_id(tid: int):
            return predefined_colors[int(tid) % len(predefined_colors)]

        # Draw POD markers
        for pod_idx, (px, py) in enumerate(pods_xy):
            # Default: unknown/unassigned
            st = status.get(int(pod_idx), {"entry": None, "id": None, "captured": False})
            ent = st.get("entry")
            tid = st.get("id")
            captured = bool(st.get("captured", False))

            cx, cy = int(round(px)), int(round(py))

            if ent is None or tid is None:
                # Unassigned/unknown: gray ring + '?'
                ring = (180, 180, 180)
                cv2.circle(img, (cx, cy), r, ring, 3)
                cv2.putText(img, "?", (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ring, 2)
            else:
                col = _color_for_id(int(tid))
                if captured:
                    # Captured: filled
                    cv2.circle(img, (cx, cy), r, col, -1)
                    cv2.circle(img, (cx, cy), r, (255, 255, 255), 2)
                else:
                    # Assigned but not captured: outline + X
                    cv2.circle(img, (cx, cy), r, col, 3)
                    cv2.line(img, (cx - r + 3, cy - r + 3), (cx + r - 3, cy + r - 3), col, 3)
                    cv2.line(img, (cx - r + 3, cy + r - 3), (cx + r - 3, cy - r + 3), col, 3)

            # POD label (small)
            label = f"P{pod_idx}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            lx, ly = cx - tw // 2, cy - r - 6
            # outline for readability
            cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Simple legend (top-right) showing entrant->color swatches
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        pad = 12
        sw = 14
        gap = 10
        line_h = 22

        lines = [title] + [f"Entrant #{ent}" for ent, _tid in entrant_items]
        max_w = 0
        for t in lines:
            (tw, _), _ = cv2.getTextSize(t, font, font_scale, thickness)
            max_w = max(max_w, tw)

        panel_w = pad * 2 + sw + gap + max_w
        panel_h = pad * 2 + line_h * max(1, len(lines))

        # Extend canvas (right margin for legend; add bottom margin if panel is taller than image)
        extra_right = panel_w + pad * 2
        extra_bottom = max(0, panel_h + pad * 2 - h)

        bg = tuple(int(x) for x in img[0, 0].tolist())
        canvas = np.full((h + extra_bottom, w0 + extra_right, 3), bg, dtype=np.uint8)
        canvas[:h, :w0] = img
        img = canvas
        h, w = img.shape[:2]

        # Legend anchor in the new right margin
        x0 = w0 + pad
        y0 = pad

        # Panel background
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (220, 220, 220), 2)

        # Title
        cv2.putText(img, title, (x0 + pad, y0 + pad + 16), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        y = y0 + pad + line_h
        for ent, tid in entrant_items:
            col = _color_for_id(int(tid))
            sx1, sy1 = x0 + pad, y + 4
            sx2, sy2 = sx1 + sw, sy1 + sw
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), col, -1)
            cv2.rectangle(img, (sx1, sy1), (sx2, sy2), (255, 255, 255), 1)
            txt = f"Entrant #{ent}"
            cv2.putText(img, txt, (sx2 + gap, y + 16), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y += line_h

        cv2.imwrite(os.path.join(output_folder, out_name), img)