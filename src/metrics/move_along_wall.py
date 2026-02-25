import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from shapely.geometry import Point, MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import Polygon

from .metric import AbstractMetric
from .utils import buffer_shapely_polygon


class MoveAlongWall_Metric(AbstractMetric):
    def __init__(self, config, pWall: float = 0.2) -> None:
        super().__init__(config)
        self.metricName = "STAY_ALONG_WALL"

        self.boundary_region: Polygon = config["Boundary"]

        # Instructor-configurable wall-band thickness.
        # Priority:
        #   1) config["stay_along_wall_pWall"] (recommended)
        #   2) config["pWall"] (legacy/short name)
        #   3) constructor default argument
        cfg_pwall = config.get("stay_along_wall_pWall", config.get("pWall", pWall))
        self.pWall = float(cfg_pwall)

        # Optional: explicit wall-band thickness in map pixels (preferred over factor scaling)
        cfg_wall_px = config.get("stay_along_wall_distance_px", None)
        self.wall_distance_px = float(cfg_wall_px) if cfg_wall_px is not None else None

        # Interior (inset) polygon: points OUTSIDE this polygon are considered "along the wall".
        # pWall controls how far in from the boundary the interior polygon sits.
        # NOTE: buffer_shapely_polygon returns a Shapely geometry (Polygon or MultiPolygon).
        # For concave rooms, an inward buffer can split into multiple pieces; we keep the largest.
        interior_geom: BaseGeometry = buffer_shapely_polygon(
            self.boundary_region,
            self.pWall,
            distance_px=self.wall_distance_px,
        )
        if interior_geom is None or getattr(interior_geom, "is_empty", True):
            # Degenerate: no interior; treat everything as near-wall (i.e., interior contains nothing)
            self.interior_polygon = Polygon()
        elif isinstance(interior_geom, MultiPolygon):
            self.interior_polygon = max(list(interior_geom.geoms), key=lambda g: g.area)
        elif isinstance(interior_geom, Polygon):
            self.interior_polygon = interior_geom
        else:
            # Fallback: try to coerce to polygon via buffer(0)
            coerced = interior_geom.buffer(0)
            if isinstance(coerced, MultiPolygon):
                self.interior_polygon = max(list(coerced.geoms), key=lambda g: g.area)
            elif isinstance(coerced, Polygon):
                self.interior_polygon = coerced
            else:
                self.interior_polygon = Polygon()

        # Per-track near-wall fractions (0..1)
        self.scores_by_id: List[float] = []
        self.map = config.get("Map Image", None)

    # ------------------------------------------------------------------
    # Metric
    # ------------------------------------------------------------------
    def process(self, ctx):
        """Score how often each teammate stays near the wall.

        A frame counts as "near the wall" when the teammate is OUTSIDE the interior polygon.
        Per teammate, we measure this from first appearance until POD capture (if any),
        otherwise until last seen.
        """
        self.scores_by_id = []

        # track_id -> capture_frame (only when assigned + captured)
        capture_map: Dict[int, int] = {}
        for _, info in (ctx.pod_capture or {}).items():
            aid = info.get("assigned_id")
            cf = info.get("capture_frame")
            if aid is not None and cf is not None:
                try:
                    capture_map[int(aid)] = int(cf)
                except Exception:
                    continue

        enemy_ids = set(self.config.get("enemy_ids", [99]))
        fps = float(self.config.get("frame_rate", 30))
        fps = fps if fps > 0 else 30.0

        for track_id, positions in ctx.tracks_by_id.items():
            if int(track_id) in enemy_ids:
                continue

            frames_present = [i + 1 for i, pos in enumerate(positions) if pos is not None]
            if not frames_present:
                continue

            entry_frame = int(frames_present[0])
            end_frame = int(capture_map.get(int(track_id), frames_present[-1]))
            end_frame = max(entry_frame, min(end_frame, len(positions)))

            near_wall = 0
            seen = 0

            for frame_idx in range(entry_frame, end_frame + 1):
                pos = positions[frame_idx - 1]
                if pos is None:
                    continue

                seen += 1
                x, y = float(pos[0]), float(pos[1])

                # Outside the interior polygon => closer to the wall (desired)
                if not self.interior_polygon.contains(Point(x, y)):
                    near_wall += 1

            if seen > 0:
                self.scores_by_id.append(near_wall / seen)

    def getFinalScore(self) -> float:
        """Return the average near-wall fraction across teammates (higher is better)."""
        if not self.scores_by_id:
            return 0.0
        return round(float(np.mean(self.scores_by_id)), 2)

    # ------------------------------------------------------------------
    # Expert compare
    # ------------------------------------------------------------------
    def expertCompare(
        self,
        session_folder: str,
        expert_folder: str,
        map_image=None,
        _config: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ) -> Dict[str, Any]:
        """Compare trainee vs expert near-wall holding and write two map images."""

        enemy_ids = set(self.config.get("enemy_ids", [99]))
        fps = float(self.config.get("frame_rate", 30))
        fps = fps if fps > 0 else 30.0

        def _pick_latest(folder: str, pattern: str) -> Optional[str]:
            matches = glob.glob(os.path.join(folder, pattern))
            if not matches:
                return None
            matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return matches[0]

        def _coerce_map_image(img_or_path):
            if img_or_path is None:
                return None
            if isinstance(img_or_path, str):
                return cv2.imread(img_or_path) if os.path.exists(img_or_path) else None
            return img_or_path

        def _load_position_cache(folder: str) -> List[Dict[str, Any]]:
            """Load `*_PositionCache.txt` like EntranceVectors (frame,id,mapx,mapy)."""
            cache_path = _pick_latest(folder, "*_PositionCache.txt")
            if cache_path is None:
                raise FileNotFoundError(f"No PositionCache found in {folder}")

            df = pd.read_csv(cache_path)
            cols = {c.strip().lower(): c for c in df.columns}
            frame_col = cols.get("frame")
            id_col = cols.get("id")
            x_col = cols.get("mapx")
            y_col = cols.get("mapy")
            if frame_col is None or id_col is None or x_col is None or y_col is None:
                raise ValueError(f"Unexpected PositionCache format: {cache_path}")

            df = df[[frame_col, id_col, x_col, y_col]].dropna()
            if df.empty:
                return []

            df[frame_col] = df[frame_col].astype(int)
            df[id_col] = df[id_col].astype(int)
            df[x_col] = df[x_col].astype(float)
            df[y_col] = df[y_col].astype(float)

            # Exclude enemies
            df = df[~df[id_col].isin([int(x) for x in enemy_ids])].copy()
            if df.empty:
                return []

            max_frame = int(df[frame_col].max()) if len(df) else 0
            tracks: Dict[int, List[Optional[Tuple[float, float]]]] = {}

            for tid, g in df.groupby(id_col):
                tid_i = int(tid)
                traj: List[Optional[Tuple[float, float]]] = [None] * max_frame
                for _, row in g.iterrows():
                    fidx = int(row[frame_col])
                    if 1 <= fidx <= max_frame:
                        traj[fidx - 1] = (float(row[x_col]), float(row[y_col]))
                tracks[tid_i] = traj

            return [{"id": int(tid), "traj": traj} for tid, traj in tracks.items()]

        def _load_capture_frames(folder: str) -> Dict[int, int]:
            """Load `*_PodCache.txt` and return track_id -> capture_frame."""
            cache_path = _pick_latest(folder, "*_PodCache.txt")
            if cache_path is None:
                return {}

            try:
                df = pd.read_csv(cache_path)
            except Exception:
                return {}

            if df is None or df.empty:
                return {}

            cols = {c.strip().lower(): c for c in df.columns}
            aid_col = cols.get("assigned_id")
            cf_col = cols.get("capture_frame")
            if aid_col is None or cf_col is None:
                return {}

            out: Dict[int, int] = {}
            for _, r in df[[aid_col, cf_col]].dropna().iterrows():
                try:
                    out[int(r[aid_col])] = int(r[cf_col])
                except Exception:
                    continue
            return out

        def _first_valid_index(trk: List[Optional[Tuple[float, float]]]) -> Optional[int]:
            for i, v in enumerate(trk):
                if v is not None:
                    return i
            return None

        def _last_valid_index(trk: List[Optional[Tuple[float, float]]]) -> Optional[int]:
            for i in range(len(trk) - 1, -1, -1):
                if trk[i] is not None:
                    return i
            return None

        def _entry_map(track_dicts: List[Dict[str, Any]]) -> Dict[int, int]:
            starts = []
            for t in track_dicts:
                tid = int(t["id"])
                first = _first_valid_index(t["traj"])
                if first is not None:
                    starts.append((tid, int(first)))
            starts.sort(key=lambda x: x[1])
            return {tid: i + 1 for i, (tid, _f) in enumerate(starts)}

        def _near_wall_stats(
            traj: List[Optional[Tuple[float, float]]],
            entry_f: int,
            end_f: int,
        ) -> Tuple[float, float]:
            """Return (score, outside_time_sec).

            Score = fraction of seen frames that are OUTSIDE the interior polygon (safe).
            Outside time = seconds spent INSIDE the interior polygon (out of safe band).
            """
            safe = 0
            inside = 0
            seen = 0
            for f in range(entry_f, end_f + 1):
                pt = traj[f - 1]
                if pt is None:
                    continue
                seen += 1
                x, y = float(pt[0]), float(pt[1])
                if self.interior_polygon.contains(Point(x, y)):
                    inside += 1
                else:
                    safe += 1

            score = float(safe) / float(seen) if seen > 0 else 0.0
            outside_sec = float(inside) / float(fps) if seen > 0 else 0.0
            return score, outside_sec

        def _crop_end_frame(traj: List[Optional[Tuple[float, float]]], entry_f: int, end_f: int) -> int:
            last = _last_valid_index(traj)
            if last is None:
                return entry_f
            end_f = min(end_f, last + 1)
            return max(entry_f, end_f)

        # Resolve map image
        if map_image is None:
            map_image = self.map
        map_image = _coerce_map_image(map_image)

        expert_img_path = os.path.join(session_folder, "STAY_ALONG_WALL_Expert.jpg")
        trainee_img_path = os.path.join(session_folder, "STAY_ALONG_WALL_Trainee.jpg")
        txt_path = os.path.join(session_folder, "STAY_ALONG_WALL_Comparison.txt")
        os.makedirs(session_folder, exist_ok=True)

        if map_image is None:
            err_text = "There was an error while processing this comparison. Missing map image (config['Map Image'] / self.map)."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "STAY_ALONG_WALL",
                "Type": "SideBySide",
                "ExpertImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Load caches
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
                "Name": "STAY_ALONG_WALL",
                "Type": "SideBySide",
                "ExpertImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        if len(expert_tracks) == 0 or len(trainee_tracks) == 0:
            err_text = "There was an error while processing this comparison. No valid tracks found."
            try:
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(err_text)
            except Exception:
                pass
            return {
                "Name": "STAY_ALONG_WALL",
                "Type": "SideBySide",
                "ExpertImageLocation": expert_img_path,
                "TraineeImageLocation": trainee_img_path,
                "TxtLocation": txt_path,
                "Text": err_text,
            }

        # Point-of-dominance cutoff: stop trajectories at capture_frame when available.
        expert_cap = _load_capture_frames(expert_folder)
        trainee_cap = _load_capture_frames(session_folder)

        expert_entry = _entry_map(expert_tracks)
        trainee_entry = _entry_map(trainee_tracks)

        def _prep_infos(track_dicts: List[Dict[str, Any]], cap_map: Dict[int, int], entry_map: Dict[int, int]):
            infos = []
            for t in track_dicts:
                tid = int(t["id"])
                traj = t["traj"]
                first = _first_valid_index(traj)
                if first is None:
                    continue
                entry_f = int(first) + 1
                last_seen = (_last_valid_index(traj) or first) + 1
                reached_pod = tid in cap_map
                end_f = int(cap_map.get(tid, last_seen))
                end_f = _crop_end_frame(traj, entry_f, end_f)
                sc, outside_sec = _near_wall_stats(traj, entry_f, end_f)
                infos.append(
                    {
                        "track_id": tid,
                        "entry_number": int(entry_map.get(tid, 0)) or None,
                        "traj": traj,
                        "entry_frame": entry_f,
                        "end_frame": end_f,
                        "score": float(sc),
                        "outside_time_sec": float(outside_sec),
                        "reached_pod": bool(reached_pod),
                    }
                )
            # Sort by entry number (time of first appearance)
            infos.sort(key=lambda d: int(d["entry_number"] or 10**9))
            return infos

        expert_infos = _prep_infos(expert_tracks, expert_cap, expert_entry)
        trainee_infos = _prep_infos(trainee_tracks, trainee_cap, trainee_entry)

        # Draw two map panels
        self.__generateExpertCompareGraphic(
            output_folder=session_folder,
            map_view=map_image,
            boundary=self.boundary_region,
            interior=self.interior_polygon,
            infos=expert_infos,
            out_name="STAY_ALONG_WALL_Expert.jpg",
            title="Expert",
        )
        self.__generateExpertCompareGraphic(
            output_folder=session_folder,
            map_view=map_image,
            boundary=self.boundary_region,
            interior=self.interior_polygon,
            infos=trainee_infos,
            out_name="STAY_ALONG_WALL_Trainee.jpg",
            title="Trainee",
        )

        # ---- Summary + details ----
        ex_scores = [float(i.get("score", 0.0)) for i in expert_infos]
        tr_scores = [float(i.get("score", 0.0)) for i in trainee_infos]
        ex_outside = [float(i.get("outside_time_sec", 0.0)) for i in expert_infos]
        tr_outside = [float(i.get("outside_time_sec", 0.0)) for i in trainee_infos]

        ex_final = float(np.mean(ex_scores)) if ex_scores else 0.0
        tr_final = float(np.mean(tr_scores)) if tr_scores else 0.0
        delta_final = tr_final - ex_final

        ex_out_avg = float(np.mean(ex_outside)) if ex_outside else 0.0
        tr_out_avg = float(np.mean(tr_outside)) if tr_outside else 0.0
        delta_out = tr_out_avg - ex_out_avg

        if abs(delta_out) < 1e-6:
            outside_part = (
                f"On average, the trainee spent about {tr_out_avg:.2f}s out of the safe band (same as the expert). "
            )
        elif delta_out < 0:
            outside_part = (
                f"On average, the trainee spent about {abs(delta_out):.2f}s less out of the safe band than the expert "
                f"(T {tr_out_avg:.2f}s vs E {ex_out_avg:.2f}s)."
            )
        else:
            outside_part = (
                f"On average, the trainee spent about {abs(delta_out):.2f}s more out of the safe band than the expert "
                f"(T {tr_out_avg:.2f}s vs E {ex_out_avg:.2f}s)."
            )

        if abs(delta_final) < 0.01:
            score_part = (
                f"Overall near-wall score looks similar (Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
            )
        elif delta_final > 0:
            score_part = (
                f"Overall near-wall score looks better than the expert (Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
            )
        else:
            score_part = (
                f"Overall near-wall score looks worse than the expert (Trainee {tr_final:.2f} vs Expert {ex_final:.2f}, Δ {delta_final:+.2f})."
            )


        # Per-entry details aligned by entry order
        max_n = max(len(expert_infos), len(trainee_infos))
        rows: List[str] = []

        header = (
            "Entry #, Trainee ID, Trainee Score, Trainee Time Outside (s), Expert ID, Expert Score, Expert Time Outside (s), "
            "Time Δ (T−E), Score Δ (T−E), Performance\n"
        )

        eps_score = 0.01
        eps_time = 0.10

        for i in range(max_n):
            e = expert_infos[i] if i < len(expert_infos) else None
            t = trainee_infos[i] if i < len(trainee_infos) else None

            e_id = e.get("track_id") if e is not None else None
            t_id = t.get("track_id") if t is not None else None
            e_sc = e.get("score") if e is not None else None
            t_sc = t.get("score") if t is not None else None
            e_out = e.get("outside_time_sec") if e is not None else None
            t_out = t.get("outside_time_sec") if t is not None else None

            if (e_sc is not None) and (t_sc is not None):
                ds = float(t_sc) - float(e_sc)
                ds_str = f"{ds:+.2f}"
            else:
                ds = None
                ds_str = "N/A"

            if (e_out is not None) and (t_out is not None):
                dt = float(t_out) - float(e_out)
                dt_str = f"{dt:+.2f}s"
            else:
                dt = None
                dt_str = "N/A"

            # Performance: primarily by score diff (higher is better), tie-break by outside time (lower is better)
            if ds is None:
                perf = "N/A"
            elif abs(ds) <= eps_score:
                if dt is None or abs(dt) <= eps_time:
                    perf = "SIMILAR"
                elif dt < -eps_time:
                    perf = "BETTER"
                else:
                    perf = "WORSE"
            elif ds > eps_score:
                perf = "BETTER"
            else:
                perf = "WORSE"

            rows.append(
                f"{i+1}, "
                f"{t_id if t_id is not None else 'N/A'}, "
                f"{('N/A' if t_sc is None else f'{float(t_sc):.2f}')}, "
                f"{('N/A' if t_out is None else f'{float(t_out):.2f}')}, "
                f"{e_id if e_id is not None else 'N/A'}, "
                f"{('N/A' if e_sc is None else f'{float(e_sc):.2f}')}, "
                f"{('N/A' if e_out is None else f'{float(e_out):.2f}')}, "
                f"{dt_str}, {ds_str}, {perf}"
            )

        # Returned payload remains comma-separated (easy to parse)
        header_line = header.strip()  # remove trailing newline
        details_csv = header_line + "\n" + "\n".join(rows)
        text = outside_part + "\n" + score_part + "\n\n" + details_csv

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

        pretty_headers = [h.strip() for h in header_line.split(",")]
        pretty_rows: List[List[str]] = []
        for ln in rows:
            parts = [p.strip() for p in ln.split(",")]
            while len(parts) < len(pretty_headers):
                parts.append("N/A")
            pretty_rows.append(parts[: len(pretty_headers)])

        details_pretty = _broken_table(pretty_headers, pretty_rows)
        saved_text = outside_part + "\n" + score_part + "\n\n" + details_pretty + "\n"

        # Save the readable table to TXT (but keep comma-separated text in the returned payload)
        try:
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(saved_text)
        except Exception:
            # Non-fatal: still return the summary even if file write fails
            pass

        return {
            "Name": "STAY_ALONG_WALL",
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
        boundary: Polygon,
        interior: Polygon,
        infos: List[Dict[str, Any]],
        out_name: str,
        title: str,
    ) -> None:
        """Write a single map panel with safe-region overlay + colored trajectories."""
        os.makedirs(output_folder, exist_ok=True)

        img = map_view.copy()
        h, w = img.shape[:2]
        w0 = w

        # Palette matches helper_functions / EntranceVectors (BGR)
        predefined_colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128),
        ]

        def _color_for_id(tid: int):
            return predefined_colors[int(tid) % len(predefined_colors)]

        # ---- Safe region overlay (boundary shaded, interior masked out) ----
        overlay = img.copy()

        def _iter_polys(geom):
            if geom is None or getattr(geom, "is_empty", True):
                return []
            if getattr(geom, "geom_type", None) == "Polygon":
                return [geom]
            if getattr(geom, "geom_type", None) == "MultiPolygon":
                return list(geom.geoms)
            return []

        def _poly_pts(poly: Polygon) -> Optional[np.ndarray]:
            try:
                return np.asarray(list(poly.exterior.coords), dtype=np.int32)
            except Exception:
                return None

        # Shade the whole boundary region green
        for poly in _iter_polys(boundary):
            pts = _poly_pts(poly)
            if pts is not None:
                cv2.fillPoly(overlay, [pts], (60, 180, 60))

        # Mask out the interior region back to the original image
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in _iter_polys(interior):
            pts = _poly_pts(poly)
            if pts is not None:
                cv2.fillPoly(mask, [pts], 255)

        overlay[mask == 255] = img[mask == 255]

        # Blend overlay (interior remains unchanged because overlay==img there)
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)

        # Boundary outline (supports Polygon/MultiPolygon)
        try:
            if getattr(boundary, "geom_type", None) == "MultiPolygon":
                for g in boundary.geoms:
                    bcoords = np.asarray(list(g.exterior.coords), dtype=np.int32)
                    cv2.polylines(img, [bcoords], isClosed=True, color=(255, 255, 255), thickness=2)
            else:
                bcoords = np.asarray(list(boundary.exterior.coords), dtype=np.int32)
                cv2.polylines(img, [bcoords], isClosed=True, color=(255, 255, 255), thickness=2)
        except Exception:
            pass

        # Interior outline (supports Polygon/MultiPolygon)
        try:
            if getattr(interior, "geom_type", None) == "MultiPolygon":
                for g in interior.geoms:
                    icoords = np.asarray(list(g.exterior.coords), dtype=np.int32)
                    cv2.polylines(img, [icoords], isClosed=True, color=(255, 255, 255), thickness=1)
            else:
                icoords = np.asarray(list(interior.exterior.coords), dtype=np.int32)
                cv2.polylines(img, [icoords], isClosed=True, color=(255, 255, 255), thickness=1)
        except Exception:
            pass

        # ---- Trajectories ----

        # We'll dim safe (outside interior) segments and brighten unsafe segments.
        safe_segments: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int, int]]] = []
        unsafe_segments: List[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int, int]]] = []
        safe_alpha = 0.35

        # Draw each entrant
        for info in infos:
            tid = int(info.get("track_id"))
            entry_num = info.get("entry_number")
            traj: List[Optional[Tuple[float, float]]] = info.get("traj")
            entry_f = int(info.get("entry_frame"))
            end_f = int(info.get("end_frame"))

            color = _color_for_id(tid)

            # Collect points and whether each point is inside the interior (unsafe)
            pts: List[Optional[Tuple[int, int]]] = []
            inside_flags: List[Optional[bool]] = []
            for f in range(entry_f, end_f + 1):
                pt = traj[f - 1] if 0 <= (f - 1) < len(traj) else None
                if pt is None:
                    pts.append(None)
                    inside_flags.append(None)
                    continue
                x, y = float(pt[0]), float(pt[1])
                pxy = (int(round(x)), int(round(y)))
                pts.append(pxy)
                inside_flags.append(bool(interior.contains(Point(x, y))))

            # Separate safe and unsafe segments
            prev_p: Optional[Tuple[int, int]] = None
            prev_inside: Optional[bool] = None
            for p, inside in zip(pts, inside_flags):
                if p is None or inside is None:
                    prev_p = None
                    prev_inside = None
                    continue

                if prev_p is not None and prev_inside is not None:
                    # Treat a segment as unsafe if either endpoint is inside the interior.
                    is_unsafe = bool(inside) or bool(prev_inside)
                    if is_unsafe:
                        unsafe_segments.append((prev_p, p, color))
                    else:
                        safe_segments.append((prev_p, p, color))

                prev_p = p
                prev_inside = inside

        # Draw safe segments translucently (less emphasis)
        if safe_segments:
            safe_overlay = img.copy()
            for p1, p2, col in safe_segments:
                cv2.line(safe_overlay, p1, p2, col, 2, cv2.LINE_AA)
            img = cv2.addWeighted(safe_overlay, safe_alpha, img, 1.0 - safe_alpha, 0)

        # Draw unsafe segments brightly (highlight) with a stronger halo, keeping the same color
        for p1, p2, col in unsafe_segments:
            cv2.line(img, p1, p2, (255, 255, 255), 6, cv2.LINE_AA)  # halo
            cv2.line(img, p1, p2, col, 3, cv2.LINE_AA)

        # Draw a star where each entrant reaches their POD (capture frame only)
        for info in infos:
            if not info.get("reached_pod", False):
                continue
            traj = info.get("traj")
            end_f = int(info.get("end_frame"))
            color = _color_for_id(int(info.get("track_id")))

            # Find the last valid point at/before end_f
            star_pt = None
            for f in range(end_f, 0, -1):
                if 0 <= (f - 1) < len(traj) and traj[f - 1] is not None:
                    x, y = float(traj[f - 1][0]), float(traj[f - 1][1])
                    star_pt = (int(round(x)), int(round(y)))
                    break
            if star_pt is None:
                continue

            # Outline + color star
            cv2.drawMarker(img, star_pt, (0, 0, 0), markerType=cv2.MARKER_STAR, markerSize=16, thickness=2, line_type=cv2.LINE_AA)
            cv2.drawMarker(img, star_pt, color, markerType=cv2.MARKER_STAR, markerSize=16, thickness=1, line_type=cv2.LINE_AA)

        # ---- Legend panel (outside map, right margin) ----
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        pad = 12
        sw = 14
        gap = 10
        line_h = 22

        items = []
        for info in infos:
            en = info.get("entry_number")
            tid = info.get("track_id")
            if en is None or tid is None:
                continue
            items.append((int(en), int(tid)))
        items.sort(key=lambda t: t[0])

        lines = [title, "Safe band: green", "Star = POD reached"] + [f"Entrant #{en}" for en, _tid in items]
        max_w = 0
        for t in lines:
            (tw, _), _ = cv2.getTextSize(t, font, font_scale, thickness)
            max_w = max(max_w, tw)

        panel_w = pad * 2 + sw + gap + max_w
        panel_h = pad * 2 + line_h * max(1, len(lines))

        # Extend the image to the right so the legend sits outside the map.
        extra_right = panel_w + pad * 2
        bg = tuple(int(x) for x in img[0, 0].tolist())
        canvas = np.full((h, w0 + extra_right, 3), bg, dtype=np.uint8)
        canvas[:, :w0] = img
        img = canvas
        w = img.shape[1]

        # Legend anchor in the new right margin
        x0 = w0 + pad
        y0 = pad

        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
        cv2.rectangle(img, (x0, y0), (x0 + panel_w, y0 + panel_h), (220, 220, 220), 2)

        y = y0 + pad + 16
        cv2.putText(img, title, (x0 + pad, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        y = y0 + pad + line_h
        # Safe band row
        cv2.rectangle(img, (x0 + pad, y + 4), (x0 + pad + sw, y + 4 + sw), (60, 180, 60), -1)
        cv2.putText(
            img,
            "Safe band: green",
            (x0 + pad + sw + gap, y + 16),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_h

        # Star row (POD reached)
        star_cx = x0 + pad + sw // 2
        star_cy = y + 12
        cv2.drawMarker(img, (star_cx, star_cy), (0, 0, 0), markerType=cv2.MARKER_STAR, markerSize=12, thickness=2, line_type=cv2.LINE_AA)
        cv2.drawMarker(img, (star_cx, star_cy), (255, 255, 255), markerType=cv2.MARKER_STAR, markerSize=12, thickness=1, line_type=cv2.LINE_AA)
        cv2.putText(
            img,
            "Star = POD reached",
            (x0 + pad + sw + gap, y + 16),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_h

        for en, tid in items:
            col = _color_for_id(tid)
            cv2.rectangle(img, (x0 + pad, y + 4), (x0 + pad + sw, y + 4 + sw), col, -1)
            cv2.rectangle(img, (x0 + pad, y + 4), (x0 + pad + sw, y + 4 + sw), (255, 255, 255), 1)
            txt = f"Entrant #{en}"
            cv2.putText(img, txt, (x0 + pad + sw + gap, y + 16), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y += line_h

        cv2.imwrite(os.path.join(output_folder, out_name), img)