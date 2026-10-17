"""Dynamic per-capture scale/translate normalization between Ava-256's
kinematic_tracking world space (~100-1000 units) and the FLAME template's own
local space (~0.2 units) -- validated by hand earlier this session on capture
20210810--1306--FXN596 (wrapping without this produced "incredibly deformed"
output; wrapping the transformed scan/correspondence, then un-transforming
the wrap output for camera-space work, produced a clean result). No rotation
(user explicitly not concerned with it) -- uniform scalar scale from the bbox
diagonal ratio, translate scan bbox center to neutral bbox center. Computed
ONCE per capture from the neutral frame's own scan bbox (head scale/rough
position doesn't meaningfully change across expressions of the same actor),
cached to <output_root>/<capture_id>/dynamic_transform_params.npz and reused
for every other frame's wrap.

Two forms of the scale (2026-10-16):
  - "face_p5" (default for new captures, compute_face_params): a 3x3 linear map
    A along the face's own axes from the neutral frame's keypoints_3d --
    vertical = template / capture brow-chin (standard 30/47/69/72 mean -> 50),
    horizontal = template / capture outer-eye-corner distance (16 <-> 35),
    depth = sqrt(vertical * horizontal); transform(v) = (v - scan_center) @ A.T
    + neutral_center. The bbox diagonal (below) includes hair and neck, so it
    shrank most faces ~10% below the template and pushed the forehead keyline
    up (~6.7 mm vs Nersemble); the face-based map brings it to ~2.6 mm with no
    landmark-error change (22-capture sandbox comparison, "P5").
  - "bbox" (compute_params): the original uniform scalar -- the fallback when
    the neutral frame lacks those keypoints, and what every params file saved
    before this change holds (loaded and applied exactly as before).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def _bbox_center_extent(verts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    return (lo + hi) / 2.0, hi - lo


def compute_params(neutral_mesh_path: Path, scan_verts: np.ndarray) -> dict[str, np.ndarray]:
    neutral_verts = np.asarray(trimesh.load(str(neutral_mesh_path), process=False).vertices, dtype=np.float64)
    neutral_center, neutral_extent = _bbox_center_extent(neutral_verts)
    scan_center, scan_extent = _bbox_center_extent(scan_verts)
    scale = float(np.linalg.norm(neutral_extent) / np.linalg.norm(scan_extent))
    return {"scale": np.array(scale), "scan_center": scan_center, "neutral_center": neutral_center}


BROW_INDICES = (30, 47, 69, 72)  # standard landmarks averaged into the brow point
CHIN_INDEX = 50
OUTER_EYE_CORNERS = (16, 35)  # left, right


def _face_frame(points: dict[int, np.ndarray]) -> tuple[np.ndarray, float, float]:
    """(R rows = right, up, forward of the face; brow-chin length; outer-eye-corner distance)."""
    brow = np.mean([points[i] for i in BROW_INDICES], axis=0)
    up = brow - points[CHIN_INDEX]
    brow_chin = float(np.linalg.norm(up))
    up /= brow_chin
    right = points[OUTER_EYE_CORNERS[0]] - points[OUTER_EYE_CORNERS[1]]
    eye_width = float(np.linalg.norm(right))
    right = right - (right @ up) * up
    right /= np.linalg.norm(right)
    return np.stack([right, up, np.cross(right, up)]), brow_chin, eye_width


def compute_face_params(neutral_mesh_path: Path, pot_rows: list, keypoints_3d: dict, scan_verts: np.ndarray) -> dict | None:
    """The "face_p5" params (see module docstring), or None when the neutral
    frame's keypoints_3d lacks a brow, chin or outer-eye-corner point. Centres
    are compute_params()'s (bbox), as validated."""
    import landmark_map
    import mesh_utils

    needed = (*BROW_INDICES, CHIN_INDEX, *OUTER_EYE_CORNERS)
    ids = landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID
    if not all(ids.get(i) in keypoints_3d for i in needed):
        return None
    mesh = trimesh.load(str(neutral_mesh_path), process=False)
    tv, tf = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)
    template = {i: tv[mesh_utils.pot_row_vertex_index(tf, pot_rows[i])] for i in needed}
    capture = {i: np.asarray(keypoints_3d[ids[i]], dtype=np.float64) for i in needed}
    _, t_v, t_h = _face_frame(template)
    R, k_v, k_h = _face_frame(capture)
    s_v, s_h = t_v / k_v, t_h / k_h
    A = R.T @ np.diag([s_h, s_v, np.sqrt(s_h * s_v)]) @ R
    base = compute_params(neutral_mesh_path, scan_verts)
    return {"A": A, "scale": np.array(float(np.cbrt(abs(np.linalg.det(A))))), "scan_center": base["scan_center"],
            "neutral_center": base["neutral_center"], "method": np.array("face_p5")}


def params_path(output_root: Path, capture_id: str) -> Path:
    return Path(output_root) / capture_id / "dynamic_transform_params.npz"


def save_params(path: Path, params: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **params)


def load_params(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    params = {"scale": float(data["scale"]), "scan_center": data["scan_center"], "neutral_center": data["neutral_center"]}
    if "A" in data.files:
        params["A"] = np.asarray(data["A"], dtype=np.float64)
    params["method"] = str(data["method"]) if "method" in data.files else "bbox"
    return params


def get_or_compute_params(output_root: Path, capture_id: str, neutral_mesh_path: Path, neutral_scan_verts: np.ndarray) -> dict:
    path = params_path(output_root, capture_id)
    if path.exists():
        return load_params(path)
    params = compute_params(neutral_mesh_path, neutral_scan_verts)
    save_params(path, params)
    return params


def transform(verts: np.ndarray, params: dict) -> np.ndarray:
    """real-world -> FLAME-template-comparable (small) frame."""
    if "A" in params:
        return (np.asarray(verts) - params["scan_center"]) @ np.asarray(params["A"]).T + params["neutral_center"]
    return (verts - params["scan_center"]) * float(params["scale"]) + params["neutral_center"]


def untransform(verts: np.ndarray, params: dict) -> np.ndarray:
    """FLAME-template-comparable (small) frame -> real-world."""
    if "A" in params:
        return (np.asarray(verts) - params["neutral_center"]) @ np.linalg.inv(np.asarray(params["A"])).T + params["scan_center"]
    return (verts - params["neutral_center"]) / float(params["scale"]) + params["scan_center"]


def transform_target_points(target_points: list[dict[str, float]], params: dict) -> list[dict[str, float]]:
    """Transforms every non-placeholder ({0,0,0}) row in place-equivalent
    (returns a new list); placeholder rows are left untouched so they stay
    recognizable as "no data" rather than mapping to the new origin."""
    out = []
    for p in target_points:
        xyz = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
        if np.allclose(xyz, 0.0):
            out.append(dict(p))
            continue
        txyz = transform(xyz[None, :], params)[0]
        out.append({"x": float(txyz[0]), "y": float(txyz[1]), "z": float(txyz[2])})
    return out
