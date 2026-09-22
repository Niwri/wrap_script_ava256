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


def params_path(output_root: Path, capture_id: str) -> Path:
    return Path(output_root) / capture_id / "dynamic_transform_params.npz"


def save_params(path: Path, params: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **params)


def load_params(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {"scale": float(data["scale"]), "scan_center": data["scan_center"], "neutral_center": data["neutral_center"]}


def get_or_compute_params(output_root: Path, capture_id: str, neutral_mesh_path: Path, neutral_scan_verts: np.ndarray) -> dict:
    path = params_path(output_root, capture_id)
    if path.exists():
        return load_params(path)
    params = compute_params(neutral_mesh_path, neutral_scan_verts)
    save_params(path, params)
    return params


def transform(verts: np.ndarray, params: dict) -> np.ndarray:
    """real-world -> FLAME-template-comparable (small) frame."""
    return (verts - params["scan_center"]) * float(params["scale"]) + params["neutral_center"]


def untransform(verts: np.ndarray, params: dict) -> np.ndarray:
    """FLAME-template-comparable (small) frame -> real-world."""
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
