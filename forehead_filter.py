"""Forehead slope filter for neutral/expression face masks (FLAME without eyes: 3931 vertices, 7800 faces).

A forehead that wrapped onto hair (fringe) bulges forward instead of rolling back towards the hairline. On the
wrap a face mask is computed on, this flags such faces in the forehead band so they are left out of the mask
(and the skin targets pinned to their vertices are dropped), so the final wrap is not pulled onto the hair.

Band (forehead_band.json, precomputed from the template): faces between the 2nd outer keyline and the outer
(hairline) keyline above eye level, plus one row of faces just below the 2nd keyline (checked) and one more row
(reference only). Each band vertex has a harmonic coordinate s (uniform-weight Laplace solve on the template
topology: 1 on the hairline keyline, 0 on the 2nd keyline, -0.25 / -0.5 on the rows below), so on any wrap the
per-face gradient of s is the in-surface direction across the band towards the hairline.

Per checked face, on the wrap V:
  slope = unit(grad s) . forward, forward = perpendicular to up (chin -> brow midpoint) and right (outer eye
          corners), pointing to the nose tip;   > 0 means the surface comes forward going to the hairline.
  bend  = [slope - mean slope of the edge neighbours with lower s] on V minus the same on the template;
          > 0 means the surface bends outward there even if it still rolls back.
Flagged = slope > tau (tau_row on the row below the 2nd keyline) OR bend > tau_bend, then every checked face
above a flagged one (edge neighbours with higher s, repeated up to the hairline).
Defaults (0.1 / 0.2 / 0.3) from 223 Nersemble neutral wraps: slope 99th pct -0.03 (band) / 0.03 (row), bend
99th pct 0.13. Sandbox test: ava_forehead_slope/batch (22 Ava-256 captures).
"""
from __future__ import annotations

import collections
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

BAND_PATH = Path(__file__).with_name("forehead_band.json")
# Nersemble: removed faces stored next to the face mask files (<mask>_forehead.json), read by run_pipeline_all.py.
SIDECAR_SUFFIX = "_forehead.json"
NUM_FACES = 7800
NUM_VERTICES = 3931


@lru_cache(maxsize=4)
def _band(path: str = str(BAND_PATH)) -> dict:
    data = json.load(open(path))
    s = np.full(NUM_VERTICES, np.nan)
    for v, val in data["s"].items():
        s[int(v)] = val
    data["s_arr"] = s
    data["check_set"] = set(data["check_faces"])
    data["row_set"] = set(data["row_faces"])
    data["all_faces"] = sorted(data["check_set"] | set(data["reference_faces"]))
    data["template_bend"] = {int(k): v for k, v in data["template_bend"].items()}
    return data


@lru_cache(maxsize=4)
def _neighbours(faces_key: bytes, n_faces: int, path: str = str(BAND_PATH)) -> tuple[dict, dict]:
    faces = np.frombuffer(faces_key, dtype=np.int64).reshape(n_faces, 3)
    b = _band(path); s = b["s_arr"]
    sf = {f: float(s[faces[f]].mean()) for f in b["all_faces"]}
    edge_faces = collections.defaultdict(list)
    for f in b["all_faces"]:
        a, c, d = faces[f]
        for e in ((a, c), (c, d), (d, a)):
            edge_faces[tuple(sorted((int(e[0]), int(e[1]))))].append(f)
    lower = {f: [] for f in b["all_faces"]}; upper = {f: [] for f in b["all_faces"]}
    for fs in edge_faces.values():
        for x in fs:
            for y in fs:
                if y != x and sf[y] < sf[x] - 1e-6:
                    lower[x].append(y); upper[y].append(x)
    return lower, upper


def forward(V: np.ndarray, path: str = str(BAND_PATH)) -> np.ndarray:
    fv = _band(path)["frame_vertices"]
    brow = V[fv["brow"]].mean(0)
    up = brow - V[fv["chin"]]; up /= np.linalg.norm(up)
    right = V[fv["eye_right"]] - V[fv["eye_left"]]; right -= (right @ up) * up; right /= np.linalg.norm(right)
    fwd = np.cross(right, up)
    return fwd if fwd @ (V[fv["nose_tip"]] - brow) > 0 else -fwd


def face_slopes(V: np.ndarray, faces: np.ndarray, path: str = str(BAND_PATH)) -> dict[int, float]:
    """face -> slope (unit gradient of s . forward) for every band face (checked + reference)."""
    b = _band(path); fl = b["all_faces"]
    tri = V[faces[fl]]; s = b["s_arr"][faces[fl]]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); a2 = (n ** 2).sum(1, keepdims=True)
    g = (np.cross(n, tri[:, 2] - tri[:, 1]) * s[:, :1] + np.cross(n, tri[:, 0] - tri[:, 2]) * s[:, 1:2]
         + np.cross(n, tri[:, 1] - tri[:, 0]) * s[:, 2:3]) / a2
    g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
    return dict(zip(fl, (g @ forward(V, path)).tolist()))


def flag_faces(V: np.ndarray, faces: np.ndarray, included, tau: float | None = None, tau_row: float | None = None,
               tau_bend: float | None = None, path: str = str(BAND_PATH)) -> dict:
    """Faces of `included` (the face mask's included set, on the wrap V it was computed on) to remove.
    Returns {"removed": sorted list, "absolute": [...], "bend": [...], "above": [...]}."""
    V = np.asarray(V, dtype=np.float64); faces = np.asarray(faces, dtype=np.int64)
    if faces.shape != (NUM_FACES, 3):
        raise ValueError(f"forehead filter expects the FLAME no-eyes topology ({NUM_FACES} faces), got {faces.shape}")
    b = _band(path); d = b["defaults"]
    tau = d["tau"] if tau is None else tau; tau_row = d["tau_row"] if tau_row is None else tau_row
    tau_bend = d["tau_bend"] if tau_bend is None else tau_bend
    inc = set(int(f) for f in included)
    lower, upper = _neighbours(faces.tobytes(), len(faces), path)
    sl = face_slopes(V, faces, path)
    absolute, bend = set(), set()
    for f in b["check_faces"]:
        if f not in inc:
            continue
        if sl[f] > (tau_row if f in b["row_set"] else tau):
            absolute.add(f)
        lo = lower[f]
        if lo and f in b["template_bend"] and (sl[f] - np.mean([sl[y] for y in lo])) - b["template_bend"][f] > tau_bend:
            bend.add(f)
    seen = absolute | bend; stack = list(seen)
    while stack:
        x = stack.pop()
        for y in upper[x]:
            if y in b["check_set"] and y not in seen:
                seen.add(y); stack.append(y)
    above = {f for f in seen - absolute - bend if f in inc}
    removed = (absolute | bend | above) & inc
    return {"removed": sorted(removed), "absolute": sorted(absolute), "bend": sorted(bend), "above": sorted(above),
            "params": {"tau": tau, "tau_row": tau_row, "tau_bend": tau_bend}}


def targets_on_removed(pot_rows, faces: np.ndarray, removed, first_index: int = 74) -> list[int]:
    """Correspondence rows (index >= first_index, i.e. skin) pinned to a vertex of a removed face. A POT row
    [face, u, w] pins one corner vertex of its face, which can be shared with a removed neighbour."""
    if not removed:
        return []
    faces = np.asarray(faces, dtype=np.int64)
    removed_v = set(faces[sorted(removed)].ravel().tolist())
    out = []
    for i in range(first_index, len(pot_rows)):
        face_index, u, w = pot_rows[i][:3]
        # same corner convention as mesh_utils.pot_row_local_corner / scan_face_masking.pot_row_vertex
        corner = 0 if u >= 0.5 else 1 if w >= 0.5 else 2
        if int(faces[int(face_index)][corner]) in removed_v:
            out.append(i)
    return out


def merge_into_landmark_filter(payload: dict, forehead: dict, target_rows: list[int]) -> dict:
    """Adds the forehead filter's dropped targets to a landmark-filter payload ({"dropped", "kept", ...}), so
    every consumer of "dropped" leaves them out; the filter result is recorded under "forehead_filter"."""
    rows = set(int(i) for i in target_rows)
    payload = dict(payload)
    payload["dropped"] = sorted(set(int(i) for i in payload.get("dropped", [])) | rows)
    if "kept" in payload:
        payload["kept"] = [i for i in payload["kept"] if int(i) not in rows]
    payload["forehead_filter"] = {
        "removed_faces": forehead["removed"], "dropped_targets": sorted(rows),
        "n_absolute": len(forehead["absolute"]), "n_bend": len(forehead["bend"]), "n_above": len(forehead["above"]),
        "params": forehead["params"],
    }
    return payload


def summary(forehead: dict, target_rows: list[int]) -> str:
    return (f"forehead filter removed {len(forehead['removed'])} face(s) (absolute {len(forehead['absolute'])}, "
            f"bend {len(forehead['bend'])}, above {len(forehead['above'])}), dropped target(s) {sorted(target_rows)}")
