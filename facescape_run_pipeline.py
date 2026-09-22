#!/usr/bin/env python3
"""Ava-256 local copy of wrap_script_facescape/run_pipeline.py, stripped down
to ONLY what mesh_utils._facescape_run_pipeline_module() actually calls:
load_skin_vertex_indices() and _compute_skin_pot_entries() (confirmed via
grep -- nothing else from that original file is ever touched by this
pipeline). Everything else in the original -- run_pipeline()/main() (the
actual FaceScape wrap driver, not used here), ensure_extended_correspondence_
files() (Ava-256 has its own equivalent, mesh_utils.build_live_extended_pot()),
render_overlay_frame/wrap_mesh/triangulate/is_in_bounds imports, and
`from backend.app.core.config import get_settings` / `backend.app.services.
facescape_neutral` / `backend.app.services.label_tracker` (which needed a
hardcoded `sys.path.insert(0, ".../FaceView")` -- a real, previously-missed
FaceView dependency this pipeline otherwise has none of) -- has been removed
as dead weight for our purposes, the same way triangulate.py was stripped of
its own unused get_settings dependency.

skin_landmarks.py is now a local copy too (wrap_script_ava256/skin_landmarks.py,
byte-identical to wrap_script/skin_landmarks.py -- already self-contained, its
own `ROOT = Path(__file__).resolve().parent` just needed to BE this directory
to correctly resolve its keypoint_front.txt/keypoint_right.txt/
keypoint_left.txt/keypoint_under.txt reads to our local copies instead of
wrap_script/'s).
"""
from __future__ import annotations

import sys
from pathlib import Path

import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import skin_landmarks


def load_skin_vertex_indices() -> list[int]:
    """Wrapped-mesh vertex index for each global skin landmark index (74, 75,
    ...), in that same global-index order -- see skin_landmarks.assign_global_indices.
    Same helper as wrap_script/run_pipeline_all.py's own (Nersemble) version."""
    return [vertex_idx for _global_idx, _region, vertex_idx in skin_landmarks.assign_global_indices()]


def _compute_skin_pot_entries(topology_mesh_path: str, skin_vertex_indices: list[int]) -> list[list[float]]:
    """Point-on-triangle (face_index, u, w) entries for each skin vertex index, in
    the same format as the base 74-standard POT file's existing entries. Duplicated
    from wrap_script/run_pipeline_all.py's own version rather than imported -- that
    module pulls in cv2/pytorch3d/meshroom/SAM-U2Net at import time, which would
    defeat this file's own "as minimal as possible" purpose. Convention: a face's
    0th vertex is (1,0), 1st is (0,1), 2nd is (0,0) -- see the Nersemble version's
    own docstring for how this was inferred."""
    mesh = trimesh.load(topology_mesh_path, process=False)
    vertex_to_face: dict[int, tuple[int, int]] = {}
    for face_idx, face in enumerate(mesh.faces):
        for local_pos, vertex_idx in enumerate(face):
            vertex_idx = int(vertex_idx)
            if vertex_idx not in vertex_to_face:
                vertex_to_face[vertex_idx] = (face_idx, local_pos)

    entries = []
    for v in skin_vertex_indices:
        if v not in vertex_to_face:
            raise ValueError(f"Skin vertex index {v} not found in {topology_mesh_path} (mesh has {len(mesh.vertices)} vertices)")
        face_idx, local_pos = vertex_to_face[v]
        if local_pos == 0:
            u, w = 1.0, 0.0
        elif local_pos == 1:
            u, w = 0.0, 1.0
        else:
            u, w = 0.0, 0.0
        entries.append([face_idx, u, w])
    return entries
