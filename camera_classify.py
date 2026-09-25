"""Torch-free camera front/right/left classification for Ava-256 captures.

classify_front_right_left() used to live only in face_ray_masking_ava256.py,
which imports torch (and, lazily, pytorch3d/SAM). The CPU-only wrap side --
run_pipeline_batch.py's post-capture videos and point_overlay.py -- only needs
this numpy function, so it lives here; face_ray_masking_ava256 re-exports it.
"""
from __future__ import annotations

import numpy as np

FRONT_ANGLE_THRESHOLD_DEG = 35.0  # camera-to-centroid direction within this of "true front" -> FRONT weight


def classify_front_right_left(
    wrapped_verts: np.ndarray,
    wrapped_faces: np.ndarray,
    pot_rows_by_index: dict[int, list[float]],
    camera_ids: list[str],
    camera_params: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[set[str], set[str], set[str]]:
    """Geometric front/right/left split, derived from the bootstrap wrap's
    own landmark geometry (see module docstring point 2) rather than a
    fixed rig convention. Only affects the 1.0-vs-1.5 camera vote weight in
    face_ray_masking() -- every camera with a usable image still
    participates in the vote regardless of which bucket it lands in."""
    import mesh_utils  # local import: avoid a hard dependency at module load

    undernose_xyz, undernose_normal = (
        mesh_utils.pot_row_world_xyz(wrapped_verts, wrapped_faces, pot_rows_by_index[61]),
        mesh_utils.pot_row_face_normal(wrapped_verts, wrapped_faces, pot_rows_by_index[61]),
    )
    # canonical_front_dir must match `rel` below (centroid->camera, i.e. where
    # a camera SITS relative to the face), not a camera's view/look direction
    # (which would be the opposite). A genuine front camera sits further out
    # along the face's own outward surface normal at the nose -- so this is
    # the outward normal itself, unnegated. (Bug found via a real render:
    # the negated version put "front" camera picks on the far side of the
    # head, at ~the same +Z the back-of-skull bulk sits at, for CDR970 --
    # confirmed by computing camera_center - centroid manually and comparing
    # against the mesh's own bbox extent in the same axis.)
    canonical_front_dir = undernose_normal

    right_nostril = mesh_utils.pot_row_world_xyz(wrapped_verts, wrapped_faces, pot_rows_by_index[38])
    left_nostril = mesh_utils.pot_row_world_xyz(wrapped_verts, wrapped_faces, pot_rows_by_index[19])
    right_axis = right_nostril - left_nostril
    right_axis = right_axis / np.linalg.norm(right_axis)

    centroid = wrapped_verts.mean(axis=0)
    front_threshold = np.cos(np.radians(FRONT_ANGLE_THRESHOLD_DEG))

    import camera_utils  # local import, same reasoning as mesh_utils above

    front, right, left = set(), set(), set()
    for cam_id in camera_ids:
        _K, Rt = camera_params[cam_id]
        cam_center = camera_utils.camera_center_world(Rt)
        rel = cam_center - centroid
        rel = rel / np.linalg.norm(rel)
        front_component = float(np.dot(rel, canonical_front_dir))
        if front_component >= front_threshold:
            front.add(cam_id)
        elif np.dot(rel, right_axis) > 0:
            right.add(cam_id)
        else:
            left.add(cam_id)
    return front, right, left
