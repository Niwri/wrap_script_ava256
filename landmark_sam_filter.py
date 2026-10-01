#!/usr/bin/env python3
"""SAM landmark filter for Ava-256 -- the same rule as Nersemble's
wrap_script/scan_face_masking.py. Computed on the neutral frame
(run_neutral_skin_propagation.ensure_capture_face_mask -> landmark_filter.json,
used for the neutral frame's own wrap) and per expression / segment
(ensure_segment_landmark_filters -> landmark_filter_<seg_id>.json, used for
every other frame of that segment; see the per-segment section below). The
rule, for one frame:

  - Every used landmark (landmark_map.used_landmark_indices: the standard
    keypoints_3d-sourced ones AND the skin ones) is checked in 2D against the
    front cameras' SAM face masks (the same nosebridge-prompted pass the face
    masks use).
  - A camera only votes when the landmark is on-image and NOT occluded: the
    scan's (kinematic_tracking mesh) depth at the landmark's pixel must not be
    more than OCCLUSION_TOLERANCE_MM in front of it.
  - It votes "inside" when the landmark is inside the SAM mask, or within
    MASK_MARGIN_MM of it (converted to pixels per landmark: fx * mm / depth).
    MASK_MARGIN_MM is 0: no boundary allowance, unlike Nersemble's 30 px.
  - Kept when at least MIN_AGREEING_CAMERAS (1) of the voting cameras have
    it inside their SAM mask (the "> AGREEMENT_FRACTION of the voting
    cameras" alternative is then implied); dropped when no voting camera has
    it inside -- including when it's occluded / off-image in every camera.
  - NEVER_FILTERED_INDICES (eyelids + lips, nosebridge, undernose) are always
    kept; ears and everything else are filtered.

Output: <output>/<capture>/<neutral>/landmark_filter.json
  {"dropped": [...], "kept": [...], "votes": {index: {...}}, "rule": {...}}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import landmark_map

LANDMARK_FILTER_NAME = "landmark_filter.json"
MASK_MARGIN_MM = 0.0  # no boundary allowance (was 4.5 mm)
OCCLUSION_TOLERANCE_MM = 10.0
AGREEMENT_FRACTION = 0.5
MIN_AGREEING_CAMERAS = 1  # one camera seeing it inside its SAM mask is enough (was 3)
_groups = landmark_map.AVA256_STANDARD_INDEX_REGION_GROUPS
NEVER_FILTERED_INDICES = frozenset(_groups["eyelid_lip"]) | frozenset(_groups["nosebridge"]) | frozenset(_groups["undernose"])


def load_dropped_indices(output_dir: Path) -> list[int] | None:
    """Dropped landmark indices from <output_dir>/landmark_filter.json (the
    neutral frame's), or None when it doesn't exist yet."""
    path = Path(output_dir) / LANDMARK_FILTER_NAME
    if not path.exists():
        return None
    return sorted(int(i) for i in json.loads(path.read_text(encoding="utf-8"))["dropped"])


# --- per-expression (segment) filters -----------------------------------------
# The neutral frame's landmark_filter.json is used for the neutral frame's own
# wrap only; every other frame uses its segment's filter,
# <neutral dir>/landmark_filter_<seg_id>.json, computed by
# run_neutral_skin_propagation.ensure_segment_landmark_filters() on the
# segment's FIRST frame (the first listed frame with propagated landmarks,
# neutral frame excluded) -- that frame's dropped indices apply to the whole
# segment. Standard landmarks are additionally gated per frame by keypoints_3d
# in run_pipeline.py.


def segment_filter_path(output_dir: Path, seg_id: str) -> Path:
    return Path(output_dir) / f"landmark_filter_{seg_id}.json"


def segment_face_mask_path(output_dir: Path, seg_id: str) -> Path:
    """The segment's FLAME face mask (EXCLUDED faces, what Wrap's FaceMask node
    reads), next to its _included counterpart -- see
    run_neutral_skin_propagation.ensure_segment_masks()."""
    return Path(output_dir) / f"face_ray_mask_{seg_id}.json"


def load_segment_dropped_indices(output_dir: Path, seg_id: str) -> list[int] | None:
    """Dropped indices of segment seg_id's filter, or None when it doesn't exist."""
    path = segment_filter_path(output_dir, seg_id)
    if not path.exists():
        return None
    return sorted(int(i) for i in json.loads(path.read_text(encoding="utf-8"))["dropped"])


def segment_of_frame(actor_dir: Path, frame_id: str) -> str | None:
    """seg_id of frame_id in the capture's frame_list.csv (None if unlisted)."""
    import csv

    with (Path(actor_dir) / "decoder" / "frame_list.csv").open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row["frame_id"].strip().zfill(6) == frame_id:
                return row["seg_id"]
    return None


_glctx = None


def _depth_map(verts: np.ndarray, faces: np.ndarray, K: np.ndarray, Rt: np.ndarray, width: int, height: int) -> np.ndarray:
    """Camera-space depth of the visible surface per pixel (inf = background),
    depth-tested, row 0 = top (nvdiffrast)."""
    import nvdiffrast.torch as dr
    import torch

    global _glctx
    if _glctx is None:
        _glctx = dr.RasterizeCudaContext()
    cam = verts @ Rt[:3, :3].T + Rt[:3, 3]
    z = cam[:, 2]
    x_ndc = (cam[:, 0] / z * K[0, 0] + K[0, 2]) / width * 2.0 - 1.0
    y_ndc = (cam[:, 1] / z * K[1, 1] + K[1, 2]) / height * 2.0 - 1.0
    near, far = 10.0, 1e5
    clip = np.stack([x_ndc * z, y_ndc * z, (far + near) / (far - near) * z - 2 * far * near / (far - near), z], axis=1)
    pos = torch.from_numpy(clip).float().cuda()[None].contiguous()
    tri = torch.from_numpy(faces.astype(np.int32)).cuda().contiguous()
    rast, _ = dr.rasterize(_glctx, pos, tri, resolution=[height, width])
    depth, _ = dr.interpolate(torch.from_numpy(z[:, None]).float().cuda()[None].contiguous(), rast, tri)
    d = depth[0, ..., 0].cpu().numpy()
    d[rast[0, ..., 3].cpu().numpy() == 0] = np.inf
    return d


def landmark_votes(targets: dict[int, np.ndarray], sam_masks: dict[str, np.ndarray],
                   camera_params: dict[str, tuple[np.ndarray, np.ndarray]],
                   scan_verts: np.ndarray, scan_faces: np.ndarray) -> dict[int, dict[str, list[str]]]:
    """{index: {"inside"|"outside"|"occluded"|"off_image": [camera ids]}}."""
    import cv2

    votes = {i: {"inside": [], "outside": [], "occluded": [], "off_image": []} for i in targets}
    indices = sorted(targets)
    pts = np.stack([targets[i] for i in indices])
    for cam, mask in sam_masks.items():
        K, Rt = camera_params[cam]
        h, w = mask.shape
        depth = _depth_map(scan_verts, scan_faces, K, Rt, w, h)
        dist_to_mask = cv2.distanceTransform((mask == 0).astype(np.uint8), cv2.DIST_L2, 5)  # 0 inside
        cam_pts = pts @ Rt[:3, :3].T + Rt[:3, 3]
        for i, (x, y, z) in zip(indices, cam_pts):
            if z <= 0:
                votes[i]["off_image"].append(cam)
                continue
            u, v = x / z * K[0, 0] + K[0, 2], y / z * K[1, 1] + K[1, 2]
            ui, vi = int(round(u)), int(round(v))
            if not (0 <= ui < w and 0 <= vi < h):
                votes[i]["off_image"].append(cam)
            elif depth[vi, ui] < z - OCCLUSION_TOLERANCE_MM:
                votes[i]["occluded"].append(cam)
            elif dist_to_mask[vi, ui] <= MASK_MARGIN_MM * K[0, 0] / z:
                votes[i]["inside"].append(cam)
            else:
                votes[i]["outside"].append(cam)
    return votes


def passes(v: dict[str, list[str]]) -> bool:
    n_in, n_seen = len(v["inside"]), len(v["inside"]) + len(v["outside"])
    if n_seen == 0:
        return False  # occluded / off-image in every camera (e.g. an ear behind hair)
    return n_in / n_seen > AGREEMENT_FRACTION or n_in >= MIN_AGREEING_CAMERAS


def compute_landmark_filter(targets: dict[int, np.ndarray], sam_masks: dict[str, np.ndarray],
                            camera_params: dict[str, tuple[np.ndarray, np.ndarray]],
                            scan_verts: np.ndarray, scan_faces: np.ndarray) -> dict:
    """The landmark_filter.json payload for these world-space targets."""
    votes = landmark_votes(targets, sam_masks, camera_params, scan_verts, scan_faces)
    kept = sorted(i for i in targets if passes(votes[i]) or i in NEVER_FILTERED_INDICES)
    dropped = sorted(set(targets) - set(kept))
    return {
        "kept": kept, "dropped": dropped, "votes": {str(i): v for i, v in votes.items()},
        "never_filtered": sorted(NEVER_FILTERED_INDICES), "cameras": sorted(sam_masks),
        "rule": {"mask_margin_mm": MASK_MARGIN_MM, "occlusion_tolerance_mm": OCCLUSION_TOLERANCE_MM,
                 "agreement_fraction": AGREEMENT_FRACTION, "min_agreeing_cameras": MIN_AGREEING_CAMERAS},
    }
