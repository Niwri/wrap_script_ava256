#!/usr/bin/env python3
"""Face masking on any mesh, sharing one SAM pass -- used for the Ava-256 SCAN
(kinematic_tracking mesh, world space) alongside
face_ray_masking_ava256.compute_face_ray_mask()'s mask on the neutral wrap.

Same segmentation and voting as face_ray_masking_ava256:
  - compute_sam_masks(): per camera, SAM segments the face with the nosebridge
    point reprojected into that camera as the point prompt
    (face_ray_masking_ava256._get_face_mask, half-resolution input, largest
    connected component), returned at full resolution. The result can be
    passed to compute_face_ray_mask(precomputed_masks=...) so the wrap mask
    and the scan mask come from the same SAM pass.
  - vote_faces(): the mesh is rasterized into every given camera (pytorch3d;
    the caller passes only the FRONT cameras); for each face, counts the
    cameras that see it (not occluded) and those whose SAM mask covers it;
    faces covered for a strict majority (> AGREEMENT_FRACTION, 50%) of the
    cameras seeing them, or by at least MIN_AGREEING_CAMERAS (3) of them, are
    kept, then the boundary is eroded
    MASK_EROSION_ITERATIONS (1) time(s). Unlike
    the wrap mask there's no intersection with facescape_mask.txt -- that list
    is in FLAME template face indices and has no meaning on the scan's own
    topology.

The scan mask is used by run_pipeline.py to drop skin correspondence landmarks
whose target lands on a scan face outside it (e.g. hair where the ears are).
kinematic_tracking has one topology for every frame of a capture, so the mask
computed on the neutral frame's scan applies to every frame.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

import camera_selection
import camera_utils
import face_ray_masking_ava256 as frm

# SAM's positive point prompts, as landmark (POT row) indices on the frame's
# wrap: nosebridge (57), outer eyebrow ends (30 left, 47 right), lip corners
# (18 left, 37 right) and ears (1, 4, 6, 14 -- the ear points with a
# keypoints_3d source; 2/9 have none). A single nosebridge point is ambiguous
# -- on some actors (e.g. 20220307--1342--RGA575) SAM returns just the nose --
# so the extra points anchor the mask to the whole face.
SAM_PROMPT_INDICES = (57, 30, 47, 18, 37, 1, 4, 6, 14)
# Eyebrow and ear prompts are only used when the frame's keypoints_3d has that
# landmark's keypoint (a missing ear is usually hidden by hair or turned away).
KEYPOINTS_3D_GATED_PROMPT_INDICES = frozenset((30, 47, 1, 4, 6, 14))


def sam_prompt_indices(keypoints_3d_ids=None) -> list[int]:
    """SAM_PROMPT_INDICES minus the gated (eyebrow / ear) ones whose keypoint
    is not in keypoints_3d_ids (the frame's present keypoints_3d ids); no
    gating when keypoints_3d_ids is None."""
    import landmark_map

    if keypoints_3d_ids is None:
        return list(SAM_PROMPT_INDICES)
    ids = set(keypoints_3d_ids)
    return [i for i in SAM_PROMPT_INDICES
            if i not in KEYPOINTS_3D_GATED_PROMPT_INDICES or landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID.get(i) in ids]


def sam_prompt_xyz(verts_world: np.ndarray, faces: np.ndarray, pot_rows_by_index: dict,
                   keypoints_3d_ids=None) -> np.ndarray:
    """(N, 3) world positions of the SAM prompt landmarks on a wrap
    (sam_prompt_indices(keypoints_3d_ids))."""
    import mesh_utils

    return np.stack([mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows_by_index[i])
                     for i in sam_prompt_indices(keypoints_3d_ids)])


AGREEMENT_FRACTION = frm.AGREEMENT_FRACTION
MIN_AGREEING_CAMERAS = frm.MIN_AGREEING_CAMERAS
MASK_EROSION_ITERATIONS = frm.MASK_EROSION_ITERATIONS


def compute_sam_masks(
    actor_dir: Path,
    frame_id: str,
    camera_ids: list[str],
    camera_params: dict[str, tuple[np.ndarray, np.ndarray]],
    anchor_xyz: np.ndarray,
) -> dict[str, np.ndarray]:
    """camera_id -> full-resolution 0/1 SAM face mask (cameras without an image
    for this frame are left out). anchor_xyz: world-space SAM point prompt(s),
    one (3,) point or an (N, 3) array -- the pipeline passes sam_prompt_xyz()
    (nosebridge, eyebrow ends, lip corners). Every point that projects inside
    a camera's image is a positive prompt for that camera."""
    masks: dict[str, np.ndarray] = {}
    for cam_id in camera_ids:
        try:
            image = camera_utils.load_image(actor_dir, cam_id, frame_id)
        except (KeyError, FileNotFoundError):
            continue
        h, w = image.shape[:2]
        seg_h, seg_w = max(1, h // 2), max(1, w // 2)
        seg_input = cv2.resize(image, (seg_w, seg_h), interpolation=cv2.INTER_AREA)
        K, Rt = camera_params[cam_id]
        anchors = np.asarray(anchor_xyz, dtype=np.float64).reshape(-1, 3)
        hints = [(int(round(px * seg_w / w)), int(round(py * seg_h / h)))
                 for px, py in camera_utils.project_points(anchors, K, Rt)]
        hints = [p for p in hints if 0 <= p[0] < seg_w and 0 <= p[1] < seg_h]
        mask = frm._get_face_mask(seg_input, point_hints=hints).astype(np.uint8)
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n_labels > 1:
            mask = (labels == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
        if mask.shape != (h, w):
            mask = (cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        masks[cam_id] = mask
    if not masks:
        raise RuntimeError(f"No usable camera images ({Path(actor_dir).name}/{frame_id})")
    return masks


def vote_faces(
    verts_world: np.ndarray,
    faces: np.ndarray,
    masks: dict[str, np.ndarray],
    camera_params: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    agreement: float = AGREEMENT_FRACTION,
    min_votes: int = MIN_AGREEING_CAMERAS,
    erosion_iterations: int = MASK_EROSION_ITERATIONS,
    weights_out: dict | None = None,
) -> list[int]:
    """Faces of (verts_world, faces) inside the SAM mask of a strict majority
    (> agreement), or of at least min_votes, of the cameras in `masks` that
    see them (angle + occlusion checked), boundary eroded. The caller passes only the front cameras'
    masks. weights_out gets face -> (votes, cameras seeing it)."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_faces = faces.shape[0]
    cams = list(masks)

    from pytorch3d.renderer import MeshRasterizer, PerspectiveCameras, RasterizationSettings
    from pytorch3d.structures import Meshes

    h, w = masks[cams[0]].shape[:2]
    Rs, Ts, focals, pps = [], [], [], []
    for cam_id in cams:  # same pytorch3d camera conversion as face_ray_masking_ava256
        K, Rt = camera_params[cam_id]
        K_t = torch.from_numpy(K).float().to(device)
        extr = torch.from_numpy(Rt).float().to(device)
        R, T = extr[:3, :3].clone().T, extr[:3, 3].clone()
        R[:, :2] *= -1
        T[:2] *= -1
        ch, cw = masks[cam_id].shape[:2]
        scale = min(cw, ch) / 2.0
        c0 = torch.tensor([cw / 2.0, ch / 2.0], device=device)
        Rs.append(R)
        Ts.append(T)
        focals.append(K_t[[0, 1], [0, 1]] / scale)
        pps.append(-(K_t[[0, 1], [2, 2]] - c0) / scale)
    cameras = PerspectiveCameras(
        device=device, focal_length=torch.stack(focals), principal_point=torch.stack(pps),
        R=torch.stack(Rs), T=torch.stack(Ts), image_size=torch.tensor([[w, h]] * len(cams), dtype=torch.int, device=device),
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=RasterizationSettings(image_size=(h, w), blur_radius=0.0, faces_per_pixel=1))
    mesh = Meshes(verts=[torch.from_numpy(verts_world).float().to(device)],
                  faces=[torch.from_numpy(faces).long().to(device)]).extend(len(cams))
    fragments = rasterizer(mesh)
    offsets = np.concatenate(([0], np.cumsum(mesh.num_faces_per_mesh().cpu().numpy()[:-1])))

    tri = trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)
    normals, centers = tri.face_normals, tri.triangles_center
    intersector = camera_selection.build_ray_intersector(verts_world, faces)
    angle_limit = math.radians(frm.FACE_ANGLE_THRESHOLD_DEG)

    weight: dict[int, float] = {}
    seen: dict[int, int] = {}
    for i, cam_id in enumerate(cams):
        pix = fragments.pix_to_face[i, ..., 0].cpu().numpy() - offsets[i]
        in_range = (pix >= 0) & (pix < num_faces)
        inside = set(np.unique(pix[in_range & (masks[cam_id] > 0)]).tolist())
        _, Rt = camera_params[cam_id]
        center = camera_utils.camera_center_world(Rt)
        for f in np.unique(pix[in_range]):
            f = int(f)
            if camera_selection.score_camera_for_landmark(normals[f], Rt) > angle_limit:
                continue
            if camera_selection.is_occluded(intersector, centers[f], normals[f], center):
                continue
            seen[f] = seen.get(f, 0) + 1
            if f in inside:
                weight[f] = weight.get(f, 0.0) + 1.0
    if weights_out is not None:
        weights_out.update({f: (weight.get(f, 0.0), n) for f, n in seen.items()})

    hit_faces = {f for f, n in seen.items() if frm.passes_vote(weight.get(f, 0.0), n, agreement, min_votes)}
    print(f"faces inside the SAM mask of > {agreement:.0%} (or >= {min_votes}) of the cameras that see them: "
          f"{len(hit_faces)} / {num_faces} ({len(cams)} cameras)")
    hit_faces = frm.erode_hit_faces(faces, hit_faces, iterations=erosion_iterations)
    print(f"after eroding the boundary ({erosion_iterations}x): {len(hit_faces)}")
    return sorted(hit_faces)


def compute_scan_face_mask(
    scan_verts_world: np.ndarray,
    scan_faces: np.ndarray,
    actor_dir: Path,
    frame_id: str,
    camera_ids: list[str],
    camera_params: dict[str, tuple[np.ndarray, np.ndarray]],
    nosebridge_xyz: np.ndarray,
    *,
    agreement: float = AGREEMENT_FRACTION,
    min_votes: int = MIN_AGREEING_CAMERAS,
    erosion_iterations: int = MASK_EROSION_ITERATIONS,
    sam_masks_out: dict | None = None,
    weights_out: dict | None = None,
) -> list[int]:
    """SAM pass + vote on the scan in one call (included scan face indices).
    camera_ids should be the front cameras only."""
    masks = compute_sam_masks(actor_dir, frame_id, camera_ids, camera_params, nosebridge_xyz)
    if sam_masks_out is not None:
        sam_masks_out.update(masks)
    return vote_faces(scan_verts_world, scan_faces, masks, camera_params,
                      agreement=agreement, min_votes=min_votes, erosion_iterations=erosion_iterations,
                      weights_out=weights_out)


def submesh(verts: np.ndarray, faces: np.ndarray, keep_faces) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Only the kept faces and the vertices they use, reindexed.
    Returns (verts, faces, old vertex index for each new vertex)."""
    kept = faces[np.asarray(sorted(keep_faces), dtype=np.int64)]
    used = np.unique(kept)
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[kept], used
