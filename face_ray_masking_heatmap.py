"""Ava-256 adaptation of wrap_script/face_ray_masking.py's per-frame,
segmentation-driven face mask -- used only for the skin-augmented ("second")
neutral wrap, never the keypoints_3d-only bootstrap wrap (per the user's
explicit instruction: no need on the first wrap; use the first wrap's own
mesh to drive this for the subsequent one).

Three real adaptations from the original, not just a reuse:

1. Per-camera intrinsics. The original computes ONE shared K/focal/principal-
   point for its whole camera batch (valid for Nersemble's fixed rig) --
   Ava-256's ~80 cameras each have their own distinct K, so the PyTorch3D
   camera-batching section below is rewritten to compute focal/principal-
   point per camera instead of reusing camera[0]'s values for everyone.

2. Segmentation. Ported exactly from wrap_script/face_ray_masking_wrapping_
   test.py's per-camera step, module-level `segmenter = FaceSegmentation()` /
   `bg_segmenter = BackgroundSegmentation()`: U2Net's whole-person mask
   (`BackgroundSegmentation.get_mask`) finds the largest person-shaped
   connected component's centroid, which is used as SAM's point-prompt
   (`FaceSegmentation.get_face_mask`); falls back to the image center only
   when U2Net finds no person blob at all -- same as the reference. (An
   earlier version of this module wrongly believed `from model import U2NET`
   had no backing implementation anywhere on disk and fell back to a bare
   SAM-at-image-center prompt; `wrap_script/model/` is actually a package
   (`__init__.py` re-exporting `U2NET` from `u2net.py`), and the checkpoint
   at /scratch/ondemand32/irwinngo/models/u2net_human_seg.pth exists -- the
   only real issue was that wrap_script/segment.py was never actually
   imported the naive way, since FaceView/ (already on sys.path via
   mesh_utils.build_live_extended_pot's own import chain) has a DIFFERENT,
   older `segment.py` that shadows it by import order. `_load_segmentation_
   models()` below fixes that by inserting wrap_script/ at sys.path[0]
   before importing.) Both U2Net and SAM are fed the raw BGR frame directly,
   no BGR->RGB conversion -- matching the reference exactly rather than
   "fixing" what looks like a SAM colorspace mismatch.
   erode_hit_faces() is still duplicated (not imported) -- it's pure
   numpy/set logic with no segmentation dependency, no reason to route
   through the wrap_script/ import-order dance for it.

3. No FRONT_CAMERAS/SIDE_RIGHT_CAMERAS/SIDE_LEFT_CAMERAS convention exists
   for Ava-256 (Nersemble's are a fixed, hand-picked rig-specific set).
   Derived geometrically instead (see classify_front_right_left): the
   bootstrap wrap's own undernose-center landmark (index 61) normal
   approximates "true front", and the right-nostril-minus-left-nostril
   vector (indices 38/19) approximates "true right" -- both already
   meaningful anchors on THIS specific face's own wrapped geometry, not a
   generic rig assumption.

4. Nersemble's own reference-intersection file (wrap_mask.txt, a relative
   path specific to that topology/pipeline) is replaced with
   facescape_mask.txt -- the equivalent already-established face-region
   reference for the exact FLAME topology this pipeline shares with
   FaceScape. IMPORTANT polarity note (user-confirmed against the actual
   FaceForm/Wrap4D SelectPolygons node behavior, template.wrap's FaceMask
   node): a face-index list fed into that node's selection is the set the
   node EXCLUDES, not includes -- the un-listed faces are what actually
   participate. facescape_mask.txt's ~2708 listed indices are therefore an
   exclusion list (eyes/mouth-interior/ear-canal-type detail regions), not
   "the curated face region" -- _reference_hit_faces() below takes the
   complement to recover the actual allowed/included face-region set.
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
# Overridable module-level constants -- portability handoff: a caller on a
# different cluster (no /scratch/ondemand32/irwinngo paths at all) sets these
# directly after import, same monkeypatch pattern run_pipeline.py already
# uses for wrap_script_module.TEMPLATE_PATH/FACE_MASK_PATH. run_pipeline.py
# threads --wrap-script-dir/--facescape-mask-path/--sam-checkpoint-path/
# --u2net-checkpoint-path CLI flags through to these.
WRAP_SCRIPT_DIR = SCRIPT_DIR / "segmentation"
FACESCAPE_MASK_PATH = SCRIPT_DIR / "facescape_mask.txt"
SAM_CHECKPOINT_PATH: Path | None = None  # None -> segment.FaceSegmentation's own hardcoded default
U2NET_CHECKPOINT_PATH: Path | None = None  # None -> segment.BackgroundSegmentation's own hardcoded default
MASK_EROSION_ITERATIONS = 1
MIN_WEIGHT = 8.0
FRONT_ANGLE_THRESHOLD_DEG = 35.0  # camera-to-centroid direction within this of "true front" -> FRONT weight
# Per-face normal-vs-camera angle threshold for compute_face_ray_mask()'s own
# hit-rejection filter (see below) -- deliberately its own constant, NOT
# camera_selection.DEFAULT_ANGLE_THRESHOLD_DEG (80.0), since that threshold
# serves a different purpose (which cameras are usable at all for RAFT/
# CoTracker propagation) and tightening face masking's shouldn't silently
# affect that.
FACE_ANGLE_THRESHOLD_DEG = 180.0  # effectively disabled -- max possible score_camera_for_landmark() is pi rad (180 deg), so this never rejects on angle, only occlusion still applies

_bg_segmenter = None
_face_segmenter = None


def _load_segmentation_models():
    """Lazy, heavy import/load (U2Net + SAM ViT-H onto the GPU) -- only
    triggered when this function is actually called, so --no-skin/bootstrap/
    dry-run invocations never pay this cost. Forces wrap_script/ to the
    FRONT of sys.path before importing `segment` -- FaceView/ (already on
    sys.path via mesh_utils.build_live_extended_pot's own import chain) has
    a different, older `segment.py` that would otherwise shadow this one by
    import order (see module docstring point 2)."""
    global _bg_segmenter, _face_segmenter
    if _bg_segmenter is None or _face_segmenter is None:
        wrap_script_dir_str = str(WRAP_SCRIPT_DIR)
        if wrap_script_dir_str in sys.path:
            sys.path.remove(wrap_script_dir_str)
        sys.path.insert(0, wrap_script_dir_str)
        from segment import FaceSegmentation, BackgroundSegmentation

        _bg_segmenter = BackgroundSegmentation(checkpoint_path=str(U2NET_CHECKPOINT_PATH) if U2NET_CHECKPOINT_PATH else None)
        _face_segmenter = FaceSegmentation(checkpoint_path=str(SAM_CHECKPOINT_PATH) if SAM_CHECKPOINT_PATH else None)
    return _bg_segmenter, _face_segmenter


def _get_face_mask(image_bgr: np.ndarray, point_hints: list[tuple[int, int]] | None = None) -> np.ndarray:
    """Exact port of wrap_script/face_ray_masking_wrapping_test.py's
    per-camera segmentation step, EXCEPT for point_hints (a real addition,
    not in the reference): U2Net's whole-person mask picks a SAM point-
    prompt via the largest person blob's centroid -- for a bust/head+
    shoulders framing that centroid sits somewhere around the chin/neck,
    which lands squarely on a beard for a bearded actor, so SAM segments
    the beard as "the object" rather than facial skin. When point_hints is
    given (real 3D facial landmarks -- see compute_face_ray_mask's own
    nosebridge+chin-tip projection code -- reprojected into this specific
    camera), they're used directly as a multi-point SAM prompt instead,
    skipping U2Net for this camera entirely (also faster) -- multiple points
    (upper + lower face) anchor SAM's mask to span the full face rather than
    clustering around just one point. Falls back to the original U2Net-
    centroid/image-center heuristic when point_hints is empty/None (e.g.
    every candidate landmark projected outside this camera's frame). Both
    stages are fed the raw BGR image directly, no BGR->RGB conversion --
    matching the reference exactly."""
    bg_segmenter, face_segmenter = _load_segmentation_models()

    if point_hints:
        return face_segmenter.get_face_mask(image_bgr, points=point_hints).astype(np.uint8)

    person_mask = (bg_segmenter.get_mask(image_bgr) > 0).astype(np.uint8)
    num_person_labels, _person_labels, person_stats, person_centroids = cv2.connectedComponentsWithStats(
        person_mask, connectivity=8
    )
    if num_person_labels > 1:
        largest_person_label = 1 + np.argmax(person_stats[1:, cv2.CC_STAT_AREA])
        cx, cy = person_centroids[largest_person_label]
        sam_point = (int(round(cx)), int(round(cy)))
    else:
        h, w = image_bgr.shape[:2]
        sam_point = (w // 2, h // 2)

    return face_segmenter.get_face_mask(image_bgr, point=sam_point).astype(np.uint8)


def erode_hit_faces(faces: np.ndarray, hit_faces_set: set[int], iterations: int = 1) -> set[int]:
    """Duplicated from wrap_script/face_ray_masking.py's own function of the
    same name (pure numpy/set logic, no segmentation dependency -- see
    module docstring point 2 for why this isn't imported instead). Shrinks
    hit_faces_set by removing its boundary faces, where a face counts as
    boundary if it shares any vertex with a face outside the set."""
    if not hit_faces_set:
        return set(hit_faces_set)
    faces_np = faces.cpu().numpy() if torch.is_tensor(faces) else np.asarray(faces)
    vertex_to_faces = defaultdict(set)
    for face_idx, (v0, v1, v2) in enumerate(faces_np):
        vertex_to_faces[v0].add(face_idx)
        vertex_to_faces[v1].add(face_idx)
        vertex_to_faces[v2].add(face_idx)
    eroded = set(hit_faces_set)
    for _ in range(iterations):
        boundary_faces = set()
        for face_idx in eroded:
            v0, v1, v2 = faces_np[face_idx]
            neighbors = vertex_to_faces[v0] | vertex_to_faces[v1] | vertex_to_faces[v2]
            neighbors.discard(face_idx)
            if not neighbors.issubset(eroded):
                boundary_faces.add(face_idx)
        eroded -= boundary_faces
    return eroded


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


EAR_STANDARD_INDICES = (1, 2, 4, 6, 9, 14)  # landmark_map.AVA256_STANDARD_INDEX_REGION_GROUPS["ear"]
NOSE_STANDARD_INDICES = (54, 57, 60, 19, 38, 61)  # "nosebridge" + "undernose" groups combined
EAR_NOSE_FORCE_INCLUDE_HOPS = 5  # face-adjacency BFS radius from each landmark's seed face; empirically ~1245 ear + ~828 nose faces (~2073/7800, 26.6%) on this FLAME topology -- adjust if the forced patch looks too small/large


def _force_include_region_faces(
    faces: np.ndarray, pot_rows_by_index: dict[int, list[float]], standard_indices: tuple[int, ...], hops: int
) -> set[int]:
    """Nersemble has no static "ear/nose polygon list" either -- its actual
    mechanism (wrap_script/run_wrap_script.py's ignore_ears=False default) is
    to never let ear/nose landmark correspondence points get dropped, and to
    use the SMALLER wrap_mask.txt exclusion reference (not
    wrap_mask_without_ears.txt) so those regions stay in the ICP-active
    polygon set regardless of what image segmentation found. This is the
    face_ray_masking-polygon-level equivalent: BFS outward `hops`
    face-adjacency steps from each landmark's own template-side POT seed
    face (pot_rows_by_index[idx][0] is that row's face_index), unioning every
    face touched, so the ear/nose region survives even where SAM/U2Net missed
    it entirely -- scale-invariant (face-graph hops, not a world-space
    radius), so it works the same whether verts are in template or
    real-world scale."""
    seed_faces = {int(pot_rows_by_index[idx][0]) for idx in standard_indices if idx in pot_rows_by_index}
    if not seed_faces:
        return set()

    vertex_to_faces = defaultdict(set)
    for face_idx, (v0, v1, v2) in enumerate(faces):
        vertex_to_faces[v0].add(face_idx)
        vertex_to_faces[v1].add(face_idx)
        vertex_to_faces[v2].add(face_idx)

    region = set(seed_faces)
    frontier = set(seed_faces)
    for _ in range(hops):
        next_frontier = set()
        for face_idx in frontier:
            v0, v1, v2 = faces[face_idx]
            next_frontier |= vertex_to_faces[v0] | vertex_to_faces[v1] | vertex_to_faces[v2]
        next_frontier -= region
        region |= next_frontier
        frontier = next_frontier
    return region


def _reference_hit_faces(num_faces: int) -> set[int]:
    """facescape_mask.txt's listed indices are what FaceForm/Wrap4D's
    SelectPolygons node EXCLUDES (see module docstring point 4) -- take the
    complement to get the actual allowed/included face-region set."""
    with FACESCAPE_MASK_PATH.open("r", encoding="utf-8") as f:
        excluded = set(json.load(f)) & set(range(num_faces))
    return set(range(num_faces)) - excluded


def compute_face_ray_mask(
    wrapped_mesh_verts: np.ndarray,
    wrapped_mesh_faces: np.ndarray,
    actor_dir: Path,
    frame_id: str,
    camera_ids: list[str],
    camera_params: dict[str, tuple[np.ndarray, np.ndarray]],
    front_cameras: set[str],
    right_cameras: set[str],
    left_cameras: set[str],
    pot_rows_by_index: dict[int, list[float]] | None = None,
    return_raw_weights: bool = False,
) -> list[int] | dict[int, float]:
    """Returns the final face-index list (already intersected with
    facescape_mask.txt and eroded) to use as the skin-augmented wrap's
    FaceMask restriction. If pot_rows_by_index is given, the ear/nose
    landmark regions are forcefully unioned back in after erosion -- see
    _force_include_region_faces().

    HEATMAP FORK: if return_raw_weights=True, short-circuits right after the
    per-camera vote-accumulation loop and returns the raw face_idx ->
    cumulative_weight dict instead -- the CONTINUOUS signal MIN_WEIGHT/
    facescape_mask.txt/erosion/force-include normally collapse into a single
    binary include/exclude decision. This is what render_face_ray_mask_
    heatmap.py actually visualizes."""
    import camera_utils  # local import
    import mesh_utils  # local import: only needed for the prompt-anchor point_hints below

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_faces = wrapped_mesh_faces.shape[0]
    faces_t = torch.from_numpy(wrapped_mesh_faces).long().to(device)
    verts_t = torch.from_numpy(wrapped_mesh_verts).float().to(device)

    camera_weights = {c: 1.0 for c in front_cameras}
    camera_weights.update({c: 1.0 for c in (right_cameras | left_cameras)})  # was 1.5 -- testing uniform weighting

    # Two SAM prompt anchors: nosebridge (idx 57, upper face) + chin tip
    # (idx 50, keypoints_3d id 7, lower face) -- see face_ray_masking_
    # ava256.py's own copy of this comment for the full rationale.
    prompt_anchor_indices = [idx for idx in (57, 50) if pot_rows_by_index is not None and idx in pot_rows_by_index]
    prompt_anchor_xyzs = {
        idx: mesh_utils.pot_row_world_xyz(wrapped_mesh_verts, wrapped_mesh_faces, pot_rows_by_index[idx])
        for idx in prompt_anchor_indices
    }

    face_to_weight: dict[int, float] = {}
    masks: dict[str, np.ndarray] = {}
    img_hw: dict[str, tuple[int, int]] = {}
    valid_cam_ids: list[str] = []

    for cam_id in camera_ids:
        try:
            image = camera_utils.load_image(actor_dir, cam_id, frame_id)  # BGR, matches the original's own cv2-sourced convention
        except (KeyError, FileNotFoundError):
            continue
        orig_h, orig_w = image.shape[:2]
        seg_h, seg_w = max(1, orig_h // 2), max(1, orig_w // 2)
        seg_input = cv2.resize(image, (seg_w, seg_h), interpolation=cv2.INTER_AREA)

        point_hints: list[tuple[int, int]] = []
        if prompt_anchor_xyzs:
            K_cam, Rt_cam = camera_params[cam_id]
            for anchor_xyz in prompt_anchor_xyzs.values():
                px, py = camera_utils.project_points(anchor_xyz[None, :], K_cam, Rt_cam)[0]
                hint_x, hint_y = int(round(px * seg_w / orig_w)), int(round(py * seg_h / orig_h))
                if 0 <= hint_x < seg_w and 0 <= hint_y < seg_h:
                    point_hints.append((hint_x, hint_y))

        mask = _get_face_mask(seg_input, point_hints=point_hints).astype(np.uint8)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = (labels == largest_label).astype(np.uint8)
        if mask.shape != (orig_h, orig_w):
            mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)

        masks[cam_id] = mask
        img_hw[cam_id] = (orig_h, orig_w)
        valid_cam_ids.append(cam_id)

    if not valid_cam_ids:
        raise RuntimeError(f"No usable camera images for face-ray-masking (capture={actor_dir.name}, frame={frame_id})")

    h, w = img_hw[valid_cam_ids[0]]  # confirmed uniform (1024x667) across Ava-256 cameras
    img_size = torch.tensor([[w, h]] * len(valid_cam_ids), dtype=torch.int, device=device)

    from pytorch3d.renderer import MeshRasterizer, RasterizationSettings, PerspectiveCameras
    from pytorch3d.structures import Meshes

    Rs, Ts, focals, pps = [], [], [], []
    for cam_id in valid_cam_ids:
        K_cam, Rt_cam = camera_params[cam_id]
        K_torch = torch.from_numpy(K_cam).float().to(device)
        extr = torch.from_numpy(Rt_cam).float().to(device)
        R, T = extr[:3, :3], extr[:3, 3]
        R_pytorch3d = R.clone().T
        T_pytorch3d = T.clone()
        R_pytorch3d[:, :2] *= -1
        T_pytorch3d[:2] *= -1

        this_h, this_w = img_hw[cam_id]
        scale = min(this_w, this_h) / 2.0
        c0 = torch.tensor([this_w / 2.0, this_h / 2.0], device=device)
        fl = K_torch[[0, 1], [0, 1]]
        pp = K_torch[[0, 1], [2, 2]]
        focal_pytorch3d = fl / scale
        p0_pytorch3d = -(pp - c0) / scale

        Rs.append(R_pytorch3d)
        Ts.append(T_pytorch3d)
        focals.append(focal_pytorch3d)
        pps.append(p0_pytorch3d)

    cameras = PerspectiveCameras(
        device=device,
        focal_length=torch.stack(focals, dim=0),
        principal_point=torch.stack(pps, dim=0),
        R=torch.stack(Rs, dim=0),
        T=torch.stack(Ts, dim=0),
        image_size=img_size,
    )
    raster_settings = RasterizationSettings(image_size=(h, w), blur_radius=0.0, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    mesh = Meshes(verts=[verts_t], faces=[faces_t])
    meshes = mesh.extend(len(valid_cam_ids))
    fragments = rasterizer(meshes)
    num_faces_per_mesh = meshes.num_faces_per_mesh().cpu().numpy()
    face_offsets = np.concatenate(([0], np.cumsum(num_faces_per_mesh[:-1])))

    # Same angle+occlusion check camera_selection.py uses to decide which
    # cameras are trustworthy for a given LANDMARK's surface normal, applied
    # here per-FACE instead: rasterization's z-buffer already excludes
    # faces truly occluded from a camera, but happily rasterizes faces seen
    # at a near-grazing angle (normal nearly perpendicular to the camera's
    # view direction) -- exactly the marginal, unreliable-segmentation views
    # this rejects. Reuses camera_selection's own functions directly (not
    # reimplemented) so both stay in sync if that logic ever changes.
    import camera_selection

    mesh_tri = trimesh.Trimesh(vertices=wrapped_mesh_verts, faces=wrapped_mesh_faces, process=False)
    face_normals = mesh_tri.face_normals
    face_centers = mesh_tri.triangles_center
    intersector = camera_selection.build_ray_intersector(wrapped_mesh_verts, wrapped_mesh_faces)
    face_angle_threshold_rad = math.radians(FACE_ANGLE_THRESHOLD_DEG)

    for i, cam_id in enumerate(valid_cam_ids):
        mask = masks[cam_id]
        face_idx_map = fragments.pix_to_face[i, ..., 0].cpu().numpy()
        hit_face_idx = face_idx_map[mask > 0]
        hit_face_idx = hit_face_idx[hit_face_idx >= 0]
        hit_face_idx = hit_face_idx - face_offsets[i]
        hit_face_idx = hit_face_idx[(hit_face_idx >= 0) & (hit_face_idx < num_faces)]
        cam_weight = camera_weights.get(cam_id, 1.0)

        _K_cam, Rt_cam = camera_params[cam_id]
        camera_center = camera_utils.camera_center_world(Rt_cam)
        angle_rejected = occlusion_rejected = 0
        for face_idx in np.unique(hit_face_idx):
            face_idx = int(face_idx)
            normal = face_normals[face_idx]
            if camera_selection.score_camera_for_landmark(normal, Rt_cam) > face_angle_threshold_rad:
                angle_rejected += 1
                continue
            if camera_selection.is_occluded(intersector, face_centers[face_idx], normal, camera_center):
                occlusion_rejected += 1
                continue
            face_to_weight[face_idx] = face_to_weight.get(face_idx, 0.0) + cam_weight
        print(f"  face_ray_masking cam={cam_id} weight={cam_weight} cumulative_faces_hit={len(face_to_weight)} "
              f"(angle_rejected={angle_rejected} occlusion_rejected={occlusion_rejected})")

    if return_raw_weights:
        print(f"Returning raw per-face weights for {len(face_to_weight)}/{num_faces} faces "
              f"(max weight={max(face_to_weight.values()) if face_to_weight else 0.0}) -- heatmap mode")
        return face_to_weight

    hit_faces = {face_idx for face_idx, weight in face_to_weight.items() if weight >= MIN_WEIGHT}
    print(f"Faces with cumulative weight >= {MIN_WEIGHT}: {len(hit_faces)} out of {num_faces}")

    reference_hit_faces = _reference_hit_faces(num_faces)
    hit_faces = hit_faces & reference_hit_faces
    print(f"Faces after intersecting with facescape_mask.txt: {len(hit_faces)}")

    hit_faces = erode_hit_faces(wrapped_mesh_faces, hit_faces, iterations=MASK_EROSION_ITERATIONS)
    print(f"Faces after eroding mask boundary ({MASK_EROSION_ITERATIONS} iteration(s)): {len(hit_faces)}")

    if pot_rows_by_index is not None:
        ear_faces = _force_include_region_faces(wrapped_mesh_faces, pot_rows_by_index, EAR_STANDARD_INDICES, EAR_NOSE_FORCE_INCLUDE_HOPS)
        nose_faces = _force_include_region_faces(wrapped_mesh_faces, pot_rows_by_index, NOSE_STANDARD_INDICES, EAR_NOSE_FORCE_INCLUDE_HOPS)
        forced = (ear_faces | nose_faces) - hit_faces
        hit_faces |= forced
        print(f"Faces after forcefully including ear/nose regions: {len(hit_faces)} (+{len(forced)} forced)")

    return sorted(hit_faces)
