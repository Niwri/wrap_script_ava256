#!/usr/bin/env python3
"""Ava-256 counterpart to wrap_script_facescape/run_neutral_skin_propagation.py
-- propagates every landmark (0-106, minus 2/9) from a capture's neutral
frame to every other frame in that capture, via RAFT/CoTracker, using
per-landmark automatic camera selection (camera_selection.py) instead of
FaceScape's hardcoded per-group camera lists.

decoder/frame_list.csv groups a capture's frames into temporally-disjoint
segments (different sentences/expressions). Division of labor: within the
neutral frame's own segment, genuine CoTracker multi-frame video tracking
(query at the neutral frame's actual position in that segment's clip, not
forced to t=0). For every other segment (no continuous motion path from the
neutral frame), one RAFT dense-flow hop from the neutral image to that
segment's first frame seeds a pixel position, then CoTracker chains the rest
of that segment from there.

On success, writes the capture's own label_tracker status unlabeled ->
"unreviewed" directly (no prior explicit "unlabeled" write -- the implicit
default already satisfies that transition's guard).

Usage:
    python3 run_neutral_skin_propagation.py <CAPTURE_ID> [--ava256-data-root ...]
        [--min-cameras-per-landmark 5] [--angle-threshold-deg 80.0]
        [--segments SEG_ID ...] [--max-frames-per-segment N] [--dry-run]
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import landmark_map
import mesh_utils
import camera_utils
import camera_selection
import dynamic_transform
from label_tracker_ava256 import Ava256LabelTracker, ensure_label_tracker_file
from run_pipeline import run_pipeline as wrap_pipeline, DEFAULTS as PIPELINE_DEFAULTS, _load_wrap_script


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


from torchvision.models.optical_flow import raft_large, Raft_Large_Weights  # noqa: E402
# ^ a normally-installed package import, not FaceView/co-tracker-path-dependent
# -- unlike triangulate.py/cotracker.predictor below, safe at module scope.

RAFT_WEIGHTS = Raft_Large_Weights.DEFAULT
RAFT_DOWNSCALE = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NEUTRAL_SEG_ID = "EXP_neutral_peak"

DEFAULTS = dict(PIPELINE_DEFAULTS)  # ava256_data_root/landmark_root/output_root/label_tracker_path/
                                     # ava256_mesh_topology_path/wrap_script_path/template_pot_path
# Local, self-contained copy (stripped of the get_settings/backend.app.core.config
# dependency the original wrap_script/triangulate.py only needed for its own
# unused keypoint_to_3d()/main() -- see triangulate.py's own module docstring).
# No longer needs FaceView/ on sys.path at all as a result -- that was the
# only reason a faceview_root/FACEVIEW_ROOT dependency existed here.
DEFAULTS["triangulate_script_path"] = SCRIPT_DIR / "triangulate.py"
# Local git clone (see HANDOFF.md) rather than reaching into FaceView/co-tracker/
# -- self-containment, same as every other consolidated dependency in this
# directory. Checkpoint lives under the shared models/ root instead of inside
# the clone itself (matches SAM/U2Net/RAFT's own convention: code goes in the
# repo, weights get handed off separately via scp -- see HANDOFF.md).
DEFAULTS["cotracker_root"] = SCRIPT_DIR / "co-tracker"
DEFAULTS["cotracker_checkpoint_path"] = Path("/scratch/ondemand32/irwinngo/models/scaled_offline.pth")
DEFAULTS["raft_checkpoint_path"] = Path("/scratch/ondemand32/irwinngo/models/raft_large_C_T_SKHT_V2-ff5fadd5.pth")


# --- lazy, path-configurable imports -----------------------------------------
#
# cotracker.predictor needs cotracker_root on sys.path BEFORE it's importable,
# for its own internal `from cotracker.models... import ...`. Since it's a
# CLI flag (not a fixed module-level constant), this can't be a plain
# top-of-file import (which would run before argparse parses that flag) --
# loaded lazily instead, the first time it's actually needed. triangulate.py
# no longer needs this treatment at all (see its own DEFAULTS comment above).

def load_triangulate_point(triangulate_script_path: Path):
    module = _load_module("wrap_script_triangulate", Path(triangulate_script_path))
    return module.triangulate_point


def load_cotracker_model(cotracker_root: Path, checkpoint_path: Path):
    cotracker_root = str(cotracker_root)
    if cotracker_root not in sys.path:
        sys.path.append(cotracker_root)
    from cotracker.predictor import CoTrackerPredictor
    model = CoTrackerPredictor(checkpoint=str(checkpoint_path))
    return model.to(DEVICE)


def load_raft_model(checkpoint_path: Path):
    model = raft_large(weights=None)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model = model.to(DEVICE).eval()
    return model


# --- frame_list.csv / segments -----------------------------------------------

def read_segments(actor_dir: Path) -> dict[str, list[str]]:
    """seg_id -> ordered list of zero-padded 6-digit frame_ids, in
    frame_list.csv's own row order (already ascending within a segment)."""
    path = actor_dir / "decoder" / "frame_list.csv"
    segments: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            seg_id = row["seg_id"]
            frame_id = row["frame_id"].strip().zfill(6)
            segments.setdefault(seg_id, []).append(frame_id)
    return segments


# --- RAFT / CoTracker tracking (adapted from wrap_script_facescape's own) ---

def _raft_flow(model, img0: np.ndarray, img1: np.ndarray) -> np.ndarray:
    h, w = img0.shape[:2]
    t0 = torch.from_numpy(img0).permute(2, 0, 1)[None].to(DEVICE)
    t1 = torch.from_numpy(img1).permute(2, 0, 1)[None].to(DEVICE)
    t0, t1 = RAFT_WEIGHTS.transforms()(t0, t1)
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h or pad_w:
        t0 = torch.nn.functional.pad(t0, (0, pad_w, 0, pad_h), mode="replicate")
        t1 = torch.nn.functional.pad(t1, (0, pad_w, 0, pad_h), mode="replicate")
    with torch.no_grad():
        flow_predictions = model(t0, t1)
    flow = flow_predictions[-1][0, :, :h, :w]
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    return flow.detach().cpu().numpy()


def _sample_flow_bilinear(flow: np.ndarray, x: float, y: float) -> tuple[float, float]:
    h, w = flow.shape[1], flow.shape[2]
    x = min(max(x, 0.0), w - 1.0)
    y = min(max(y, 0.0), h - 1.0)
    x0, y0 = int(x), int(y)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    wx, wy = x - x0, y - y0
    top = flow[:, y0, x0] * (1 - wx) + flow[:, y0, x1] * wx
    bot = flow[:, y1, x0] * (1 - wx) + flow[:, y1, x1] * wx
    dx, dy = top * (1 - wy) + bot * wy
    return float(dx), float(dy)


def raft_hop(model, img0: np.ndarray, img1: np.ndarray, seeds: dict[int, tuple[float, float]]) -> dict[int, tuple[np.ndarray, float]]:
    """Single dense-flow hop from img0 to img1, sampled at each seed's img0
    position. Visibility always 1.0 (RAFT has no occlusion signal)."""
    if not seeds:
        return {}
    orig_h, orig_w = img0.shape[:2]
    small_w = max(8, round(orig_w * RAFT_DOWNSCALE))
    small_h = max(8, round(orig_h * RAFT_DOWNSCALE))
    scale_x, scale_y = small_w / orig_w, small_h / orig_h
    img0_small = np.array(Image.fromarray(img0).resize((small_w, small_h), Image.BILINEAR))
    img1_small = np.array(Image.fromarray(img1).resize((small_w, small_h), Image.BILINEAR))
    flow = _raft_flow(model, img0_small, img1_small)
    result: dict[int, tuple[np.ndarray, float]] = {}
    for idx, (x, y) in seeds.items():
        dx_small, dy_small = _sample_flow_bilinear(flow, x * scale_x, y * scale_y)
        dx, dy = dx_small / scale_x, dy_small / scale_y
        result[idx] = (np.array([x + dx, y + dy], dtype=np.float64), 1.0)
    return result


def cotracker_track_clip(
    model: CoTrackerPredictor,
    frames: list[np.ndarray],
    query_t: int,
    seeds: dict[int, tuple[float, float]],
) -> dict[int, dict[int, tuple[np.ndarray, float]]]:
    """Tracks every seeded point (at frame index query_t) across the WHOLE
    clip (bidirectional -- CoTracker's scaled_offline checkpoint supports an
    arbitrary query frame, not just t=0). Returns
    {frame_index: {landmark_idx: (xy, visibility)}} for every frame in the
    clip (including query_t itself, trivially ~equal to the seed)."""
    if not seeds or not frames:
        return {}
    h = min(f.shape[0] for f in frames)
    w = min(f.shape[1] for f in frames)
    frames = [f[:h, :w] for f in frames]
    video = np.stack(frames, axis=0)
    video_tensor = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float().to(DEVICE)
    global_indices = list(seeds.keys())
    query_tensor = torch.tensor(
        [[float(query_t), seeds[idx][0], seeds[idx][1]] for idx in global_indices],
        dtype=torch.float32, device=DEVICE,
    )[None]
    with torch.no_grad():
        # backward_tracking=True: the neutral frame's own segment query_t can
        # sit anywhere in that segment's clip (wherever the neutral frame
        # falls in frame_list.csv order), so frames BEFORE query_t need
        # backward tracking too -- CoTrackerPredictor.forward() defaults to
        # forward-only (False), which would silently drop every frame before
        # query_t otherwise. A no-op cost for the query_t=0 (RAFT-seeded
        # other-segment) case, so left on unconditionally rather than
        # threading a second parameter through every caller.
        pred_tracks, pred_visibility = model(video_tensor, queries=query_tensor, backward_tracking=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    tracks = pred_tracks[0].detach().cpu().numpy()  # (T, N, 2)
    visibility = pred_visibility[0].detach().cpu().numpy()  # (T, N)

    result: dict[int, dict[int, tuple[np.ndarray, float]]] = {}
    for t in range(tracks.shape[0]):
        result[t] = {
            global_idx: (tracks[t, i], float(visibility[t, i]))
            for i, global_idx in enumerate(global_indices)
        }
    return result


# --- locking -----------------------------------------------------------------

@contextlib.contextmanager
def _capture_propagation_lock(landmark_root: Path, capture_id: str):
    lock_dir = landmark_root / capture_id
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_dir / ".propagation.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        yield False
        return
    try:
        yield True
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


# --- output --------------------------------------------------------------------

def group_for_index(index: int) -> str:
    if index >= landmark_map.NUM_STANDARD_LANDMARKS:
        return "skin"
    return landmark_map.group_for_standard_index(index) or "other"


def is_in_bounds_px(x: float, y: float, width: int, height: int) -> bool:
    return 0.0 < x < width and 0.0 < y < height


def write_frame_outputs(
    landmark_root: Path,
    capture_id: str,
    frame_id: str,
    neutral_frame_id: str,
    segment_id: str,
    points_by_group: dict[str, list[dict]],
    dry_run: bool,
) -> None:
    for group, points in points_by_group.items():
        if not points:
            continue
        payload = {
            "capture_id": capture_id,
            "frame_id": frame_id,
            "neutral_frame_id": neutral_frame_id,
            "segment_id": segment_id,
            "group": group,
            "points": points,
        }
        out_dir = landmark_root / capture_id
        out_path = out_dir / f"{group}_{frame_id}.json"
        if dry_run:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(out_path)


# --- main propagation ----------------------------------------------------------

def process_capture(
    capture_id: str,
    *,
    ava256_data_root: Path,
    ava256_landmark_root: Path,
    ava256_output_root: Path,
    label_tracker_path: Path,
    ava256_mesh_topology_path: Path,
    min_cameras_per_landmark: int,
    angle_threshold_deg: float,
    segments_filter: list[str] | None,
    max_frames_per_segment: int | None,
    wrap_script_path: Path = DEFAULTS["wrap_script_path"],
    template_pot_path: Path = DEFAULTS["template_pot_path"],
    template_wrap_path: Path = DEFAULTS["template_wrap_path"],
    faceform_wrap_cmd_path: Path = DEFAULTS["faceform_wrap_cmd_path"],
    faceform_wrap_license_path: Path = DEFAULTS["faceform_wrap_license_path"],
    sam_checkpoint_path: Path = DEFAULTS["sam_checkpoint_path"],
    u2net_checkpoint_path: Path = DEFAULTS["u2net_checkpoint_path"],
    wrap_script_dir: Path = DEFAULTS["wrap_script_dir"],
    facescape_mask_path: Path = DEFAULTS["facescape_mask_path"],
    blendshape_root: Path = DEFAULTS["blendshape_root"],
    facescape_run_pipeline_path: Path = DEFAULTS["facescape_run_pipeline_path"],
    triangulate_script_path: Path = DEFAULTS["triangulate_script_path"],
    cotracker_root: Path = DEFAULTS["cotracker_root"],
    raft_checkpoint_path: Path = DEFAULTS["raft_checkpoint_path"],
    cotracker_checkpoint_path: Path = DEFAULTS["cotracker_checkpoint_path"],
    dry_run: bool = False,
    force: bool = False,
) -> None:
    ensure_label_tracker_file(label_tracker_path)
    ava256_landmark_root.mkdir(parents=True, exist_ok=True)
    ava256_output_root.mkdir(parents=True, exist_ok=True)

    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    mesh_utils.ensure_capture_extracted(actor_dir)
    mesh_utils.FACESCAPE_RUN_PIPELINE_PATH = Path(facescape_run_pipeline_path)

    with _capture_propagation_lock(ava256_landmark_root, capture_id) as acquired:
        if not acquired:
            print(f"NOTE: {capture_id} already being propagated by another run -- skipping")
            return

        # Idempotency: confirm_labeled() below only ever fires once (its own
        # guard is "unlabeled -> unreviewed"), so a capture already past
        # "unlabeled" has already been fully propagated by a prior run --
        # skip the entire (expensive: camera selection + RAFT/CoTracker
        # across every frame) recompute, mirroring run_pipeline.py's own
        # idempotency check. Matters because discover_unlabeled_captures()'s
        # own status filter only applies to the automatic sweep -- an
        # explicit --captures list (batch OR single-capture CLI) bypasses it
        # entirely and would otherwise redo this every time.
        if not force:
            existing_status = Ava256LabelTracker(label_tracker_path).get_status(capture_id)
            if existing_status != "unlabeled":
                print(
                    f"NOTE: {capture_id} already propagated (status={existing_status!r}) "
                    f"-- skipping (use force=True/--force to re-propagate)"
                )
                return

        import neutral_frame as neutral_frame_module
        neutral = neutral_frame_module.resolve_neutral_frame(capture_id, actor_dir, ava256_landmark_root)
        neutral_frame_id = neutral["frame_id"]

        wrapped_mesh_path = ava256_output_root / capture_id / neutral_frame_id / "wrapped_mesh.obj"
        if not wrapped_mesh_path.exists():
            print(f"Wrapping {capture_id}/{neutral_frame_id} first (bootstrap, missing {wrapped_mesh_path})...")
            wrap_pipeline(
                capture_id,
                ava256_data_root=ava256_data_root,
                ava256_landmark_root=ava256_landmark_root,
                ava256_output_root=ava256_output_root,
                label_tracker_path=label_tracker_path,
                ava256_mesh_topology_path=ava256_mesh_topology_path,
                wrap_script_path=wrap_script_path,
                template_pot_path=template_pot_path,
                template_wrap_path=template_wrap_path,
                faceform_wrap_cmd_path=faceform_wrap_cmd_path,
                faceform_wrap_license_path=faceform_wrap_license_path,
                sam_checkpoint_path=sam_checkpoint_path,
                u2net_checkpoint_path=u2net_checkpoint_path,
                wrap_script_dir=wrap_script_dir,
                facescape_mask_path=facescape_mask_path,
                blendshape_root=blendshape_root,
                facescape_run_pipeline_path=facescape_run_pipeline_path,
                include_skin=False,
                dry_run=dry_run,
            )
            if not dry_run and not wrapped_mesh_path.exists():
                print(f"FAIL {capture_id}: {wrapped_mesh_path} still missing after bootstrap wrap")
                return

        if dry_run and not wrapped_mesh_path.exists():
            print(f"[dry-run] {capture_id}: would bootstrap-wrap {neutral_frame_id}, then select cameras + propagate")
            return

        mesh = trimesh.load(str(wrapped_mesh_path), process=False)
        # wrapped_mesh.obj is in the small FLAME-template-comparable frame
        # run_pipeline.py's dynamic_transform step wraps into (~0.2-unit
        # bbox), but camera_selection.py's occlusion/scoring and the neutral-
        # frame seed projection below both need REAL-world-scale vertices to
        # be geometrically meaningful against real camera K/Rt -- undo it.
        transform_params = dynamic_transform.load_params(dynamic_transform.params_path(ava256_output_root, capture_id))
        wrapped_verts = dynamic_transform.untransform(np.asarray(mesh.vertices, dtype=np.float64), transform_params)
        wrapped_faces = np.asarray(mesh.faces, dtype=np.int64)

        # template_pot_path is the base 74-row file; live-extend with
        # today's actual skin vertex set, same as run_pipeline.py -- both
        # scripts must agree on the SAME live-extended row indices/order.
        _wrap_mesh_fn, default_neutral_mesh_path, _wrap_script_module = _load_wrap_script(wrap_script_path)
        extended_pot_path = mesh_utils.build_live_extended_pot(template_pot_path, default_neutral_mesh_path)
        with Path(extended_pot_path).open("r", encoding="utf-8") as f:
            pot_rows = json.load(f)
        # Only propagate indices with a REAL source: standard indices with a
        # keypoints_3d id (45 of them -- STANDARD_INDEX_TO_KEYPOINT3D_ID's
        # own keys already exclude 2/9 and everything else with no sapiens2
        # source) plus every skin index (74+, FLAME-vertex-anchored, same as
        # FaceScape). The other ~27 standard indices (0-73) have NO
        # automatic source at all -- not even in FaceScape -- so their
        # neutral-frame position is just wherever generic ICP/blend-wrap
        # interpolation happened to put that mesh vertex, not a real
        # landmark; propagating/tracking that is meaningless and was
        # inflating the non-skin point count well past FaceScape's own 45.
        landmark_indices = sorted(
            set(landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID.keys())
            | set(range(landmark_map.NUM_STANDARD_LANDMARKS, len(pot_rows)))
        )
        pot_rows_by_index = {i: pot_rows[i] for i in landmark_indices}

        camera_ids = camera_utils.load_all_camera_ids(actor_dir)
        camera_params = {cid: camera_utils.load_camera(actor_dir, cid) for cid in camera_ids}

        print(f"Selecting cameras for {len(landmark_indices)} landmarks across {len(camera_ids)} cameras...")
        assignment = camera_selection.select_cameras_for_landmarks(
            landmark_indices, pot_rows_by_index, wrapped_verts, wrapped_faces,
            camera_ids, camera_params,
            min_cameras=min_cameras_per_landmark, angle_threshold_deg=angle_threshold_deg,
        )
        cameras_used = sorted({c for cams in assignment.values() for c in cams})
        print(f"Camera selection done: {len(cameras_used)} distinct cameras across {len(landmark_indices)} landmarks")

        # camera -> landmark_indices it's responsible for
        landmarks_by_camera: dict[str, list[int]] = {}
        for idx, cams in assignment.items():
            for cam_id in cams:
                landmarks_by_camera.setdefault(cam_id, []).append(idx)

        # Neutral-frame 2D seed per (camera, landmark): projection of the
        # already-known 3D position, not detection -- exact by construction.
        neutral_seed_px: dict[str, dict[int, tuple[float, float]]] = {}
        for cam_id, indices in landmarks_by_camera.items():
            K, Rt = camera_params[cam_id]
            xyzs = np.array([mesh_utils.pot_row_world_xyz(wrapped_verts, wrapped_faces, pot_rows_by_index[i]) for i in indices])
            px = camera_utils.project_points(xyzs, K, Rt)
            neutral_seed_px[cam_id] = {idx: (float(px[j, 0]), float(px[j, 1])) for j, idx in enumerate(indices)}

        segments = read_segments(actor_dir)
        neutral_segment_id = next((seg for seg, frames in segments.items() if neutral_frame_id in frames), NEUTRAL_SEG_ID)

        if segments_filter:
            segments = {seg: frames for seg, frames in segments.items() if seg in segments_filter}
        if max_frames_per_segment:
            segments = {seg: frames[:max_frames_per_segment] for seg, frames in segments.items()}

        if dry_run:
            print(f"[dry-run] {capture_id}: would propagate to {sum(len(f) for f in segments.values())} frame(s) "
                  f"across {len(segments)} segment(s), using {len(cameras_used)} camera(s)")
            return

        cotracker_model = load_cotracker_model(cotracker_root, cotracker_checkpoint_path)
        raft_model = load_raft_model(raft_checkpoint_path)
        triangulate_point = load_triangulate_point(triangulate_script_path)

        # frame_id -> group -> list of point payloads, accumulated across
        # every camera's own per-frame tracked result, then triangulated
        # once all cameras for that frame are in.
        tracked_by_frame_and_landmark: dict[str, dict[int, list[tuple[str, tuple[float, float]]]]] = {}

        def _try_load_image(cam_id: str, frame_id: str) -> np.ndarray | None:
            """Not every camera's zip contains every frame_id (confirmed on
            disk) -- returns None instead of raising so one missing frame
            skips gracefully rather than crashing the whole capture."""
            try:
                return camera_utils.load_image(actor_dir, cam_id, frame_id)
            except (KeyError, FileNotFoundError) as exc:
                print(f"NOTE: missing image cam={cam_id} frame={frame_id} ({exc}) -- skipping this (camera, frame)")
                return None

        for cam_id, indices in landmarks_by_camera.items():
            seeds = neutral_seed_px[cam_id]

            for seg_id, frame_ids in segments.items():
                if not frame_ids:
                    continue
                if seg_id == neutral_segment_id and neutral_frame_id in frame_ids:
                    loaded = [(fid, _try_load_image(cam_id, fid)) for fid in frame_ids]
                    usable_frame_ids = [fid for fid, img in loaded if img is not None]
                    if neutral_frame_id not in usable_frame_ids:
                        print(f"NOTE: cam={cam_id} missing its own neutral frame image -- skipping this camera's neutral segment")
                        continue
                    images = [img for _fid, img in loaded if img is not None]
                    query_t = usable_frame_ids.index(neutral_frame_id)
                    per_frame = cotracker_track_clip(cotracker_model, images, query_t, seeds)
                    for t, frame_id in enumerate(usable_frame_ids):
                        for idx, (xy, vis) in per_frame.get(t, {}).items():
                            tracked_by_frame_and_landmark.setdefault(frame_id, {}).setdefault(idx, []).append(
                                (cam_id, (float(xy[0]), float(xy[1])))
                            )
                else:
                    neutral_img = _try_load_image(cam_id, neutral_frame_id)
                    if neutral_img is None:
                        continue
                    loaded = [(fid, _try_load_image(cam_id, fid)) for fid in frame_ids]
                    usable_frame_ids = [fid for fid, img in loaded if img is not None]
                    images = [img for _fid, img in loaded if img is not None]
                    if not usable_frame_ids:
                        continue
                    hop = raft_hop(raft_model, neutral_img, images[0], seeds)
                    seg_seeds = {idx: (float(xy[0]), float(xy[1])) for idx, (xy, _v) in hop.items()}
                    per_frame = cotracker_track_clip(cotracker_model, images, 0, seg_seeds)
                    for t, frame_id in enumerate(usable_frame_ids):
                        for idx, (xy, vis) in per_frame.get(t, {}).items():
                            tracked_by_frame_and_landmark.setdefault(frame_id, {}).setdefault(idx, []).append(
                                (cam_id, (float(xy[0]), float(xy[1])))
                            )

        # Triangulate per (frame, landmark) and write outputs.
        for frame_id, by_landmark in tracked_by_frame_and_landmark.items():
            points_by_group: dict[str, list[dict]] = {}
            seg_id = next((seg for seg, frames in segments.items() if frame_id in frames), "")
            for idx, observations in by_landmark.items():
                cams_attempted = [cam_id for cam_id, _ in observations]
                point_list = []
                intrinsics = []
                extrinsics = []
                for local_i, (cam_id, (x, y)) in enumerate(observations):
                    K, Rt = camera_params[cam_id]
                    width, height = resolution_from_K(K)
                    intrinsics.append(K)
                    extrinsics.append(Rt)
                    point_list.append((local_i, (x, y), is_in_bounds_px(x, y, width, height)))
                if len(point_list) < 2:
                    # triangulate_point()'s own MIN_VIEWS_FOR_TRIANGULATION=2
                    # fallback ("use every view anyway") still needs >=2 raw
                    # observations to solve a non-singular least-squares
                    # system -- a single camera successfully tracking this
                    # landmark this frame (e.g. every other assigned camera
                    # was missing this frame's image) is a real gap, not a
                    # bug, so skip it rather than let the singular-matrix
                    # exception propagate.
                    continue
                xyz = triangulate_point(point_list, np.array(intrinsics), extrinsics)
                if xyz is None:
                    continue
                group = group_for_index(idx)
                points_by_group.setdefault(group, []).append({
                    "index": idx,
                    "x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2]),
                    "cameras_used": [
                        c for c, (px, py) in observations
                        if is_in_bounds_px(px, py, *resolution_from_K(camera_params[c][0]))
                    ],
                    "cameras_attempted": cams_attempted,
                    "method": "cotracker_intra_segment" if seg_id == neutral_segment_id else "raft_hop+cotracker_intra_segment",
                })
            write_frame_outputs(
                ava256_landmark_root, capture_id, frame_id, neutral_frame_id, seg_id, points_by_group, dry_run
            )

        tracker = Ava256LabelTracker(label_tracker_path)
        result = tracker.confirm_labeled(capture_id)
        if result is None:
            print(f"NOTE: not moving {capture_id} to 'unreviewed' -- current status is {tracker.get_status(capture_id)!r}")


def resolution_from_K(K: np.ndarray) -> tuple[int, int]:
    """Approximate (width, height) from the principal point (cx, cy),
    assumed centered -- deliberately NOT derived by loading any specific
    frame's image: confirmed on disk that not every camera's zip contains
    every frame_id, so probing an arbitrary "first" frame_id across the
    whole capture crashes on cameras missing that specific one. K is
    already AVIF-scaled (render_overlay.py's load_camera()), so this lands
    in the same pixel space as the images/projections already used for the
    in-bounds check this feeds -- exact resolution isn't needed for that,
    just a close bound."""
    return int(round(K[0, 2] * 2)), int(round(K[1, 2] * 2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_id")
    parser.add_argument("--ava256-data-root", default=str(DEFAULTS["ava256_data_root"]))
    parser.add_argument("--ava256-landmark-root", default=str(DEFAULTS["ava256_landmark_root"]))
    parser.add_argument("--ava256-output-root", default=str(DEFAULTS["ava256_output_root"]))
    parser.add_argument("--label-tracker-path", default=str(DEFAULTS["label_tracker_path"]))
    parser.add_argument("--ava256-mesh-topology-path", default=str(DEFAULTS["ava256_mesh_topology_path"]))
    parser.add_argument("--wrap-script-path", default=str(DEFAULTS["wrap_script_path"]))
    parser.add_argument("--template-pot-path", default=str(DEFAULTS["template_pot_path"]))
    parser.add_argument("--template-wrap-path", default=str(DEFAULTS["template_wrap_path"]))
    parser.add_argument("--faceform-wrap-cmd-path", default=str(DEFAULTS["faceform_wrap_cmd_path"]))
    parser.add_argument("--faceform-wrap-license-path", default=str(DEFAULTS["faceform_wrap_license_path"]))
    parser.add_argument("--sam-checkpoint-path", default=str(DEFAULTS["sam_checkpoint_path"]))
    parser.add_argument("--u2net-checkpoint-path", default=str(DEFAULTS["u2net_checkpoint_path"]))
    parser.add_argument("--wrap-script-dir", default=str(DEFAULTS["wrap_script_dir"]))
    parser.add_argument("--facescape-mask-path", default=str(DEFAULTS["facescape_mask_path"]))
    parser.add_argument("--blendshape-root", default=str(DEFAULTS["blendshape_root"]))
    parser.add_argument("--facescape-run-pipeline-path", default=str(DEFAULTS["facescape_run_pipeline_path"]))
    parser.add_argument("--triangulate-script-path", default=str(DEFAULTS["triangulate_script_path"]))
    parser.add_argument("--cotracker-root", default=str(DEFAULTS["cotracker_root"]))
    parser.add_argument("--cotracker-checkpoint-path", default=str(DEFAULTS["cotracker_checkpoint_path"]))
    parser.add_argument("--raft-checkpoint-path", default=str(DEFAULTS["raft_checkpoint_path"]))
    parser.add_argument("--min-cameras-per-landmark", type=int, default=5)
    parser.add_argument("--angle-threshold-deg", type=float, default=80.0)
    parser.add_argument("--segments", nargs="+", default=None, help="Debug: restrict propagation to these seg_ids only")
    parser.add_argument("--max-frames-per-segment", type=int, default=None, help="Debug: cap frames processed per segment")
    parser.add_argument("--force", action="store_true",
                         help="Re-propagate even if this capture's label_tracker.json status is already past "
                              "'unlabeled' (normally skipped for free; use this to force a genuine re-propagate)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    process_capture(
        args.capture_id,
        ava256_data_root=Path(args.ava256_data_root),
        ava256_landmark_root=Path(args.ava256_landmark_root),
        ava256_output_root=Path(args.ava256_output_root),
        label_tracker_path=Path(args.label_tracker_path),
        ava256_mesh_topology_path=Path(args.ava256_mesh_topology_path),
        wrap_script_path=Path(args.wrap_script_path),
        template_pot_path=Path(args.template_pot_path),
        template_wrap_path=Path(args.template_wrap_path),
        faceform_wrap_cmd_path=Path(args.faceform_wrap_cmd_path),
        faceform_wrap_license_path=Path(args.faceform_wrap_license_path),
        sam_checkpoint_path=Path(args.sam_checkpoint_path),
        u2net_checkpoint_path=Path(args.u2net_checkpoint_path),
        wrap_script_dir=Path(args.wrap_script_dir),
        facescape_mask_path=Path(args.facescape_mask_path),
        blendshape_root=Path(args.blendshape_root),
        facescape_run_pipeline_path=Path(args.facescape_run_pipeline_path),
        triangulate_script_path=Path(args.triangulate_script_path),
        cotracker_root=Path(args.cotracker_root),
        cotracker_checkpoint_path=Path(args.cotracker_checkpoint_path),
        raft_checkpoint_path=Path(args.raft_checkpoint_path),
        min_cameras_per_landmark=args.min_cameras_per_landmark,
        angle_threshold_deg=args.angle_threshold_deg,
        segments_filter=args.segments,
        max_frames_per_segment=args.max_frames_per_segment,
        dry_run=args.dry_run,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
