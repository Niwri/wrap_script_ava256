#!/usr/bin/env python3
"""Face-ray-masking heatmap renderer -- runs face_ray_masking_heatmap.py's
compute_face_ray_mask(..., return_raw_weights=True) fork (a copy of the real
face_ray_masking_ava256.py) to get every face's raw, CONTINUOUS cumulative
camera-vote weight, then colors each visible triangle by that weight via a
colormap instead of the production script's binary green(included)/
red(excluded) split. Useful for seeing the actual vote distribution --
e.g. whether a face just barely crossed MIN_WEIGHT or is deep in confident
territory -- that the binary threshold + facescape_mask.txt intersection +
erosion + force-include pipeline stages normally hide.

This runs the full SAM/U2Net segmentation pass across every camera (same
cost as a normal face_ray_masking run) -- it does NOT reuse a previously
computed mask, since the binary pipeline never saves raw per-face weights.

Usage:
    python3 render_face_ray_mask_heatmap.py <CAPTURE_ID> <FRAME_ID> <CAMERA_ID> [--ava256-data-root ...] [--ava256-output-root ...] [--output-root ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import mesh_utils
import camera_utils
import dynamic_transform
import face_ray_masking_heatmap

DEFAULT_AVA256_DATA_ROOT = Path("/scratch/thirty/irwinngo/ava-256")
DEFAULT_AVA256_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/ava_256_output/")
DEFAULT_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/line_renders_ava256")
DEFAULT_TEMPLATE_POT_LIVE_PATH = SCRIPT_DIR / "neutral_correspondence_without_eyes_POT_all_live.json"

NEAR, FAR = 10.0, 5000.0
OVERLAY_ALPHA = 0.6
NO_VOTES_COLOR_BGR = (40, 40, 40)  # dark grey -- rasterized/visible but never hit by any camera

_glctx = None


def _build_clip_matrix(K: np.ndarray, width: int, height: int) -> np.ndarray:
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([
        [2 * fx / width, 0, 2 * cx / width - 1, 0],
        [0, 2 * fy / height, 2 * cy / height - 1, 0],
        [0, 0, (FAR + NEAR) / (FAR - NEAR), -2 * FAR * NEAR / (FAR - NEAR)],
        [0, 0, 1, 0],
    ], dtype=np.float64)


def render_face_ray_mask_heatmap(
    capture_id: str,
    frame_id: str,
    camera_id: str,
    *,
    ava256_data_root: Path = DEFAULT_AVA256_DATA_ROOT,
    ava256_output_root: Path = DEFAULT_AVA256_OUTPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    template_pot_path: Path = DEFAULT_TEMPLATE_POT_LIVE_PATH,
    save_weights_path: Path | None = None,
) -> tuple[Path, Path]:
    import nvdiffrast.torch as dr

    global _glctx

    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    frame_dir = Path(ava256_output_root) / capture_id / frame_id

    mesh = trimesh.load(str(frame_dir / "wrapped_mesh.obj"), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    num_faces = faces.shape[0]

    transform_params = dynamic_transform.load_params(
        dynamic_transform.params_path(ava256_output_root, capture_id)
    )
    verts_undone = dynamic_transform.untransform(verts, transform_params)

    camera_ids = camera_utils.load_all_camera_ids(actor_dir)
    camera_params = {cid: camera_utils.load_camera(actor_dir, cid) for cid in camera_ids}

    with Path(template_pot_path).open("r", encoding="utf-8") as f:
        pot_rows_by_index = {i: row for i, row in enumerate(json.load(f))}

    front, right, left = face_ray_masking_heatmap.classify_front_right_left(
        verts_undone, faces, pot_rows_by_index, camera_ids, camera_params,
    )
    print(f"front={len(front)} right={len(right)} left={len(left)}")

    face_to_weight = face_ray_masking_heatmap.compute_face_ray_mask(
        verts_undone, faces, actor_dir, frame_id, camera_ids, camera_params,
        front, right, left, pot_rows_by_index=pot_rows_by_index, return_raw_weights=True,
    )

    if save_weights_path is not None:
        Path(save_weights_path).write_text(
            json.dumps({str(k): v for k, v in face_to_weight.items()}), encoding="utf-8"
        )
        print(f"Saved raw per-face weights ({len(face_to_weight)} faces) -> {save_weights_path}")

    weights = np.zeros(num_faces, dtype=np.float32)
    for face_idx, weight in face_to_weight.items():
        weights[face_idx] = weight
    max_weight = float(weights.max()) if weights.max() > 0 else 1.0
    # uint8 heatmap indices: 0 reserved for "no votes at all" (rendered dark
    # grey below, not colormap black, so it reads as "never hit" rather than
    # "hit with weight ~0") -- every real vote maps into [1, 255].
    normalized = np.zeros(num_faces, dtype=np.uint8)
    has_votes = weights > 0
    normalized[has_votes] = np.clip(1 + (weights[has_votes] / max_weight) * 254, 1, 255).astype(np.uint8)
    face_colors_bgr = cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_JET).reshape(-1, 3)

    K, Rt = camera_utils.load_camera(actor_dir, camera_id)
    R, t = Rt[:3, :3], Rt[:3, 3]
    image = camera_utils.load_image(actor_dir, camera_id, frame_id)
    height, width = image.shape[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("nvdiffrast CUDA rasterizer requires a GPU")

    verts_cam = verts_undone @ R.T + t
    proj = _build_clip_matrix(K, width, height)
    ones = np.ones((verts_undone.shape[0], 1))
    cam_h = np.concatenate([verts_cam, ones], axis=1)
    clip = (proj @ cam_h.T).T

    clip_t = torch.from_numpy(clip).float().to(device)[None].contiguous()
    faces_t = torch.from_numpy(faces.astype(np.int32)).to(device).contiguous()

    if _glctx is None:
        _glctx = dr.RasterizeCudaContext()
    rast_out, _rast_db = dr.rasterize(_glctx, clip_t, faces_t, resolution=[height, width])

    tri_id = rast_out[0, ..., 3].detach().cpu().numpy().astype(np.int64)
    alpha = tri_id > 0
    face_idx_map = np.clip(tri_id - 1, 0, num_faces - 1)

    rendered = np.full((height, width, 3), NO_VOTES_COLOR_BGR, dtype=np.uint8)
    per_pixel_color = face_colors_bgr[face_idx_map]
    has_votes_px = has_votes[face_idx_map]
    rendered[alpha & has_votes_px] = per_pixel_color[alpha & has_votes_px]
    rendered[alpha & ~has_votes_px] = np.array(NO_VOTES_COLOR_BGR, dtype=np.uint8)

    overlay = image.copy()
    overlay[alpha] = (OVERLAY_ALPHA * rendered[alpha] + (1 - OVERLAY_ALPHA) * image[alpha]).astype(np.uint8)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    flat_path = output_root / f"{capture_id}_{frame_id}_{camera_id}_face_ray_mask_heatmap.jpg"
    overlay_path = output_root / f"{capture_id}_{frame_id}_{camera_id}_face_ray_mask_heatmap_over_photo.jpg"
    cv2.imwrite(str(flat_path), rendered)
    cv2.imwrite(str(overlay_path), overlay)
    print(f"{capture_id}/{frame_id}/{camera_id}: {int(has_votes.sum())}/{num_faces} faces with >0 votes, "
          f"max weight={max_weight:.2f} -> {flat_path}, {overlay_path}")
    return flat_path, overlay_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_id")
    parser.add_argument("frame_id")
    parser.add_argument("camera_id")
    parser.add_argument("--ava256-data-root", default=str(DEFAULT_AVA256_DATA_ROOT))
    parser.add_argument("--ava256-output-root", default=str(DEFAULT_AVA256_OUTPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--template-pot-path", default=str(DEFAULT_TEMPLATE_POT_LIVE_PATH))
    parser.add_argument("--save-weights-path", default=None,
                         help="Also dump the raw per-face weight dict as JSON to this path")
    args = parser.parse_args()

    frame_id = args.frame_id.strip().zfill(6)

    render_face_ray_mask_heatmap(
        args.capture_id, frame_id, args.camera_id,
        ava256_data_root=Path(args.ava256_data_root),
        ava256_output_root=Path(args.ava256_output_root),
        output_root=Path(args.output_root),
        template_pot_path=Path(args.template_pot_path),
        save_weights_path=Path(args.save_weights_path) if args.save_weights_path else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
