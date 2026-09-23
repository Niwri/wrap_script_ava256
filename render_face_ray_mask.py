#!/usr/bin/env python3
"""Consolidated face_ray_masking visualization tool -- formalizes the render
that was run ad-hoc (inline heredoc scripts) for every face_ray_mask
verification image produced this session (FXN596, CDR970, DOT682, CRV122):
load a frame's own wrapped_mesh.obj, undo the capture-level dynamic scale/
translate transform back to real-world coordinates, rasterize via nvdiffrast
from a real camera's K/Rt, and color each visible triangle green (included in
the wrap's FaceMask restriction) or red (excluded) -- both a flat render and
a composite over the real photo.

IMPORTANT polarity note (see face_ray_masking_ava256.py's own module
docstring point 4, and run_pipeline.py's face-ray-masking step): the
face_ray_mask.json file this reads is written in EXCLUDED-polarity (the set
FaceForm/Wrap4D's FaceMask node actually excludes) -- this script takes the
complement to recover and color the actual INCLUDED (green) set, the same
way run_pipeline.py itself does right before writing that file.

Usage:
    python3 render_face_ray_mask.py <CAPTURE_ID> <FRAME_ID> <CAMERA_ID> [--mask-path ...] [--ava256-data-root ...] [--output-root ...]
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

DEFAULT_AVA256_DATA_ROOT = Path("/scratch/thirty/irwinngo/ava-256")
DEFAULT_AVA256_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/ava_256_output/")
DEFAULT_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/line_renders_ava256")

NEAR, FAR = 10.0, 5000.0  # clip planes, same units as the wrapped mesh (real-world mm)
INCLUDED_COLOR_BGR = (40, 220, 40)   # green
EXCLUDED_COLOR_BGR = (60, 60, 220)   # red
OVERLAY_ALPHA = 0.55

_glctx = None


def _build_clip_matrix(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """Same OpenGL-style clip-space projection as render_wrapped_mesh_
    nvdiffrast.py's own (now-trashed) version -- w = +Z_cam (OpenCV camera-
    space convention), validated there to reproduce camera_utils.
    project_points()'s pixel coordinates exactly; duplicated here rather than
    imported since that file is no longer part of the handoff set."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([
        [2 * fx / width, 0, 2 * cx / width - 1, 0],
        [0, 2 * fy / height, 2 * cy / height - 1, 0],
        [0, 0, (FAR + NEAR) / (FAR - NEAR), -2 * FAR * NEAR / (FAR - NEAR)],
        [0, 0, 1, 0],
    ], dtype=np.float64)


def render_face_ray_mask(
    capture_id: str,
    frame_id: str,
    camera_id: str,
    *,
    ava256_data_root: Path = DEFAULT_AVA256_DATA_ROOT,
    ava256_output_root: Path = DEFAULT_AVA256_OUTPUT_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    mask_path: Path | None = None,
) -> tuple[Path, Path]:
    """mask_path defaults to {ava256_output_root}/{capture_id}/{frame_id}/
    face_ray_mask.json -- the file run_pipeline.py's own face-ray-masking
    step writes (or, for a non-neutral frame reusing the neutral's mask via
    --face-ray-mask-path, wherever that pointed). Pass an explicit path to
    render a different/older mask file (e.g. one saved under a different
    name while comparing angle-threshold settings)."""
    import nvdiffrast.torch as dr

    global _glctx

    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    frame_dir = Path(ava256_output_root) / capture_id / frame_id
    mask_path = Path(mask_path) if mask_path is not None else frame_dir / "face_ray_mask.json"

    mesh = trimesh.load(str(frame_dir / "wrapped_mesh.obj"), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    transform_params = dynamic_transform.load_params(
        dynamic_transform.params_path(ava256_output_root, capture_id)
    )
    verts_undone = dynamic_transform.untransform(verts, transform_params)

    excluded_faces = set(json.load(mask_path.open("r", encoding="utf-8")))
    included_faces = set(range(faces.shape[0])) - excluded_faces

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

    tri_id = rast_out[0, ..., 3].detach().cpu().numpy().astype(np.int64)  # 1-indexed, 0=background
    alpha = tri_id > 0
    face_idx_map = np.clip(tri_id - 1, 0, faces.shape[0] - 1)

    included_lut = np.zeros(faces.shape[0], dtype=bool)
    if included_faces:
        included_lut[np.array(sorted(included_faces))] = True
    included_px = included_lut[face_idx_map]

    rendered = np.zeros((height, width, 3), dtype=np.uint8)
    rendered[alpha & included_px] = np.array(INCLUDED_COLOR_BGR, dtype=np.uint8)
    rendered[alpha & ~included_px] = np.array(EXCLUDED_COLOR_BGR, dtype=np.uint8)

    overlay = image.copy()
    overlay[alpha] = (OVERLAY_ALPHA * rendered[alpha] + (1 - OVERLAY_ALPHA) * image[alpha]).astype(np.uint8)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    flat_path = output_root / f"{capture_id}_{frame_id}_{camera_id}_face_ray_mask.jpg"
    overlay_path = output_root / f"{capture_id}_{frame_id}_{camera_id}_face_ray_mask_over_photo.jpg"
    cv2.imwrite(str(flat_path), rendered)
    cv2.imwrite(str(overlay_path), overlay)
    print(f"{capture_id}/{frame_id}/{camera_id}: {len(included_faces)}/{faces.shape[0]} faces included "
          f"-> {flat_path}, {overlay_path}")
    return flat_path, overlay_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_id")
    parser.add_argument("frame_id")
    parser.add_argument("camera_id")
    parser.add_argument("--mask-path", default=None,
                         help="Override which face_ray_mask.json to render (default: "
                              "{ava256-output-root}/{capture_id}/{frame_id}/face_ray_mask.json)")
    parser.add_argument("--ava256-data-root", default=str(DEFAULT_AVA256_DATA_ROOT))
    parser.add_argument("--ava256-output-root", default=str(DEFAULT_AVA256_OUTPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    # Same zero-pad-on-input fix as run_pipeline.py's --frame / render_
    # wrapped_landmarks.py's frame_id -- a raw user-typed 5-digit frame_id
    # otherwise silently fails to resolve against Ava-256's always-6-digit
    # on-disk file names.
    frame_id = args.frame_id.strip().zfill(6)

    render_face_ray_mask(
        args.capture_id, frame_id, args.camera_id,
        ava256_data_root=Path(args.ava256_data_root),
        ava256_output_root=Path(args.ava256_output_root),
        output_root=Path(args.output_root),
        mask_path=Path(args.mask_path) if args.mask_path else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
