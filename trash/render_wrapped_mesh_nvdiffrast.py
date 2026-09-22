#!/usr/bin/env python3
"""Shaded nvdiffrast render of a capture's wrapped_mesh.obj from one exact
camera's own K/Rt (not a point/line overlay -- an actual rasterized,
Lambertian-shaded render of the mesh surface itself), for visually
inspecting a specific wrap's geometry/coverage in isolation.

The OpenGL-style clip-space projection matrix built from K below is
validated (in-code, at CALL time isn't checked, but was verified manually
against camera_utils.project_points()'s already-proven pinhole projection
before this script was written -- exact pixel-for-pixel match on the first
5 wrapped-mesh vertices) rather than assumed correct from a generic formula.

Usage:
    python3 render_wrapped_mesh_nvdiffrast.py <CAPTURE_ID> <FRAME_ID> <CAMERA_ID> [--ava256-output-root ...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
import nvdiffrast.torch as dr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import mesh_utils
import camera_utils

DEFAULT_AVA256_DATA_ROOT = Path("/scratch/thirty/irwinngo/ava-256")
DEFAULT_AVA256_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/ava_256_output/")
DEFAULT_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/line_renders_ava256")

NEAR, FAR = 10.0, 5000.0  # clip planes, same units as the wrapped mesh (~mm)

_glctx = None


def _build_clip_matrix(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """OpenGL-style clip-space projection from a pinhole K -- w = +Z_cam
    (OpenCV camera-space convention, Z forward), NOT the -Z-forward OpenGL
    camera convention, since verts_cam here already comes straight out of
    camera_utils' own OpenCV-style world_2_cam. Validated to reproduce
    camera_utils.project_points()'s pixel coordinates exactly."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    return np.array([
        [2 * fx / width, 0, 2 * cx / width - 1, 0],
        [0, 2 * fy / height, 2 * cy / height - 1, 0],
        [0, 0, (FAR + NEAR) / (FAR - NEAR), -2 * FAR * NEAR / (FAR - NEAR)],
        [0, 0, 1, 0],
    ], dtype=np.float64)


def render_wrapped_mesh(
    capture_id: str,
    frame_id: str,
    camera_id: str,
    *,
    ava256_data_root: Path,
    ava256_output_root: Path,
    output_root: Path,
    mesh_filename: str = "wrapped_mesh.obj",
    output_suffix: str = "_shaded",
) -> Path:
    """mesh_filename lets this render any per-frame mesh sitting in the same
    output_dir -- "wrapped_mesh.obj" (default, the FLAME-topology wrap
    result) or "scan.obj" (the wrap's TARGET: the raw registered
    kinematic_tracking mesh, world-space, un-wrapped -- useful for checking
    whether visible distortion comes from the wrap process itself or was
    already present in the input scan)."""
    global _glctx

    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    mesh_path = Path(ava256_output_root) / capture_id / frame_id / mesh_filename
    if not mesh_path.exists():
        raise FileNotFoundError(f"No {mesh_filename} at {mesh_path}")

    mesh = trimesh.load(str(mesh_path), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    K, Rt = camera_utils.load_camera(actor_dir, camera_id)
    R, t = Rt[:3, :3], Rt[:3, 3]
    image = camera_utils.load_image(actor_dir, camera_id, frame_id)
    height, width = image.shape[:2]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("nvdiffrast CUDA rasterizer requires a GPU")

    verts_cam = verts @ R.T + t
    proj = _build_clip_matrix(K, width, height)
    ones = np.ones((verts.shape[0], 1))
    cam_h = np.concatenate([verts_cam, ones], axis=1)
    clip = (proj @ cam_h.T).T  # (V,4)

    clip_t = torch.from_numpy(clip).float().to(device)[None].contiguous()  # (1,V,4)
    faces_t = torch.from_numpy(faces.astype(np.int32)).to(device).contiguous()

    # Headlight: light direction = camera forward (same direction the
    # camera looks), so surfaces facing the camera are lit -- a simple,
    # standard "clay render" shading choice, not meant to match the photo's
    # real lighting.
    light_dir_world = camera_utils.camera_forward_world(Rt)
    normals_t = torch.from_numpy(normals).float().to(device)[None]  # (1,V,3)
    light_t = torch.from_numpy(light_dir_world).float().to(device)

    if _glctx is None:
        _glctx = dr.RasterizeCudaContext()

    rast_out, _rast_db = dr.rasterize(_glctx, clip_t, faces_t, resolution=[height, width])
    normal_interp, _ = dr.interpolate(normals_t, rast_out, faces_t)  # (1,H,W,3)

    normal_np = normal_interp[0].detach().cpu().numpy()
    ndotl = np.clip(np.sum(normal_np * (-light_dir_world), axis=-1), 0.0, 1.0)
    ambient = 0.25
    shade = np.clip(ambient + 0.85 * ndotl, 0.0, 1.0)

    mesh_color = np.array([0.75, 0.72, 0.68])  # neutral clay tone
    rendered = (shade[..., None] * mesh_color[None, None, :] * 255.0).astype(np.uint8)
    rendered_bgr = rendered[:, :, ::-1].copy()

    alpha = (rast_out[0, ..., 3] > 0).detach().cpu().numpy()  # triangle_id>0 where mesh was hit
    composite = image.copy()
    composite[alpha] = rendered_bgr[alpha]

    output_root.mkdir(parents=True, exist_ok=True)
    shaded_path = output_root / f"{capture_id}_{frame_id}_{camera_id}{output_suffix}.jpg"
    composite_path = output_root / f"{capture_id}_{frame_id}_{camera_id}{output_suffix}_over_photo.jpg"

    import cv2
    black_bg = np.zeros_like(image)
    black_bg[alpha] = rendered_bgr[alpha]
    cv2.imwrite(str(shaded_path), black_bg)
    cv2.imwrite(str(composite_path), composite)
    print(f"{capture_id}/{frame_id}/{camera_id}: {alpha.sum()} px covered -> {shaded_path}, {composite_path}")
    return shaded_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_id")
    parser.add_argument("frame_id")
    parser.add_argument("camera_id")
    parser.add_argument("--ava256-data-root", default=str(DEFAULT_AVA256_DATA_ROOT))
    parser.add_argument("--ava256-output-root", default=str(DEFAULT_AVA256_OUTPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--target", action="store_true",
                         help="Render scan.obj (the wrap's target/input mesh) instead of wrapped_mesh.obj (the wrap's output)")
    args = parser.parse_args()

    mesh_filename = "scan.obj" if args.target else "wrapped_mesh.obj"
    output_suffix = "_scan_shaded" if args.target else "_shaded"
    render_wrapped_mesh(
        args.capture_id, args.frame_id, args.camera_id,
        ava256_data_root=Path(args.ava256_data_root),
        ava256_output_root=Path(args.ava256_output_root),
        output_root=Path(args.output_root),
        mesh_filename=mesh_filename,
        output_suffix=output_suffix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
