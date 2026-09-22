#!/usr/bin/env python3
"""AVA-256 counterpart to wrap_script_facescape/render_overlay_frame.py --
overlays one frame's tracked mesh (kinematic_tracking/registration_vertices)
directly onto the matching camera photo.

Deviates from the FaceScape version in two ways, both forced by what's
actually in this download (checked across every actor under
/scratch/thirty/irwinngo/ava-256/):

1. Point overlay instead of a wireframe. registration_vertices.ply has 7306
   vertices (a full head+neck+shoulder tracking mesh), but the only topology
   asset in this checkout -- FaceView/ava-256/assets/face_topology.obj -- is a
   face-only 5779-vertex subset ("Does not include neck", per assets/README.md).
   There's no shipped index mapping between the two, and a quick per-axis
   bbox check of the first-5779-vs-remaining-1527 split showed near-identical
   ranges (not a clean anatomical prefix), so the vertex ordering isn't safe
   to assume. Drawing face_topology.obj's edges against registration_vertices
   would silently connect the wrong points. Plotting every vertex as a point
   only needs positions, which registration_vertices.ply gives directly, so
   it's correct regardless of that mismatch.

2. No "expression" argument. Unlike FaceScape's <actor>/<expression>/ layout,
   an AVA-256 actor's decoder/ directory covers every frame directly (no
   expression-level subdirectory), so this only takes actor/frame/camera.

Also, unlike ava-256/demos/mesh.py's plot_mesh_on_image (the reference this
was built from), this does NOT apply a decoder/head_pose/head_pose.zip
per-frame rigid transform before projecting -- no actor in this download has
a head_pose/ directory at all, so registration_vertices are projected as-is,
treated as already being in the same world/capture space as
camera_calibration.json. If a future download does include head_pose/, that
transform needs to be applied first (see mesh.py) or the overlay will drift.

Usage:
    python3 render_overlay_frame.py <ACTOR> <FRAME> <CAMERA>
"""
from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image

try:
    import pillow_avif  # noqa: F401  -- registers AVIF support with Pillow
except ImportError as exc:
    raise ImportError(
        "pillow-avif-plugin is required to decode AVA-256's .avif frames "
        "(pip install pillow-avif-plugin)"
    ) from exc

DEFAULT_AVA256_DATA_ROOT = Path("/scratch/thirty/irwinngo/ava-256")
DEFAULT_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/line_renders_ava256")

# camera_calibration.json's K/T are calibrated for the native capture
# resolution (4096x2668, see ava-256/utils.py's load_camera_calibration),
# but the .avif frames actually stored are downscaled by this factor
# (confirmed empirically: decoded frame is 667x1024, i.e. 2668/4 x 4096/4) --
# same factor ava-256/demos/mesh.py applies after projecting.
IMAGE_DOWNSCALE = 4

POINT_RADIUS = 2
POINT_COLOR_BGR = (20, 255, 20)


def _load_camera(camera_calibration_path: Path, camera_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (K, RT) as a standard 3x3 intrinsic and 3x4 [R|t] extrinsic --
    same transpose-undoing as ava-256/utils.py's load_camera_calibration:
    camera_calibration.json stores K and T transposed relative to that
    convention (T's translation is its last ROW, not last column)."""
    with camera_calibration_path.open("r", encoding="utf-8") as f:
        camera_list = json.load(f)["KRT"]

    for item in camera_list:
        if item["cameraId"] == camera_id:
            K = np.array(item["K"], dtype=np.float64).T
            T = np.array(item["T"], dtype=np.float64)
            RT = T[:4, :3].T
            return K, RT

    available = [item["cameraId"] for item in camera_list]
    raise KeyError(f"Camera {camera_id!r} not found in {camera_calibration_path} (available: {available})")


def _read_zip_member(zip_path: Path, member_name: str) -> bytes:
    with zipfile.ZipFile(zip_path) as z:
        return z.read(member_name)


def render_overlay_frame(
    actor_id: str,
    frame_id: int,
    camera_id: str,
    *,
    ava256_data_root: Path = DEFAULT_AVA256_DATA_ROOT,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    decoder_dir = Path(ava256_data_root) / actor_id / "decoder"
    if not decoder_dir.is_dir():
        raise FileNotFoundError(f"No decoder/ directory for actor {actor_id!r} at {decoder_dir}")

    camera_calibration_path = decoder_dir / "camera_calibration.json"
    if not camera_calibration_path.exists():
        raise FileNotFoundError(f"No camera_calibration.json at {camera_calibration_path}")
    K, RT = _load_camera(camera_calibration_path, camera_id)

    ply_zip_path = decoder_dir / "kinematic_tracking" / "registration_vertices.zip"
    if not ply_zip_path.exists():
        raise FileNotFoundError(f"No registration_vertices.zip at {ply_zip_path}")
    ply_member = f"{frame_id:06d}.ply"
    try:
        ply_bytes = _read_zip_member(ply_zip_path, ply_member)
    except KeyError:
        raise FileNotFoundError(f"No {ply_member} inside {ply_zip_path} -- frame {frame_id} may not exist")

    mesh = trimesh.load(io.BytesIO(ply_bytes), file_type="ply", process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)  # (V, 3)

    image_zip_path = decoder_dir / "image" / f"cam{camera_id}.zip"
    if not image_zip_path.exists():
        raise FileNotFoundError(f"No image zip at {image_zip_path}")
    image_member = f"cam{camera_id}/{frame_id:06d}.avif"
    try:
        image_bytes = _read_zip_member(image_zip_path, image_member)
    except KeyError:
        raise FileNotFoundError(f"No {image_member} inside {image_zip_path} -- frame {frame_id} may not exist")
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    verts_cam = verts @ RT[:3, :3].T + RT[:3, 3]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = verts_cam[:, 0] / verts_cam[:, 2] * fx + cx
    y = verts_cam[:, 1] / verts_cam[:, 2] * fy + cy
    verts2d = np.stack([x, y], axis=-1) / IMAGE_DOWNSCALE

    height, width = image.shape[:2]
    in_front = verts_cam[:, 2] > 0
    for (px, py), visible in zip(verts2d, in_front):
        if not visible:
            continue
        pxi, pyi = int(round(px)), int(round(py))
        if 0 <= pxi < width and 0 <= pyi < height:
            cv2.circle(image, (pxi, pyi), POINT_RADIUS, POINT_COLOR_BGR, -1, cv2.LINE_AA)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{actor_id}_{frame_id:06d}_{camera_id}.jpg"
    cv2.imwrite(str(output_path), image)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("actor_id", help="e.g. 20220809--1034--BJM420")
    parser.add_argument("frame_id", type=int, help="e.g. 28091 (matches a <frame_id:06d>.ply in registration_vertices.zip)")
    parser.add_argument("camera_id", help="e.g. 400939 (matches camera_calibration.json's cameraId / cam<id>.zip)")
    parser.add_argument("--ava256-data-root", default=str(DEFAULT_AVA256_DATA_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    output_path = render_overlay_frame(
        args.actor_id, args.frame_id, args.camera_id,
        ava256_data_root=Path(args.ava256_data_root),
        output_root=Path(args.output_root),
    )
    print(f"Saved overlay frame render to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
