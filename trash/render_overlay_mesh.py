#!/usr/bin/env python3
"""Renders one ava-256 actor's per-frame registered mesh
(decoder/kinematic_tracking/<frame>.ply) as a proper, depth-sorted mesh
render composited over one camera's photo at alpha=0.3 -- like
FaceView/camera_mesh_render.py, not render_overlay.py's flat-fill cousin.

Uses FaceView/render_nvdiffrast.py's NvdiffrastMeshRenderer (a real GPU
rasterizer: depth-tested, so near/far and front/back-facing triangles
occlude each other correctly, unlike a per-triangle cv2.fillPoly loop with
no depth sorting) to render the mesh alone against a black background, then
composites it onto the photo with a per-pixel mask (any non-black rendered
pixel) scaled by alpha -- same masking pattern as camera_mesh_render.py's
own frame loop, adapted from a video loop to a single frame/camera.

Every triangle from FaceView/ava-256/assets/face_topology.obj (see
render_overlay.py's own module docstring for why its face list, not its own
vertex positions, is what's used, and why it's parsed directly rather than
via trimesh) is what's rasterized.

Usage:
    python3 render_overlay_mesh.py <FRAME_ID> <CAMERA_ID> [--actor ACTOR] [--actor-dir ...] [--alpha 0.3]

--actor takes just the 6-character actor code (e.g. FXN596), the last
'--'-separated segment of the actor directory's own name (e.g.
20210810--1306--FXN596) -- resolved against ACTOR_ROOT by globbing for that
suffix, since the leading date/session segments aren't something a caller
would otherwise know. --actor-dir still accepts a full path directly (e.g.
for an actor directory that lives outside ACTOR_ROOT).
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/scratch/ondemand32/irwinngo/FaceView")
from render_nvdiffrast import NvdiffrastMeshRenderer  # noqa: E402

ACTOR_ROOT = Path("/scratch/thirty/irwinngo/ava-256")
DEFAULT_ACTOR_DIR = ACTOR_ROOT / "20210810--1306--FXN596"
OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/renders")
FACE_TOPOLOGY_PATH = Path("/scratch/ondemand32/irwinngo/FaceView/ava-256/assets/face_topology.obj")

# See render_overlay.py's own module docstring -- same AVIF-vs-raw-resolution
# mismatch, same fix.
AVIF_SCALE = 0.25
DEFAULT_ALPHA = 0.5


def resolve_actor_dir(actor: str) -> Path:
    """Resolves a 6-character actor code (e.g. 'FXN596') to its full
    directory under ACTOR_ROOT (e.g. .../20210810--1306--FXN596), by
    globbing for that suffix -- the leading date/session segments aren't
    something a caller would otherwise know."""
    matches = sorted(ACTOR_ROOT.glob(f"*--{actor}"))
    if not matches:
        raise FileNotFoundError(f"No actor directory matching '*--{actor}' found under {ACTOR_ROOT}")
    if len(matches) > 1:
        raise ValueError(f"Multiple actor directories matched '*--{actor}' under {ACTOR_ROOT}: {matches}")
    return matches[0]


def load_head_pose(actor_dir: Path, frame_id: str) -> np.ndarray:
    """Row-major 3x4 [R|t] -- local (registered-mesh) space -> world space."""
    path = actor_dir / "decoder" / "head_pose" / f"{frame_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"head_pose not found: {path} (extract decoder/head_pose/head_pose.zip first)")
    return np.loadtxt(path).reshape(3, 4)


def load_mesh_vertices(actor_dir: Path, frame_id: str) -> np.ndarray:
    import trimesh

    path = actor_dir / "decoder" / "kinematic_tracking" / f"{frame_id}.ply"
    if not path.exists():
        raise FileNotFoundError(f"registration mesh not found: {path}")
    mesh = trimesh.load(str(path), process=False)
    return np.asarray(mesh.vertices, dtype=np.float64)


def load_faces(num_vertices_expected: int) -> np.ndarray:
    """Minimal direct parser for FACE_TOPOLOGY_PATH's own 'f i/j j/k k/l ...'
    lines -- deliberately NOT trimesh, see render_overlay.py's own
    load_wireframe_edges()/_load_obj_faces() docstring for why (trimesh drops/
    renumbers vertices unreferenced by any (v, vt) pair in this specific
    file, desyncing face indices from kinematic_tracking's own per-frame
    vertex numbering)."""
    if not FACE_TOPOLOGY_PATH.exists():
        raise FileNotFoundError(f"face topology not found: {FACE_TOPOLOGY_PATH}")
    faces: list[list[int]] = []
    with FACE_TOPOLOGY_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("f "):
                continue
            faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    faces_arr = np.array(faces, dtype=np.int64)
    max_idx = int(faces_arr.max())
    if max_idx >= num_vertices_expected:
        raise ValueError(
            f"{FACE_TOPOLOGY_PATH} references vertex index {max_idx} (0-based), "
            f"but this frame's mesh only has {num_vertices_expected} vertices"
        )
    return faces_arr


def load_camera(actor_dir: Path, camera_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (K, world_2_cam) -- K already scaled by AVIF_SCALE (3x3),
    world_2_cam is the standard OpenCV-convention 4x4 extrinsic
    (p_cam = R @ p_world + t)."""
    calib_path = actor_dir / "decoder" / "camera_calibration.json"
    with open(calib_path) as f:
        calib = json.load(f)
    entry = next((c for c in calib["KRT"] if c["cameraId"] == camera_id), None)
    if entry is None:
        available = [c["cameraId"] for c in calib["KRT"]]
        raise KeyError(f"Camera {camera_id!r} not found in {calib_path}. Available: {available}")

    K = np.array(entry["K"], dtype=np.float64).T  # stored column-major -- see render_overlay.py
    K[0, 0] *= AVIF_SCALE  # fx
    K[1, 1] *= AVIF_SCALE  # fy
    K[0, 2] *= AVIF_SCALE  # cx
    K[1, 2] *= AVIF_SCALE  # cy

    world_2_cam = np.array(entry["T"], dtype=np.float64).T  # stored column-major, already 4x4 -- see render_overlay.py
    return K, world_2_cam


def load_image(actor_dir: Path, camera_id: str, frame_id: str) -> np.ndarray:
    """Reads one frame's AVIF photo straight out of its camera's zip, as BGR
    (cv2 convention)."""
    zip_path = actor_dir / "decoder" / "image" / f"cam{camera_id}.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"No image zip for camera {camera_id}: {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        member = f"cam{camera_id}/{frame_id}.avif"
        with zf.open(member) as f:
            pil_image = Image.open(f).convert("RGB")
            rgb = np.array(pil_image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def render_overlay_mesh(actor_dir: Path, frame_id: str, camera_id: str, alpha: float = DEFAULT_ALPHA) -> Path:
    head_pose = load_head_pose(actor_dir, frame_id)  # (3,4)
    verts_local = load_mesh_vertices(actor_dir, frame_id)  # (V,3)
    verts_world = verts_local @ head_pose[:3, :3].T + head_pose[:3, 3]

    faces = load_faces(num_vertices_expected=verts_local.shape[0])

    K, world_2_cam = load_camera(actor_dir, camera_id)

    image = load_image(actor_dir, camera_id, frame_id)
    height, width = image.shape[:2]

    # ava-256's kinematic_tracking mesh + head_pose/camera_calibration are all
    # millimeter-scale (camera-space depth is in the hundreds/thousands), unlike
    # NvdiffrastMeshRenderer's meter-scale defaults (near=0.01, far=100) -- with
    # those defaults every vertex here lands past the far clip plane and gets
    # rasterized as nothing, which is why the mesh never appeared in the
    # composite. See setup_camera_from_extrinsics()'s own docstring.
    renderer = NvdiffrastMeshRenderer(width=width, height=height)
    renderer.setup_camera_from_extrinsics(K, world_2_cam, near=1.0, far=5000.0)
    rendered = renderer.render_frame(verts_world, faces)  # (H, W, 3) RGB, black background
    rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
    rendered_bgr = cv2.flip(rendered_bgr, 0)  # see render_faceview_animation's own flip

    # Same per-pixel masking as camera_mesh_render.py's own frame loop: any
    # non-black rendered pixel is mesh, scaled by alpha -- not a flat
    # whole-image blend, so background pixels are untouched.
    mask = np.any(rendered_bgr != 0, axis=-1).astype(np.float32) * alpha
    mask3 = mask[..., None]
    composited = rendered_bgr.astype(np.float32) * mask3 + image.astype(np.float32) * (1 - mask3)
    composited = np.clip(composited, 0, 255).astype(np.uint8)

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{actor_dir.name}_{frame_id}_{camera_id}_mesh.jpg"
    cv2.imwrite(str(output_path), composited)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("frame_id", help="Frame identifier, e.g. 029693 (matches kinematic_tracking/head_pose/image filenames)")
    parser.add_argument("camera_id", help="Camera identifier without the 'cam' prefix, e.g. 400939")
    actor_group = parser.add_mutually_exclusive_group()
    actor_group.add_argument("--actor", default=None,
                              help="6-character actor code, e.g. FXN596 (resolved against ACTOR_ROOT="
                                   f"{ACTOR_ROOT})")
    actor_group.add_argument("--actor-dir", default=None, help=f"Full ava-256 actor directory (default: {DEFAULT_ACTOR_DIR})")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA, help=f"Mesh opacity, 0-1 (default: {DEFAULT_ALPHA})")
    args = parser.parse_args()

    if args.actor is not None:
        actor_dir = resolve_actor_dir(args.actor)
    elif args.actor_dir is not None:
        actor_dir = Path(args.actor_dir)
    else:
        actor_dir = DEFAULT_ACTOR_DIR

    output_path = render_overlay_mesh(actor_dir, args.frame_id, args.camera_id, args.alpha)
    print(f"Saved mesh overlay render to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
