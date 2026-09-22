#!/usr/bin/env python3
"""Keyline wireframe render for an Ava-256 capture's NEUTRAL frame only --
mirrors wrap_script_facescape/render_overlay_frame.py's exact approach
(landmark_lines.npy's curated wireframe drawn over the real photo), which is
only possible for the neutral frame since that's the only frame with a full
wrapped_mesh.obj (see render_wrapped_landmarks.py's own module docstring for
why every other frame only gets sparse propagated points instead).

Usage:
    python3 render_neutral_keylines.py <CAPTURE_ID> <CAMERA_ID> [--ava256-output-root ...] [--frame FRAME_ID]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import mesh_utils
import camera_utils

LANDMARK_LINES_PATH = Path("/scratch/ondemand32/irwinngo/landmark_lines.npy")
# Same cutoff FaceScape's own render_overlay_frame.py uses -- the FLAME
# template both pipelines wrap onto has exactly 3931 vertices (eye region
# BFS-removed), so this is a no-op filter here too, kept for parity/safety.
WRAPPED_MESH_EYE_CUTOFF = 3931
LINE_THICKNESS = 2
LINE_COLOR_BGR = (20, 255, 20)

DEFAULT_AVA256_DATA_ROOT = Path("/scratch/thirty/irwinngo/ava-256")
DEFAULT_AVA256_LANDMARK_ROOT = Path("/scratch/ondemand32/irwinngo/landmarks_ava-256/")
DEFAULT_AVA256_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/ava_256_output/")
DEFAULT_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/line_renders_ava256")


def render_neutral_keylines(
    capture_id: str,
    camera_id: str,
    *,
    ava256_data_root: Path,
    ava256_landmark_root: Path,
    ava256_output_root: Path,
    output_root: Path,
    frame_id: str | None = None,
) -> Path:
    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)

    if frame_id is None:
        import neutral_frame as neutral_frame_module
        neutral = neutral_frame_module.resolve_neutral_frame(capture_id, actor_dir, ava256_landmark_root)
        frame_id = neutral["frame_id"]

    wrapped_mesh_path = ava256_output_root / capture_id / frame_id / "wrapped_mesh.obj"
    if not wrapped_mesh_path.exists():
        raise FileNotFoundError(f"No wrapped_mesh.obj at {wrapped_mesh_path} -- run run_pipeline.py for this capture first")

    mesh = trimesh.load(str(wrapped_mesh_path), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)

    landmark_lines = np.load(LANDMARK_LINES_PATH)
    landmark_lines = landmark_lines[(landmark_lines < WRAPPED_MESH_EYE_CUTOFF).all(axis=1)]
    landmark_lines = landmark_lines[(landmark_lines < len(verts)).all(axis=1)]

    K, Rt = camera_utils.load_camera(actor_dir, camera_id)
    image = camera_utils.load_image(actor_dir, camera_id, frame_id)
    verts2d = camera_utils.project_points(verts, K, Rt)

    for line in landmark_lines:
        p1 = (int(verts2d[line[0], 0]), int(verts2d[line[0], 1]))
        p2 = (int(verts2d[line[1], 0]), int(verts2d[line[1], 1]))
        cv2.line(image, p1, p2, LINE_COLOR_BGR, LINE_THICKNESS, cv2.LINE_AA)

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{capture_id}_{frame_id}_{camera_id}_keylines.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"{capture_id}/{frame_id}/{camera_id}: drew {len(landmark_lines)} keyline(s) -> {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_id")
    parser.add_argument("camera_id")
    parser.add_argument("--frame", default=None, help="Override the auto-resolved neutral frame")
    parser.add_argument("--ava256-data-root", default=str(DEFAULT_AVA256_DATA_ROOT))
    parser.add_argument("--ava256-landmark-root", default=str(DEFAULT_AVA256_LANDMARK_ROOT))
    parser.add_argument("--ava256-output-root", default=str(DEFAULT_AVA256_OUTPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    render_neutral_keylines(
        args.capture_id, args.camera_id,
        ava256_data_root=Path(args.ava256_data_root),
        ava256_landmark_root=Path(args.ava256_landmark_root),
        ava256_output_root=Path(args.ava256_output_root),
        output_root=Path(args.output_root),
        frame_id=args.frame,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
