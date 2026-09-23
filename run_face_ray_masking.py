#!/usr/bin/env python3
"""Standalone CLI for face_ray_masking_ava256.py -- formalizes the inline
heredoc scripts run ad-hoc all session for every face_ray_masking test
(FXN596, CDR970, DOT682, CRV122): auto-resolves the capture's own neutral
frame (same neutral_frame.resolve_neutral_frame() every other pipeline
script uses -- face_ray_masking is only ever run on the neutral, since
that's the frame the skin-augmented wrap step actually needs it for; see
run_pipeline.py's own "no need on the first wrap" note), loads that frame's
wrapped_mesh.obj, undoes the capture-level dynamic transform back to
real-world coordinates, classifies cameras via classify_front_right_left(),
then runs compute_face_ray_mask() and writes face_ray_mask.json to that
frame's own output dir -- the SAME file run_pipeline.py itself writes
during a real skin-augmented wrap, in the same EXCLUDED-polarity convention
(see face_ray_masking_ava256.py's own module docstring point 4), so render_
face_ray_mask.py picks it up with zero extra flags afterward.

This runs the full SAM/U2Net segmentation pass across every camera --
there's no shortcut for that part. Pass --save-weights-path to also dump
the raw (pre-threshold) per-face weight dict as JSON, so a later run can
re-threshold at a different MIN_WEIGHT without repeating the segmentation
pass (see face_ray_masking_ava256.MIN_WEIGHT and its own save_weights_path
parameter).

Usage:
    python3 run_face_ray_masking.py <CAPTURE_ID> [--save-weights-path ...] [--ava256-data-root ...] [--ava256-output-root ...] [--ava256-landmark-root ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import mesh_utils
import camera_utils
import dynamic_transform
import neutral_frame as neutral_frame_module
import face_ray_masking_ava256

DEFAULT_AVA256_DATA_ROOT = Path("/scratch/thirty/irwinngo/ava-256")
DEFAULT_AVA256_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/ava_256_output/")
DEFAULT_AVA256_LANDMARK_ROOT = Path("/scratch/ondemand32/irwinngo/landmarks_ava-256/")
DEFAULT_TEMPLATE_POT_LIVE_PATH = SCRIPT_DIR / "neutral_correspondence_without_eyes_POT_all_live.json"


def run_face_ray_masking(
    capture_id: str,
    *,
    ava256_data_root: Path = DEFAULT_AVA256_DATA_ROOT,
    ava256_output_root: Path = DEFAULT_AVA256_OUTPUT_ROOT,
    ava256_landmark_root: Path = DEFAULT_AVA256_LANDMARK_ROOT,
    template_pot_path: Path = DEFAULT_TEMPLATE_POT_LIVE_PATH,
    save_weights_path: Path | None = None,
) -> Path:
    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    neutral = neutral_frame_module.resolve_neutral_frame(capture_id, actor_dir, ava256_landmark_root)
    frame_id = neutral["frame_id"]
    print(f"{capture_id}: neutral frame = {frame_id}")

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

    front, right, left = face_ray_masking_ava256.classify_front_right_left(
        verts_undone, faces, pot_rows_by_index, camera_ids, camera_params,
    )
    print(f"front={len(front)} right={len(right)} left={len(left)}")

    included = face_ray_masking_ava256.compute_face_ray_mask(
        verts_undone, faces, actor_dir, frame_id, camera_ids, camera_params,
        front, right, left, pot_rows_by_index=pot_rows_by_index,
        save_weights_path=save_weights_path,
    )

    # Standard polarity (matches run_pipeline.py's own face_ray_mask.json):
    # write the EXCLUDED complement, not the raw included set -- see
    # face_ray_masking_ava256.py's module docstring point 4 for why.
    excluded = sorted(set(range(num_faces)) - set(included))
    mask_path = frame_dir / "face_ray_mask.json"
    mask_path.write_text(json.dumps(excluded), encoding="utf-8")
    print(f"{capture_id}/{frame_id}: {len(included)}/{num_faces} faces included -> {mask_path}")
    return mask_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_id")
    parser.add_argument("--ava256-data-root", default=str(DEFAULT_AVA256_DATA_ROOT))
    parser.add_argument("--ava256-output-root", default=str(DEFAULT_AVA256_OUTPUT_ROOT))
    parser.add_argument("--ava256-landmark-root", default=str(DEFAULT_AVA256_LANDMARK_ROOT))
    parser.add_argument("--template-pot-path", default=str(DEFAULT_TEMPLATE_POT_LIVE_PATH))
    parser.add_argument("--save-weights-path", default=None,
                         help="Also dump the raw (pre-threshold) per-face weight dict as JSON to this path")
    args = parser.parse_args()

    run_face_ray_masking(
        args.capture_id,
        ava256_data_root=Path(args.ava256_data_root),
        ava256_output_root=Path(args.ava256_output_root),
        ava256_landmark_root=Path(args.ava256_landmark_root),
        template_pot_path=Path(args.template_pot_path),
        save_weights_path=Path(args.save_weights_path) if args.save_weights_path else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
