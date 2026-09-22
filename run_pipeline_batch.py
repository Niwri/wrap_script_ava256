#!/usr/bin/env python3
"""Batch/sweep wrapper for run_pipeline.py. Simpler than FaceScape's
run_pipeline_batch.py -- no neutral-first two-pass ordering is needed, since
there's exactly one wrap target per Ava-256 capture (the neutral frame
itself), not many expressions chaining off a shared neutral.

Usage:
    python3 run_pipeline_batch.py [--captures ID ...] [--parallel N] [--gpu-ids 0 1 ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = SCRIPT_DIR / "run_pipeline.py"
sys.path.insert(0, str(SCRIPT_DIR))

from label_tracker_ava256 import Ava256LabelTracker, ensure_label_tracker_file

DEFAULTS = {
    "ava256_data_root": Path("/scratch/thirty/irwinngo/ava-256"),
    "ava256_landmark_root": Path("/scratch/ondemand32/irwinngo/landmarks_ava-256/"),
    "ava256_output_root": Path("/scratch/ondemand32/irwinngo/ava_256_output/"),
    "label_tracker_path": Path("/scratch/ondemand32/irwinngo/label_tracker.json"),
    # Local, self-contained copy -- see run_pipeline.py's own DEFAULTS
    # comment on the directory-consolidation into wrap_script_ava256/.
    "ava256_mesh_topology_path": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/face_topology.obj"),
    "wrap_script_path": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/run_wrap_script.py"),
    # Base 74-row file -- see run_pipeline.py's own DEFAULTS comment for why
    # not the cached "_all" file (skin rows are live-extended at runtime).
    "template_pot_path": Path(
        "/scratch/ondemand32/irwinngo/wrap_script_ava256/neutral_correspondence_without_eyes_POT.json"
    ),
    "template_wrap_path": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/template.wrap"),
    "faceform_wrap_cmd_path": Path("/scratch/ondemand32/irwinngo/faceform/Faceform_Wrap_2025.11.14_Linux/WrapCmd"),
    "faceform_wrap_license_path": Path(
        "/scratch/ondemand32/irwinngo/faceform/Faceform_Wrap_License_1035244_10483637.lic"
    ),
    # --- portability: same new flags run_pipeline.py itself gained (see its
    # own DEFAULTS comment) -- forwarded to each per-capture subprocess below.
    "sam_checkpoint_path": Path("/scratch/ondemand32/irwinngo/models/sam_vit_h_4b8939.pth"),
    "u2net_checkpoint_path": Path("/scratch/ondemand32/irwinngo/models/u2net_human_seg.pth"),
    "wrap_script_dir": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/segmentation"),
    "facescape_mask_path": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/facescape_mask.txt"),
    "blendshape_root": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/blendshape"),
    "facescape_run_pipeline_path": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/facescape_run_pipeline.py"),
    # Ava-256's own equivalent of Nersemble's LINE_RENDERS_ROOT convention --
    # see render_wrapped_landmarks.py's render_keyline_video().
    "line_renders_root": Path("/scratch/ondemand32/irwinngo/line_renders_ava256"),
}

POLL_INTERVAL_SECONDS = 60


@dataclass
class CaptureResult:
    capture_id: str
    passed: bool


def discover_unreviewed_captures(ava256_data_root: Path, label_tracker_path: Path) -> list[str]:
    tracker = Ava256LabelTracker(label_tracker_path)
    all_statuses = tracker.get_all()
    captures = sorted(p.name for p in Path(ava256_data_root).iterdir() if p.is_dir())
    return [
        c for c in captures
        if tracker.get_status(c, all_statuses=all_statuses) == "unreviewed"
    ]


def process_capture(capture_id: str, root_flags: list[str], log_dir: Path | None, gpu_id: str | None, dry_run: bool) -> CaptureResult:
    tag = f"[capture={capture_id}]" + (f" gpu={gpu_id}" if gpu_id is not None else "")
    cmd = [sys.executable, str(PIPELINE_SCRIPT), capture_id, *root_flags]
    if dry_run:
        cmd.append("--dry-run")
    print("RUN  " + " ".join(cmd) + f"  {tag}")

    job_env = None
    if gpu_id is not None:
        job_env = os.environ.copy()
        job_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    if log_dir is None:
        returncode = subprocess.run(cmd, env=job_env).returncode
    else:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{capture_id}.log"
        with log_path.open("w") as log_file:
            returncode = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=job_env).returncode

    if returncode != 0:
        print(f"FAIL {tag}: exit_code={returncode}")
        return CaptureResult(capture_id, passed=False)
    return CaptureResult(capture_id, passed=True)


def render_capture_keyline_video(capture_id: str, args) -> None:
    """Post-capture step: after every frame run_pipeline.py wrapped for this
    capture is on disk, render one keyline-only video across all of them
    (Ava-256's equivalent of Nersemble's own keyline-video convention -- see
    wrap_script/render_landmarks.py/batch_render_gt.py). Camera choice is
    derived fresh via classify_front_right_left() on the capture's own
    neutral-frame wrapped mesh (NOT hardcoded -- confirmed this session that
    no single camera id is front-facing for every actor's rig framing)."""
    import mesh_utils
    import neutral_frame as neutral_frame_module
    import render_wrapped_landmarks
    import face_ray_masking_ava256
    import camera_utils

    ava256_data_root = Path(args.ava256_data_root)
    ava256_output_root = Path(args.ava256_output_root)
    ava256_landmark_root = Path(args.ava256_landmark_root)

    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    neutral = neutral_frame_module.resolve_neutral_frame(capture_id, actor_dir, ava256_landmark_root)
    neutral_frame_id = neutral["frame_id"]

    try:
        verts_world, faces = render_wrapped_landmarks.load_wrapped_mesh_world(ava256_output_root, capture_id, neutral_frame_id)
    except FileNotFoundError as exc:
        print(f"NOTE: skipping keyline video for {capture_id} -- {exc}")
        return

    with render_wrapped_landmarks.DEFAULT_TEMPLATE_POT_LIVE_PATH.open("r", encoding="utf-8") as f:
        pot_rows_by_index = {i: row for i, row in enumerate(json.load(f))}

    camera_ids = camera_utils.load_all_camera_ids(actor_dir)
    camera_params = {cid: camera_utils.load_camera(actor_dir, cid) for cid in camera_ids}
    front, _right, _left = face_ray_masking_ava256.classify_front_right_left(
        verts_world, faces, pot_rows_by_index, camera_ids, camera_params,
    )
    if not front:
        print(f"WARNING: no front-facing camera derived for {capture_id} -- skipping keyline video")
        return
    camera_id = sorted(front)[0]  # deterministic pick among equally-valid front cameras

    frame_ids = render_wrapped_landmarks.list_wrapped_frames_in_capture_order(
        actor_dir, ava256_output_root, capture_id, neutral_frame_id=neutral_frame_id,
    )
    render_wrapped_landmarks.render_keyline_video(
        capture_id, frame_ids, camera_id,
        ava256_data_root=ava256_data_root,
        ava256_output_root=ava256_output_root,
        line_renders_root=Path(args.line_renders_root),
    )


def run_sweep(args, root_flags: list[str]) -> tuple[int, int]:
    if args.captures:
        captures = args.captures
    else:
        captures = discover_unreviewed_captures(Path(args.ava256_data_root), Path(args.label_tracker_path))

    if not captures:
        print("No captures to process.")
        return 0, 0

    passed = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = {
            executor.submit(
                process_capture, capture_id, root_flags,
                Path(args.log_dir) if args.log_dir else None,
                args.gpu_ids[i % len(args.gpu_ids)] if args.gpu_ids else None,
                args.dry_run,
            ): capture_id
            for i, capture_id in enumerate(captures)
        }
        for future in as_completed(futures):
            result = future.result()
            if result.passed:
                passed += 1
                if not args.no_video and not args.dry_run:
                    try:
                        render_capture_keyline_video(result.capture_id, args)
                    except Exception as exc:  # noqa: BLE001 -- a video-render failure shouldn't fail the whole sweep
                        print(f"WARNING: keyline video failed for {result.capture_id}: {exc}")
            else:
                failed += 1
    return passed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", nargs="+", default=None, help="Explicit capture ids; omit to sweep every capture at 'unreviewed' status")
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
    parser.add_argument("--no-face-ray-masking", action="store_true")
    parser.add_argument("--force", action="store_true",
                         help="Forward run_pipeline.py's --force to every capture -- re-wrap even if label_tracker.json "
                              "already shows it done (normally skipped for free; use this to force a genuine re-wrap)")
    parser.add_argument("--line-renders-root", default=str(DEFAULTS["line_renders_root"]),
                         help="Output directory for the per-capture keyline video (Ava-256's own LINE_RENDERS_ROOT)")
    parser.add_argument("--no-video", action="store_true",
                         help="Skip the automatic post-capture keyline video render (batch wrapping only)")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="+", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--poll", action="store_true", help="Loop forever, re-sweeping every %ds (default: one-shot)" % POLL_INTERVAL_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_label_tracker_file(Path(args.label_tracker_path))
    Path(args.ava256_landmark_root).mkdir(parents=True, exist_ok=True)
    Path(args.ava256_output_root).mkdir(parents=True, exist_ok=True)

    root_flags = [
        "--ava256-data-root", args.ava256_data_root,
        "--ava256-landmark-root", args.ava256_landmark_root,
        "--ava256-output-root", args.ava256_output_root,
        "--label-tracker-path", args.label_tracker_path,
        "--ava256-mesh-topology-path", args.ava256_mesh_topology_path,
        "--wrap-script-path", args.wrap_script_path,
        "--template-pot-path", args.template_pot_path,
        "--template-wrap-path", args.template_wrap_path,
        "--faceform-wrap-cmd-path", args.faceform_wrap_cmd_path,
        "--faceform-wrap-license-path", args.faceform_wrap_license_path,
        "--sam-checkpoint-path", args.sam_checkpoint_path,
        "--u2net-checkpoint-path", args.u2net_checkpoint_path,
        "--wrap-script-dir", args.wrap_script_dir,
        "--facescape-mask-path", args.facescape_mask_path,
        "--blendshape-root", args.blendshape_root,
        "--facescape-run-pipeline-path", args.facescape_run_pipeline_path,
    ]
    if args.no_face_ray_masking:
        root_flags.append("--no-face-ray-masking")
    if args.force:
        root_flags.append("--force")

    total_passed = total_failed = 0
    while True:
        passed, failed = run_sweep(args, root_flags)
        total_passed += passed
        total_failed += failed
        print(f"Sweep complete: {passed} passed, {failed} failed")
        if not args.poll:
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    return 2 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
