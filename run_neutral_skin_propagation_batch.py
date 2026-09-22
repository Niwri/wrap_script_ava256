#!/usr/bin/env python3
"""Batch/sweep wrapper for run_neutral_skin_propagation.py. Mirrors
run_pipeline_batch.py's mechanics, gated on "unlabeled" status (the implicit
default for any capture with no label_tracker entry yet) instead of
"unreviewed".

Usage:
    python3 run_neutral_skin_propagation_batch.py [--captures ID ...] [--parallel N] [--gpu-ids 0 1 ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROPAGATION_SCRIPT = SCRIPT_DIR / "run_neutral_skin_propagation.py"
sys.path.insert(0, str(SCRIPT_DIR))

from label_tracker_ava256 import Ava256LabelTracker, ensure_label_tracker_file
from run_neutral_skin_propagation import DEFAULTS

POLL_INTERVAL_SECONDS = 60


@dataclass
class CaptureResult:
    capture_id: str
    passed: bool


def discover_unlabeled_captures(ava256_data_root: Path, label_tracker_path: Path) -> list[str]:
    tracker = Ava256LabelTracker(label_tracker_path)
    all_statuses = tracker.get_all()
    captures = sorted(p.name for p in Path(ava256_data_root).iterdir() if p.is_dir())
    return [
        c for c in captures
        if tracker.get_status(c, all_statuses=all_statuses) == "unlabeled"
    ]


def process_capture(capture_id: str, extra_flags: list[str], log_dir: Path | None, gpu_id: str | None, dry_run: bool) -> CaptureResult:
    tag = f"[capture={capture_id}]" + (f" gpu={gpu_id}" if gpu_id is not None else "")
    cmd = [sys.executable, str(PROPAGATION_SCRIPT), capture_id, *extra_flags]
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


def run_sweep(args, extra_flags: list[str]) -> tuple[int, int]:
    if args.captures:
        captures = args.captures
    else:
        captures = discover_unlabeled_captures(Path(args.ava256_data_root), Path(args.label_tracker_path))

    if not captures:
        print("No captures to process.")
        return 0, 0

    passed = failed = 0
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as executor:
        futures = {
            executor.submit(
                process_capture, capture_id, extra_flags,
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
            else:
                failed += 1
    return passed, failed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--captures", nargs="+", default=None, help="Explicit capture ids; omit to sweep every capture at 'unlabeled' status")
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
    parser.add_argument("--force", action="store_true",
                         help="Forward run_neutral_skin_propagation.py's --force to every capture -- re-propagate "
                              "even if label_tracker.json already shows it done")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="+", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--poll", action="store_true", help="Loop forever, re-sweeping every %ds (default: one-shot)" % POLL_INTERVAL_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ensure_label_tracker_file(Path(args.label_tracker_path))
    Path(args.ava256_landmark_root).mkdir(parents=True, exist_ok=True)
    Path(args.ava256_output_root).mkdir(parents=True, exist_ok=True)

    extra_flags = [
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
        "--triangulate-script-path", args.triangulate_script_path,
        "--cotracker-root", args.cotracker_root,
        "--cotracker-checkpoint-path", args.cotracker_checkpoint_path,
        "--raft-checkpoint-path", args.raft_checkpoint_path,
        "--min-cameras-per-landmark", str(args.min_cameras_per_landmark),
        "--angle-threshold-deg", str(args.angle_threshold_deg),
    ]
    if args.force:
        extra_flags.append("--force")

    total_passed = total_failed = 0
    while True:
        passed, failed = run_sweep(args, extra_flags)
        total_passed += passed
        total_failed += failed
        print(f"Sweep complete: {passed} passed, {failed} failed")
        if not args.poll:
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    return 2 if total_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
