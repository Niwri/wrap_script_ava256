#!/usr/bin/env python3
"""Ava-256 counterpart to wrap_script_facescape/check_skin_landmark_errors.py:
per-frame mean skin-landmark error of each wrap, and flags frames above a
threshold as "Invalid" in the capture's review CSV (FaceView's Ava-256 tab).

For every frame listed in line_renders_ava256/<capture>.csv:
  - target points: the frame's target_correspondence_filtered.json (where the
    wrap was asked to put each landmark),
  - wrap points:   the frame's neutral_correspondence_filtered.json POT rows
    evaluated on its wrapped_mesh.obj (where the wrap actually put them),
  - skin rows only (original correspondence index >= 74, recovered from the
    non-empty rows of target_correspondence.json, which line up 1:1 with the
    filtered files),
  - both taken back to world space (capture's dynamic_transform_params.npz)
    and projected into the capture's most frontal camera (images are 667 px
    wide); error = mean 2D distance in pixels. The 3D mean (world units) is
    recorded too.

Chin check: the wrap's chin (standard landmark 50, its POT row on the
wrapped_mesh.obj, back in world space) vs. that frame's keypoints_3d chin
(keypoint id landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID[50]), projected into
the same camera; frames where keypoints_3d has no chin are not chin-checked.

Writes line_renders_ava256/skin_errors/<capture>.json:
    {"camera": ..., "threshold_px": ..., "chin_threshold_px": ...,
     "frames": {frame_id: {"px": .., "world": .., "n": .., "chin_px": ..}}}
which FaceView's Ava-256 tab shows per frame. With --apply, frames whose mean
skin error exceeds --threshold OR whose chin is more than --chin-threshold
(default 20 px) from the keypoints_3d chin, and whose CSV status is still
"unconfirmed", are set to "Invalid" (under the same per-capture lock FaceView's backend uses). Labels
you already set (Valid/Invalid/Rerun) are never overwritten, and captures
already finished with Done (archived CSV) are skipped.

Usage:
    python3 check_ava256_skin_landmark_errors.py [--threshold 4.0] [--chin-threshold 20] [--captures ...] [--parallel 16] [--apply]
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import camera_classify  # noqa: E402
import camera_utils  # noqa: E402
import dynamic_transform  # noqa: E402
import landmark_map  # noqa: E402
import mesh_utils  # noqa: E402
from render_capture_line_video import EAR_INDICES, NOSE_TIP_INDEX  # noqa: E402

WRAPS_ROOT = Path("/scratch/ondemand32/irwinngo/ava256_killarney_hz")
DATA_ROOT = Path("/scratch/thirty/irwinngo/ava-256")
REVIEW_ROOT = Path("/scratch/ondemand32/irwinngo/line_renders_ava256")  # <capture>.csv, FaceView's line_renders_ava256_root
ERRORS_DIR = REVIEW_ROOT / "skin_errors"
LOCK_DIR = REVIEW_ROOT / ".locks"  # same per-capture lock as FaceView's services/ava256_labels.py
DEFAULT_THRESHOLD_PX = 4.0
DEFAULT_CHIN_THRESHOLD_PX = 20.0
CHIN_INDEX = 50  # standard chin landmark
CHIN_KEYPOINT3D_ID = landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID[CHIN_INDEX]
NUM_STANDARD_LANDMARKS = 74


@contextmanager
def capture_lock(capture_id: str):
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    with (LOCK_DIR / f"{capture_id}.lock").open("w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def read_csv(path: Path) -> list[tuple[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [(row["FRAME_ID"], row["STATUS"]) for row in csv.DictReader(f)]


def write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["FRAME_ID", "STATUS"])
        writer.writerows(rows)
    tmp.replace(path)


def frontal_camera(actor_dir: Path, verts_world: np.ndarray, faces: np.ndarray, pot_rows: dict) -> tuple[str, np.ndarray, np.ndarray]:
    """Most frontal calibrated camera (no image needed): the front set from
    classify_front_right_left, then nose-tip-minus-ear-midpoint alignment."""
    ids = camera_utils.load_all_camera_ids(actor_dir)
    params = {c: camera_utils.load_camera(actor_dir, c) for c in ids}
    front, _, _ = camera_classify.classify_front_right_left(verts_world, faces, pot_rows, ids, params)
    ears = np.mean([mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows[i]) for i in EAR_INDICES], axis=0)
    nose = mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows[NOSE_TIP_INDEX])
    fwd = (nose - ears) / np.linalg.norm(nose - ears)

    def frontalness(c):
        v = camera_utils.camera_center_world(params[c][1]) - ears
        return float(np.dot(v / np.linalg.norm(v), fwd))

    cam = max(sorted(front) or sorted(ids), key=frontalness)
    return cam, *params[cam]


def chin_error(actor_dir: Path, frame_id: str, V: np.ndarray, F: np.ndarray, pot_rows: dict, params: dict,
               K: np.ndarray, Rt: np.ndarray) -> float | None:
    """Pixels between the wrap's chin and the frame's keypoints_3d chin (None if keypoints_3d has no chin)."""
    kp = mesh_utils.keypoints_3d_by_id(actor_dir, frame_id).get(CHIN_KEYPOINT3D_ID)
    if kp is None:
        return None
    wrap_chin = dynamic_transform.untransform(mesh_utils.pot_row_world_xyz(V, F, pot_rows[CHIN_INDEX])[None], params)
    uv = camera_utils.project_points(np.vstack([wrap_chin, np.asarray(kp, dtype=np.float64)[None]]), K, Rt)
    return float(np.linalg.norm(uv[0] - uv[1]))


def frame_error(frame_dir: Path, params: dict, K: np.ndarray, Rt: np.ndarray, actor_dir: Path, pot_rows: dict) -> dict | None:
    target = json.loads((frame_dir / "target_correspondence_filtered.json").read_text())
    neutral = json.loads((frame_dir / "neutral_correspondence_filtered.json").read_text())
    full = json.loads((frame_dir / "target_correspondence.json").read_text())
    kept = [i for i, p in enumerate(full) if any(float(v) != 0.0 for v in p.values())]
    if not (len(target) == len(neutral) == len(kept)):
        raise ValueError(f"correspondence files don't line up ({len(target)}/{len(neutral)}/{len(kept)})")
    skin = [j for j, i in enumerate(kept) if i >= NUM_STANDARD_LANDMARKS]
    mesh = trimesh.load(str(frame_dir / "wrapped_mesh.obj"), process=False)
    V, F = np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces)
    chin = chin_error(actor_dir, frame_dir.name, V, F, pot_rows, params, K, Rt)
    out = {} if chin is None else {"chin_px": round(chin, 3)}
    if not skin:
        return out or None
    tgt = np.array([[target[j]["x"], target[j]["y"], target[j]["z"]] for j in skin], dtype=np.float64)
    src = np.array([mesh_utils.pot_row_world_xyz(V, F, neutral[j]) for j in skin])
    tgt_w, src_w = dynamic_transform.untransform(tgt, params), dynamic_transform.untransform(src, params)
    px = np.linalg.norm(camera_utils.project_points(tgt_w, K, Rt) - camera_utils.project_points(src_w, K, Rt), axis=1)
    return {"px": round(float(px.mean()), 3), "world": round(float(np.linalg.norm(tgt_w - src_w, axis=1).mean()), 3),
            "n": len(skin), **out}


def process_capture(capture_id: str, threshold: float, chin_threshold: float, apply: bool, pot_rows: dict) -> dict:
    review_csv = REVIEW_ROOT / f"{capture_id}.csv"
    actor_dir = DATA_ROOT / capture_id
    params = dynamic_transform.load_params(dynamic_transform.params_path(WRAPS_ROOT, capture_id))
    rows = read_csv(review_csv)
    if not rows:
        return {"capture": capture_id, "frames": 0}
    first = trimesh.load(str(WRAPS_ROOT / capture_id / rows[0][0] / "wrapped_mesh.obj"), process=False)
    verts0 = dynamic_transform.untransform(np.asarray(first.vertices, dtype=np.float64), params)
    camera_id, K, Rt = frontal_camera(actor_dir, verts0, np.asarray(first.faces), pot_rows)

    errors: dict[str, dict] = {}
    failed = 0
    for frame_id, _status in rows:
        try:
            err = frame_error(WRAPS_ROOT / capture_id / frame_id, params, K, Rt, actor_dir, pot_rows)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            failed += 1
            continue
        if err is not None:
            errors[frame_id] = err

    ERRORS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = ERRORS_DIR / f"{capture_id}.json.tmp"
    tmp.write_text(json.dumps({"camera": camera_id, "threshold_px": threshold, "chin_threshold_px": chin_threshold,
                               "frames": errors}))
    tmp.replace(ERRORS_DIR / f"{capture_id}.json")

    skin_over = {f for f, e in errors.items() if e.get("px", 0.0) > threshold}
    chin_over = {f for f, e in errors.items() if e.get("chin_px", 0.0) > chin_threshold}
    over = skin_over | chin_over
    flagged = kept_label = 0
    if apply and over:
        with capture_lock(capture_id):
            if review_csv.exists():  # not moved away by Done meanwhile
                current = read_csv(review_csv)
                updated = []
                for frame_id, status in current:
                    if frame_id in over and status == "unconfirmed":
                        updated.append((frame_id, "Invalid"))
                        flagged += 1
                    else:
                        if frame_id in over:
                            kept_label += 1
                        updated.append((frame_id, status))
                write_csv(review_csv, updated)
    return {"capture": capture_id, "camera": camera_id, "frames": len(rows), "measured": len(errors), "failed": failed,
            "over": len(over), "skin_over": len(skin_over), "chin_over": len(chin_over),
            "chin_checked": sum(1 for e in errors.values() if "chin_px" in e),
            "flagged": flagged, "kept_existing_label": kept_label,
            "median_px": float(np.median([e["px"] for e in errors.values() if "px" in e]))
            if any("px" in e for e in errors.values()) else None,
            "median_chin_px": float(np.median([e["chin_px"] for e in errors.values() if "chin_px" in e]))
            if any("chin_px" in e for e in errors.values()) else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PX,
                        help=f"Mean skin landmark pixel error above which a frame is flagged (default {DEFAULT_THRESHOLD_PX})")
    parser.add_argument("--chin-threshold", type=float, default=DEFAULT_CHIN_THRESHOLD_PX,
                        help=f"Chin-vs-keypoints_3d-chin pixel distance above which a frame is flagged (default {DEFAULT_CHIN_THRESHOLD_PX})")
    parser.add_argument("--captures", nargs="*", default=None, help="Only these captures (default: every capture in review)")
    parser.add_argument("--parallel", type=int, default=16)
    parser.add_argument("--apply", action="store_true", help="Set flagged, still-unconfirmed frames to Invalid in the CSV")
    args = parser.parse_args()

    captures = sorted(p.stem for p in REVIEW_ROOT.glob("*.csv") if "--" in p.stem)
    if args.captures:
        captures = [c for c in captures if c in set(args.captures)]
    print(f"{len(captures)} capture(s) in review (Done/archived captures are skipped); threshold {args.threshold}px, chin threshold {args.chin_threshold}px; "
          f"{'APPLY' if args.apply else 'dry run -- errors saved, CSVs unchanged'}")
    with (SCRIPT_DIR / "neutral_correspondence_without_eyes_POT_all_live.json").open() as f:
        pot_rows = dict(enumerate(json.load(f)))

    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as pool:
        futures = {pool.submit(process_capture, c, args.threshold, args.chin_threshold, args.apply, pot_rows): c for c in captures}
        for future in as_completed(futures):
            capture_id = futures[future]
            try:
                r = future.result()
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {capture_id}: {exc!r}", flush=True)
                continue
            results.append(r)
            if r.get("over"):
                med = "n/a" if r["median_px"] is None else f"{r['median_px']:.2f}px"
                chin_med = "n/a" if r["median_chin_px"] is None else f"{r['median_chin_px']:.2f}px"
                print(f"{capture_id}: {r['over']}/{r['measured']} frame(s) flagged -- skin > {args.threshold}px: {r['skin_over']}, "
                      f"chin > {args.chin_threshold}px: {r['chin_over']} of {r['chin_checked']} checked "
                      f"(median skin {med}, chin {chin_med}, cam {r['camera']})"
                      + (f"; set {r['flagged']} to Invalid, left {r['kept_existing_label']} already-labeled" if args.apply else ""),
                      flush=True)

    total = sum(r.get("measured", 0) for r in results)
    over = sum(r.get("over", 0) for r in results)
    print(f"\n{len(results)} capture(s), {total} frame(s) measured, {over} flagged "
          f"({sum(r.get('skin_over', 0) for r in results)} skin > {args.threshold}px, "
          f"{sum(r.get('chin_over', 0) for r in results)} chin > {args.chin_threshold}px of "
          f"{sum(r.get('chin_checked', 0) for r in results)} chin-checked) "
          f"({sum(r.get('failed', 0) for r in results)} frame(s) couldn't be measured)"
          + (f"; {sum(r.get('flagged', 0) for r in results)} set to Invalid" if args.apply else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
