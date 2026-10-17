#!/usr/bin/env python3
"""Renders one Ava-256 capture's keyline overlay video for review in
FaceView's Ava-256 tab: every wrapped frame of the capture, in capture
(frame_list.csv) order, as front | left | right panels (PANEL_CAMERAS).

Inputs:  WRAPS_ROOT/<CAPTURE>/<FRAME>/wrapped_mesh.obj (template frame) and
         WRAPS_ROOT/<CAPTURE>/dynamic_transform_params.npz (-> world space),
         DATA_ROOT/<CAPTURE>/decoder/... (cameras, AVIF images, frame_list.csv).
Outputs: OUTPUT_ROOT/<CAPTURE>.mp4 -- one video frame per wrap, "#<position>
         <frame id>" burned into the top-left corner;
         OUTPUT_ROOT/<CAPTURE>.csv -- FRAME_ID,STATUS, one row per video frame
         in the same order, STATUS "unconfirmed";
         label_tracker.json Ava256[<CAPTURE>] = "unconfirmed" once done.

Skips a capture that is already rendered (mp4 + csv exist; --force re-renders,
keeping the statuses of frames that are still present), already archived by
FaceView's Done button (LABELS_ROOT/<CAPTURE>.csv), or missing its transform
params / raw data.

Usage:
    python3 render_capture_line_video.py <CAPTURE> [--force]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import camera_classify  # noqa: E402
import camera_utils  # noqa: E402
import dynamic_transform  # noqa: E402
import mesh_utils  # noqa: E402
import render_wrapped_landmarks as rwl  # noqa: E402
from label_tracker_ava256 import Ava256LabelTracker  # noqa: E402

WRAPS_ROOT = Path("/scratch/ondemand32/irwinngo/ava256_killarney_hz")
DATA_ROOT = Path("/scratch/thirty/irwinngo/ava-256")
OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/line_renders_ava256")
LABELS_ROOT = Path("/scratch/ondemand32/irwinngo/ava256-labels")
LABEL_TRACKER_PATH = Path("/scratch/ondemand32/irwinngo/label_tracker.json")
FPS = 24.0
DEFAULT_STATUS = "unconfirmed"
CSV_HEADER = ["FRAME_ID", "STATUS"]
EAR_INDICES = (1, 4, 6, 14)
NOSE_TIP_INDEX = 60
# Panels shown side by side: front | left | right. A capture without the front
# camera uses its most frontal camera there; a missing side camera gets a
# placeholder panel.
PANEL_CAMERAS = (("401469", "front"), ("401071", "left"), ("401466", "right"))
# Where the two side cameras sit in the dome (world frame, mm; from
# 20210810--1306--FXN596's calibration -- the rig frame agrees across captures
# to a few mm). A capture without one of them uses its downloaded camera
# closest to that spot instead (e.g. 400979 / 400953 on captures that only have
# 3 cameras downloaded).
SIDE_CAMERA_CENTERS = {"401071": (470.8, 551.8, 238.2), "401466": (-1037.4, -228.3, 686.8)}


def wrapped_frames_in_order(actor_dir: Path, capture_dir: Path) -> list[str]:
    """Frames with a wrapped_mesh.obj, in frame_list.csv row order; frames not
    listed there go last, in numeric order."""
    wrapped = {p.name for p in capture_dir.iterdir()
               if p.is_dir() and p.name.isdigit() and (p / "wrapped_mesh.obj").exists()}
    ordered: list[str] = []
    frame_list = actor_dir / "decoder" / "frame_list.csv"
    if frame_list.exists():
        with frame_list.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                frame_id = row["frame_id"].strip().zfill(6)
                if frame_id in wrapped and frame_id not in ordered:
                    ordered.append(frame_id)
    listed = set(ordered)
    ordered += sorted((f for f in wrapped if f not in listed), key=int)
    return ordered


def most_frontal_camera(actor_dir: Path, verts_world: np.ndarray, faces: np.ndarray) -> str:
    """Ava-256's front set (classify_front_right_left), then the one looking most
    straight at the face: forward = nose tip minus the ear midpoint."""
    with rwl.DEFAULT_TEMPLATE_POT_LIVE_PATH.open("r", encoding="utf-8") as f:
        pot_rows = dict(enumerate(json.load(f)))
    # Only cameras whose image zip is on disk (captures may be partially downloaded).
    camera_ids = [cid for cid in camera_utils.load_all_camera_ids(actor_dir)
                  if camera_zip(actor_dir, cid).exists() and zipfile.is_zipfile(camera_zip(actor_dir, cid))]
    if not camera_ids:
        raise FileNotFoundError(f"no camera image zips under {actor_dir / 'decoder' / 'image'}")
    camera_params = {cid: camera_utils.load_camera(actor_dir, cid) for cid in camera_ids}
    front, _right, _left = camera_classify.classify_front_right_left(verts_world, faces, pot_rows, camera_ids, camera_params)
    candidates = sorted(front) or sorted(camera_ids)
    ears = np.mean([mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows[i]) for i in EAR_INDICES], axis=0)
    nose = mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows[NOSE_TIP_INDEX])
    forward = (nose - ears) / np.linalg.norm(nose - ears)

    def frontalness(cid: str) -> float:
        to_cam = camera_utils.camera_center_world(camera_params[cid][1]) - ears
        return float(np.dot(to_cam / np.linalg.norm(to_cam), forward))

    return max(candidates, key=frontalness)


def camera_zip(actor_dir: Path, camera_id: str) -> Path:
    return actor_dir / "decoder" / "image" / f"cam{camera_id}.zip"


def read_statuses(csv_path: Path) -> dict[str, str]:
    if not csv_path.exists():
        return {}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return {row["FRAME_ID"]: row["STATUS"] for row in csv.DictReader(f)}


def write_csv(csv_path: Path, rows: list[tuple[str, str]]) -> None:
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)
    tmp.replace(csv_path)


def burn_label(image: np.ndarray, text: str) -> None:
    scale = max(0.5, image.shape[1] / 1200.0)
    thickness = max(1, int(round(scale * 2)))
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(image, (0, 0), (w + 16, h + baseline + 16), (0, 0, 0), -1)
    cv2.putText(image, text, (8, h + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def render_capture(capture_id: str, force: bool = False) -> int:
    mp4_path = OUTPUT_ROOT / f"{capture_id}.mp4"
    csv_path = OUTPUT_ROOT / f"{capture_id}.csv"
    capture_dir = WRAPS_ROOT / capture_id

    if (LABELS_ROOT / f"{capture_id}.csv").exists():
        print(f"SKIP {capture_id}: already labeled and archived in {LABELS_ROOT}")
        return 0
    if mp4_path.exists() and csv_path.exists() and not force:
        print(f"SKIP {capture_id}: already rendered ({mp4_path})")
        return 0
    if not dynamic_transform.params_path(WRAPS_ROOT, capture_id).exists():
        print(f"SKIP {capture_id}: no dynamic_transform_params.npz -- wraps can't be put back in world space")
        return 0
    try:
        actor_dir = mesh_utils.resolve_capture_dir(DATA_ROOT, capture_id)
    except FileNotFoundError as exc:
        print(f"SKIP {capture_id}: no raw data yet ({exc})")
        return 0

    calibration = actor_dir / "decoder" / "camera_calibration.json"
    if not calibration.exists() or calibration.stat().st_size == 0:
        print(f"SKIP {capture_id}: raw data incomplete (decoder/camera_calibration.json missing or empty)")
        return 0

    frame_ids = wrapped_frames_in_order(actor_dir, capture_dir)
    if not frame_ids:
        print(f"SKIP {capture_id}: no wrapped frames")
        return 0

    verts_world, faces = rwl.load_wrapped_mesh_world(WRAPS_ROOT, capture_id, frame_ids[0])
    # Downloaded cameras whose image zip is readable (a truncated download fails every frame).
    have = {cid for cid in camera_utils.load_all_camera_ids(actor_dir)
            if camera_zip(actor_dir, cid).exists() and zipfile.is_zipfile(camera_zip(actor_dir, cid))}
    broken = sorted(cid for cid in camera_utils.load_all_camera_ids(actor_dir)
                    if camera_zip(actor_dir, cid).exists() and cid not in have)
    if broken:
        print(f"NOTE {capture_id}: unreadable camera zip(s) skipped: {broken}")
    panels: list[tuple[str | None, str]] = []
    for cid, view in PANEL_CAMERAS:
        if cid not in have and view == "front":
            try:
                cid = most_frontal_camera(actor_dir, verts_world, faces)
            except FileNotFoundError as exc:
                print(f"SKIP {capture_id}: raw data incomplete ({exc})")
                return 0
        if cid not in have and cid in SIDE_CAMERA_CENTERS:
            used = {c for c, _ in panels if c is not None} | {c for c, _ in PANEL_CAMERAS if c in have}
            options = sorted(have - used)
            if options:
                target = np.asarray(SIDE_CAMERA_CENTERS[cid])
                cid = min(options, key=lambda c: float(np.linalg.norm(
                    camera_utils.camera_center_world(camera_utils.load_camera(actor_dir, c)[1]) - target)))
        panels.append((cid if cid in have else None, view))
    cams = {cid: camera_utils.load_camera(actor_dir, cid) for cid, _ in panels if cid is not None}
    print(f"{capture_id}: {len(frame_ids)} wrapped frame(s), cameras {[c or '-' for c, _ in panels]}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    temp_mp4 = mp4_path.with_name(f"{capture_id}_tmp.mp4")
    previous = read_statuses(csv_path) if force else {}
    rows: list[tuple[str, str]] = []
    writer: cv2.VideoWriter | None = None
    size: tuple[int, int] | None = None
    try:
        for frame_id in frame_ids:
            try:
                verts_world, faces = rwl.load_wrapped_mesh_world(WRAPS_ROOT, capture_id, frame_id)
                tiles = []
                for (cid, view), want in zip(panels, PANEL_CAMERAS):
                    if cid is None:
                        tiles.append(None)
                        continue
                    K, Rt = cams[cid]
                    try:
                        image = camera_utils.load_image(actor_dir, cid, frame_id)
                    except (KeyError, FileNotFoundError, OSError, ValueError, zipfile.BadZipFile) as exc:
                        # This camera has no image for this frame: blank this panel only.
                        print(f"  frame {frame_id}: no image from cam {cid} ({exc}) -- placeholder panel")
                        tiles.append(None)
                        continue
                    tile = rwl.render_keyline_image(verts_world, faces, image, K, Rt)
                    tiles.append((tile, f"cam {cid} ({view})" + ("" if cid == want[0] else f", no {want[0]}")))
            except (KeyError, FileNotFoundError, OSError, ValueError, zipfile.BadZipFile) as exc:
                print(f"  skip frame {frame_id}: {exc}")
                continue
            if all(t is None for t in tiles):
                print(f"  skip frame {frame_id}: no camera has an image for it")
                continue
            ref = next(t[0] for t in tiles if t is not None)
            h, w = ref.shape[:2]
            w -= w % 2  # even width/height for libx264 yuv420p
            h -= h % 2
            row = []
            for t, (want_id, view) in zip(tiles, PANEL_CAMERAS):
                if t is None:
                    tile = np.zeros((h, w, 3), np.uint8)
                    cv2.putText(tile, f"cam {want_id} ({view}): no image", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 255), 2, cv2.LINE_AA)
                else:
                    tile = cv2.resize(t[0], (w, h)) if t[0].shape[:2] != (h, w) else t[0].copy()
                    cv2.rectangle(tile, (0, h - 40), (w, h), (0, 0, 0), -1)
                    cv2.putText(tile, t[1], (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                row.append(tile)
            frame = np.hstack(row)
            if writer is None:
                size = (frame.shape[1], frame.shape[0])
                writer = cv2.VideoWriter(str(temp_mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
            elif (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            burn_label(frame, f"#{len(rows) + 1}  {frame_id}")
            writer.write(frame)
            rows.append((frame_id, previous.get(frame_id, DEFAULT_STATUS)))
    finally:
        if writer is not None:
            writer.release()

    if not rows:
        temp_mp4.unlink(missing_ok=True)
        print(f"FAIL {capture_id}: no frame could be rendered")
        return 1

    rwl.finalize_mp4(temp_mp4, mp4_path)
    write_csv(csv_path, rows)
    Ava256LabelTracker(LABEL_TRACKER_PATH).set_status(capture_id, "unconfirmed")
    print(f"DONE {capture_id}: {len(rows)} frame(s) -> {mp4_path}, {csv_path}; label_tracker -> unconfirmed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_id")
    parser.add_argument("--force", action="store_true", help="Re-render even if the mp4/csv exist (keeps existing frame statuses)")
    args = parser.parse_args()
    return render_capture(args.capture_id, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
