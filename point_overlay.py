#!/usr/bin/env python3
"""Ava-256 landmark "point overlay" render: the annotated (propagated) 3D
landmarks in landmarks_ava-256/<capture>/<group>_<frame>.json vs. the same
landmarks read off that frame's wrapped mesh, both projected into one
camera's photo.

Mirrors wrap_script/point_overlay.py's look (kept as a local copy -- this
directory is deliberately self-contained, see HANDOFF.md): annotated = filled
dot in FaceView's Cameras-tab region color (skin = slate); wrapped mesh =
hollow ring in the same hue blended toward white; a thin gray line joins each
pair; top-left legend with the frame's mean pixel error.

- Mesh points: neutral_correspondence_without_eyes_POT_all_live.json's row for
  each index (mesh_utils.pot_row_world_xyz) on the wrapped mesh, undone to
  world space (render_wrapped_landmarks.load_wrapped_mesh_world).
- Only indices landmark_map.used_landmark_indices() says the wrap actually
  sources are drawn.
- Annotated eyelid/lip points (AVA256_STANDARD_INDEX_REGION_GROUPS["eyelid_lip"])
  come from decoder/keypoints_3d, like the wrap's own targets; every other
  annotated point comes from the propagated group JSONs.
- Camera: unless given, the most frontal of the capture's front-facing
  cameras (camera_classify.classify_front_right_left on the neutral
  frame's wrap, as run_pipeline_batch.py's keyline video does -- but that
  takes sorted(front)[0], which can be a steep below-the-chin view).

Output: /scratch/ondemand32/irwinngo/point_overlay_renders_ava256/
    <capture>_<frame>_<camera>.jpg        (single frame)
    <capture>_<camera>.mp4                (--video: every wrapped frame)

Usage:
    python3 point_overlay.py <CAPTURE_ID> <FRAME_ID> [CAMERA_ID]
    python3 point_overlay.py <CAPTURE_ID> --video [CAMERA_ID]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import camera_utils  # noqa: E402
import landmark_map  # noqa: E402
import mesh_utils  # noqa: E402
import render_wrapped_landmarks as rwl  # noqa: E402

DEFAULT_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/point_overlay_renders_ava256")

# FaceView/frontend/src/components/LandmarkOverlay.tsx's region groups and
# colors (RGB), same table as wrap_script/render_landmarks.py's REGION_COLORS.
_REGION_COLORS: list[tuple[list[int], tuple[int, int, int]]] = [
    ([31, 28, 12, 53, 48, 49, 52, 51, 50, 66, 67, 64, 63, 68, 3, 7, 10], (255, 0, 0)),  # jaw
    ([2, 4, 1], (0, 255, 0)),              # left ear
    ([9, 14, 6], (0, 0, 255)),             # right ear
    ([8, 30, 0, 73, 72], (255, 165, 0)),   # left brow
    ([29, 47, 5, 70, 69], (128, 0, 128)),  # right brow
    ([60, 62, 54, 57], (255, 255, 0)),     # nose
    ([71, 16, 13, 11], (0, 255, 255)),     # left eye
    ([35, 33, 32, 65], (255, 0, 255)),     # right eye
    ([34, 36], (200, 0, 200)),             # right eye up
    ([15, 17], (0, 200, 200)),             # left eye up
    ([38, 41, 61, 19, 22], (0, 128, 0)),   # under nose
    ([37, 43], (128, 0, 0)),               # right mouth corner
    ([18, 24], (0, 0, 128)),               # left mouth corner
    ([42, 44], (255, 105, 180)),           # outer mouth right
    ([23, 25], (0, 128, 128)),             # outer mouth left
    ([39, 20, 46, 27], (210, 105, 30)),    # outer mouth middle outer
    ([40, 21, 45, 26], (75, 0, 130)),      # inner mouth middle outer
    ([59, 55], (154, 205, 50)),            # outer mouth middle middle
    ([56, 58], (70, 130, 180)),            # inner mouth middle middle
]
INDEX_TO_COLOR: dict[int, tuple[int, int, int]] = {i: c for idxs, c in _REGION_COLORS for i in idxs}
SKIN_COLOR = (148, 163, 184)  # slate
LINE_COLOR = (170, 170, 170)
MESH_TINT = 0.5


def _color_bgr(index: int) -> tuple[int, int, int]:
    r, g, b = SKIN_COLOR if index >= landmark_map.NUM_STANDARD_LANDMARKS else INDEX_TO_COLOR.get(index, SKIN_COLOR)
    return (b, g, r)


def _tint(bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(int(round(c + (255 - c) * MESH_TINT)) for c in bgr)


def draw_point_overlay(
    image: np.ndarray, annotated: dict[int, np.ndarray], mesh_px: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[int, float]]:
    """Same drawing as wrap_script/point_overlay.draw_point_overlay(); returns
    (image, {index: pixel error})."""
    out = image.copy()
    r = max(2, int(round(min(out.shape[:2]) / 200)))
    h, w = out.shape[:2]
    indices = sorted(
        i for i in annotated
        if i in mesh_px and np.all(np.isfinite(annotated[i])) and np.all(np.isfinite(mesh_px[i]))
        and -w < annotated[i][0] < 2 * w and -h < annotated[i][1] < 2 * h
    )
    shift, scale = 4, 16

    def pt(p: np.ndarray) -> tuple[int, int]:
        return int(round(p[0] * scale)), int(round(p[1] * scale))

    ring_thickness = 1 if r < 5 else 2
    for i in indices:
        cv2.line(out, pt(annotated[i]), pt(mesh_px[i]), LINE_COLOR, 1, cv2.LINE_AA, shift)
    for i in indices:
        cv2.circle(out, pt(mesh_px[i]), (r + 2) * scale, _tint(_color_bgr(i)), ring_thickness, cv2.LINE_AA, shift)
    for i in indices:
        cv2.circle(out, pt(annotated[i]), (r + 1) * scale, (20, 20, 20), -1, cv2.LINE_AA, shift)
        cv2.circle(out, pt(annotated[i]), r * scale, _color_bgr(i), -1, cv2.LINE_AA, shift)

    errors = {i: float(np.linalg.norm(annotated[i] - mesh_px[i])) for i in indices}
    std = [e for i, e in errors.items() if i < landmark_map.NUM_STANDARD_LANDMARKS]
    skin = [e for i, e in errors.items() if i >= landmark_map.NUM_STANDARD_LANDMARKS]
    fmt = lambda v: f"{np.mean(v):.1f}px" if v else "-"  # noqa: E731

    font, fscale = cv2.FONT_HERSHEY_SIMPLEX, max(0.4, w / 1400)
    lines = ["annotated", "wrapped mesh", f"mean err  std {fmt(std)}  skin {fmt(skin)}"]
    line_h, box_w = int(22 * fscale / 0.4), int(250 * fscale / 0.4)
    overlay = out.copy()
    cv2.rectangle(overlay, (4, 4), (4 + box_w, 8 + line_h * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, dst=out)
    for row, text in enumerate(lines):
        y = 4 + line_h * (row + 1) - line_h // 4
        cx, cy = 14 + r, y - line_h // 4
        if row == 0:
            cv2.circle(out, (cx, cy), r, (255, 255, 255), -1, cv2.LINE_AA)
        elif row == 1:
            cv2.circle(out, (cx, cy), r + 2, (255, 255, 255), ring_thickness, cv2.LINE_AA)
        cv2.putText(out, text, (14 + 2 * r + 8 if row < 2 else 10, y), font, fscale, (255, 255, 255), 1, cv2.LINE_AA)
    return out, errors


def split_errors(errors: dict[int, float]) -> tuple[float | None, float | None]:
    """(mean error over standard indices, mean error over skin indices)."""
    std = [e for i, e in errors.items() if i < landmark_map.NUM_STANDARD_LANDMARKS]
    skin = [e for i, e in errors.items() if i >= landmark_map.NUM_STANDARD_LANDMARKS]
    return (float(np.mean(std)) if std else None), (float(np.mean(skin)) if skin else None)


def _load_pot_rows() -> list[list[float]]:
    with rwl.DEFAULT_TEMPLATE_POT_LIVE_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def front_camera_id(capture_id: str, *, ava256_data_root: Path, ava256_output_root: Path, ava256_landmark_root: Path) -> str:
    """Front-camera set as in run_pipeline_batch.render_capture_keyline_video(),
    then the most frontal one of that set."""
    import camera_classify  # torch-free (face_ray_masking_ava256 imports torch)
    import neutral_frame as neutral_frame_module

    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    neutral = neutral_frame_module.resolve_neutral_frame(capture_id, actor_dir, ava256_landmark_root)
    verts_world, faces = rwl.load_wrapped_mesh_world(ava256_output_root, capture_id, neutral["frame_id"])
    pot_rows_by_index = dict(enumerate(_load_pot_rows()))
    camera_ids = camera_utils.load_all_camera_ids(actor_dir)
    camera_params = {cid: camera_utils.load_camera(actor_dir, cid) for cid in camera_ids}
    front, _right, _left = camera_classify.classify_front_right_left(
        verts_world, faces, pot_rows_by_index, camera_ids, camera_params,
    )
    if not front:
        raise RuntimeError(f"No front-facing camera derived for {capture_id}")
    # Among the front cameras, the one looking most straight at the face:
    # forward = nose (60) minus the midpoint of the ear points (1, 4, 6, 14).
    ears = np.mean([mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows_by_index[i]) for i in (1, 4, 6, 14)], axis=0)
    nose = mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows_by_index[60])
    forward = (nose - ears) / np.linalg.norm(nose - ears)

    def frontalness(cid: str) -> float:
        to_cam = camera_utils.camera_center_world(camera_params[cid][1]) - ears
        return float(np.dot(to_cam / np.linalg.norm(to_cam), forward))

    return max(sorted(front), key=frontalness)


def render_point_overlay_image(
    capture_id: str, frame_id: str, camera_id: str, *,
    ava256_data_root: Path, ava256_output_root: Path, ava256_landmark_root: Path,
    image: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[int, float]]:
    """(rendered BGR image, {index: pixel error}) for one frame -- no disk write."""
    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    K, Rt = camera_utils.load_camera(actor_dir, camera_id)
    if image is None:
        image = camera_utils.load_image(actor_dir, camera_id, frame_id)

    pot_rows = _load_pot_rows()
    verts_world, faces = rwl.load_wrapped_mesh_world(ava256_output_root, capture_id, frame_id)
    used = [i for i in landmark_map.used_landmark_indices(len(pot_rows)) if i < len(pot_rows)]
    mesh_xyz = np.array([mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows[i]) for i in used])
    mesh_px = dict(zip(used, camera_utils.project_points(mesh_xyz, K, Rt)))

    # Same target sources as run_pipeline._build_propagated_target_correspondence():
    # eyelid/lip indices from keypoints_3d (dropped when keypoints_3d lacks them),
    # every other index from the propagated <group>_<frame>.json files.
    eyelid_lip = set(landmark_map.AVA256_STANDARD_INDEX_REGION_GROUPS["eyelid_lip"])
    used_set = set(used)
    xyz_by_index: dict[int, np.ndarray] = {
        p["index"]: np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
        for p in rwl.load_frame_points(ava256_landmark_root, capture_id, frame_id)
        if p["index"] in used_set and p["index"] not in eyelid_lip
    }
    kp_by_id = mesh_utils.keypoints_3d_by_id(actor_dir, frame_id)
    for idx in eyelid_lip & used_set:
        kp_id = landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID.get(idx)
        if kp_id is not None and kp_id in kp_by_id:
            xyz_by_index[idx] = np.asarray(kp_by_id[kp_id], dtype=np.float64)
    annotated: dict[int, np.ndarray] = {}
    if xyz_by_index:
        indices = list(xyz_by_index)
        annotated = dict(zip(indices, camera_utils.project_points(np.stack([xyz_by_index[i] for i in indices]), K, Rt)))

    return draw_point_overlay(image, annotated, mesh_px)


def render_point_overlay_frame(
    capture_id: str, frame_id: str, camera_id: str, *,
    ava256_data_root: Path, ava256_output_root: Path, ava256_landmark_root: Path, output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> tuple[Path, float | None, float | None]:
    """Writes the image; returns (path, mean standard error, mean skin error) in pixels."""
    rendered, errors = render_point_overlay_image(
        capture_id, frame_id, camera_id,
        ava256_data_root=ava256_data_root, ava256_output_root=ava256_output_root, ava256_landmark_root=ava256_landmark_root,
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{capture_id}_{frame_id}_{camera_id}.jpg"
    cv2.imwrite(str(output_path), rendered)
    return (output_path, *split_errors(errors))


def render_point_overlay_video(
    capture_id: str, frame_ids: list[str], camera_id: str, *,
    ava256_data_root: Path, ava256_output_root: Path, ava256_landmark_root: Path, output_root: Path = DEFAULT_OUTPUT_ROOT,
    fps: float = rwl.DEFAULT_KEYLINE_VIDEO_FPS,
) -> Path | None:
    """One video frame per wrapped frame, in the given order -- same writer /
    ffmpeg re-encode as render_wrapped_landmarks.render_keyline_video()."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{capture_id}_{camera_id}.mp4"
    temp_output_path = output_path.with_name(f"{output_path.stem}_tmp.mp4")
    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)

    writer: cv2.VideoWriter | None = None
    written = 0
    try:
        for frame_id in frame_ids:
            try:
                image = camera_utils.load_image(actor_dir, camera_id, frame_id)
            except (KeyError, FileNotFoundError) as exc:
                print(f"SKIP {capture_id}/{frame_id}: no image for cam {camera_id} ({exc})")
                continue
            rendered, _err = render_point_overlay_image(
                capture_id, frame_id, camera_id, image=image,
                ava256_data_root=ava256_data_root, ava256_output_root=ava256_output_root, ava256_landmark_root=ava256_landmark_root,
            )
            if writer is None:
                h, w = rendered.shape[:2]
                writer = cv2.VideoWriter(str(temp_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(rendered)
            written += 1
    finally:
        if writer is not None:
            writer.release()

    if written == 0:
        temp_output_path.unlink(missing_ok=True)
        print(f"No usable frames for {capture_id}/{camera_id} point overlay video -- nothing written")
        return None
    rwl.finalize_mp4(temp_output_path, output_path)
    print(f"{capture_id}/{camera_id}: wrote {written}-frame point overlay video -> {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_id")
    parser.add_argument("frame_id", nargs="?", default=None, help="Omit with --video")
    parser.add_argument("camera_id", nargs="?", default=None, help="Default: the capture's most frontal camera")
    parser.add_argument("--video", action="store_true", help="Render every wrapped frame of the capture into one video")
    parser.add_argument("--ava256-data-root", default=str(rwl.DEFAULT_AVA256_DATA_ROOT))
    parser.add_argument("--ava256-landmark-root", default=str(rwl.DEFAULT_AVA256_LANDMARK_ROOT))
    parser.add_argument("--ava256-output-root", default=str(rwl.DEFAULT_AVA256_OUTPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()
    roots = dict(
        ava256_data_root=Path(args.ava256_data_root),
        ava256_output_root=Path(args.ava256_output_root),
        ava256_landmark_root=Path(args.ava256_landmark_root),
    )
    if args.video and args.frame_id is not None and args.camera_id is None:
        args.camera_id, args.frame_id = args.frame_id, None  # `<capture> --video <camera>`
    camera_id = args.camera_id or front_camera_id(args.capture_id, **roots)

    if args.video:
        import neutral_frame as neutral_frame_module
        actor_dir = mesh_utils.resolve_capture_dir(roots["ava256_data_root"], args.capture_id)
        neutral = neutral_frame_module.resolve_neutral_frame(args.capture_id, actor_dir, roots["ava256_landmark_root"])
        frame_ids = rwl.list_wrapped_frames_in_capture_order(
            actor_dir, roots["ava256_output_root"], args.capture_id, neutral_frame_id=neutral["frame_id"],
        )
        render_point_overlay_video(args.capture_id, frame_ids, camera_id, output_root=Path(args.output_root), **roots)
        return 0
    if args.frame_id is None:
        parser.error("frame_id is required unless --video is given")
    path, std_err, skin_err = render_point_overlay_frame(
        args.capture_id, args.frame_id.strip().zfill(6), camera_id, output_root=Path(args.output_root), **roots,
    )
    fmt = lambda e: "-" if e is None else f"{e:.2f}"  # noqa: E731
    print(f"OK   {args.capture_id}/{args.frame_id} cam {camera_id} -> {path} std_err={fmt(std_err)} skin_err={fmt(skin_err)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
