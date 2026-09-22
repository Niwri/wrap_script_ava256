#!/usr/bin/env python3
"""THE consolidated verification renderer for the Ava-256 pipeline's wrapped
output -- formalizes the render pipeline that was run ad-hoc (inline heredoc
scripts) for every landmark/keyline verification image produced this session,
for both FXN596 and CDR970: load a frame's own wrapped_mesh.obj (every frame
gets one now, not just the neutral -- see run_pipeline.py), undo the
capture-level dynamic scale/translate transform (dynamic_transform.py) back
to real-world coordinates, project via a real camera's K/Rt, and draw (a)
landmark_lines.npy's curated keyline wireframe in green, (b) landmark_map.
used_landmark_indices() as a green(standard)/orange(skin) scatter. See
render_wrapped_frame() (the default CLI action) and render_keyline_video()
(a full keyline video across every wrapped frame of a capture, reused by
run_pipeline_batch.py's own post-capture video step).

A handful of older, narrower render modes predate this consolidation and are
kept as explicit --legacy-*/--target-labeled/--source-labeled flags (see
each function's own docstring for why it's still useful in isolation) rather
than deleted outright.

Usage:
    python3 render_wrapped_landmarks.py <CAPTURE_ID> <FRAME_ID> <CAMERA_ID> [--ava256-data-root ...] [--output-root ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import mesh_utils
import camera_utils
import landmark_map
import dynamic_transform

DEFAULT_AVA256_DATA_ROOT = Path("/scratch/thirty/irwinngo/ava-256")
DEFAULT_AVA256_LANDMARK_ROOT = Path("/scratch/ondemand32/irwinngo/landmarks_ava-256/")
DEFAULT_AVA256_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/ava_256_output/")
DEFAULT_OUTPUT_ROOT = Path("/scratch/ondemand32/irwinngo/line_renders_ava256")
DEFAULT_TEMPLATE_POT_LIVE_PATH = SCRIPT_DIR / "neutral_correspondence_without_eyes_POT_all_live.json"
DEFAULT_LANDMARK_LINES_PATH = SCRIPT_DIR / "landmark_lines.npy"

_source_glctx = None

POINT_RADIUS = 4
# Standard (keypoints_3d-sourced, index < 74) vs. skin (FLAME-vertex-sourced,
# index >= 74) drawn in different colors so the two sources are visually
# distinguishable, not just an undifferentiated single-color scatter.
STANDARD_COLOR_BGR = (20, 255, 20)   # green
SKIN_COLOR_BGR = (0, 140, 255)       # orange


def load_frame_points(landmark_root: Path, capture_id: str, frame_id: str) -> list[dict]:
    """Every {group}_<frame_id>.json under landmark_root/<capture_id>/,
    flattened into one list of point dicts (each carrying its own "index")."""
    capture_dir = Path(landmark_root) / capture_id
    points: list[dict] = []
    for path in sorted(capture_dir.glob(f"*_{frame_id}.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        points.extend(payload.get("points", []))
    return points


def load_wrapped_mesh_world(
    ava256_output_root: Path, capture_id: str, frame_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """(verts_world, faces) for capture_id/frame_id's own wrapped_mesh.obj,
    undone back to real-world scale via this capture's own dynamic_transform_
    params.npz (see dynamic_transform.py) -- every wrap produced since that
    integration lives in the small FLAME-template-comparable frame, not
    Ava-256's real ~100-1000-unit world space, so camera-space work (real
    K/Rt projection) needs it undone first. Falls back to the raw (untouched)
    vertices with a warning if this capture has no dynamic_transform_params.npz
    yet (a wrap produced before that integration -- shouldn't happen for
    anything wrapped by the current run_pipeline.py)."""
    wrapped_mesh_path = Path(ava256_output_root) / capture_id / frame_id / "wrapped_mesh.obj"
    if not wrapped_mesh_path.exists():
        raise FileNotFoundError(f"No wrapped_mesh.obj at {wrapped_mesh_path} -- run run_pipeline.py for this capture/frame first")

    mesh = trimesh.load(str(wrapped_mesh_path), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    transform_params_path = dynamic_transform.params_path(ava256_output_root, capture_id)
    if transform_params_path.exists():
        verts = dynamic_transform.untransform(verts, dynamic_transform.load_params(transform_params_path))
    else:
        print(f"WARNING: no dynamic_transform_params.npz for {capture_id} -- using {wrapped_mesh_path}'s vertices as-is")

    return verts, faces


def render_keyline_image(
    verts_world: np.ndarray,
    faces: np.ndarray,
    image: np.ndarray,
    K: np.ndarray,
    Rt: np.ndarray,
    *,
    landmark_lines_path: Path = DEFAULT_LANDMARK_LINES_PATH,
) -> np.ndarray:
    """Pure array in, array out (draws onto a COPY of image, never mutates
    the caller's own array) -- the keyline half of the render pipeline run
    ad-hoc, by hand, for every verification image this whole session.
    landmark_lines.npy's curated wireframe edges, filtered to the wrapped
    mesh's own vertex count (drops any edge referencing a vertex index the
    FLAME topology doesn't have), projected via a real camera's K/Rt, drawn
    green. Factored out from render_wrapped_frame() below so run_pipeline_
    batch.py's keyline-video step can call it directly per frame without a
    disk round-trip through individual JPEGs."""
    landmark_lines = np.load(landmark_lines_path)
    landmark_lines = landmark_lines[(landmark_lines < verts_world.shape[0]).all(axis=1)]
    verts2d = camera_utils.project_points(verts_world, K, Rt)

    keyline_img = image.copy()
    for line in landmark_lines:
        p1 = (int(verts2d[line[0], 0]), int(verts2d[line[0], 1]))
        p2 = (int(verts2d[line[1], 0]), int(verts2d[line[1], 1]))
        cv2.line(keyline_img, p1, p2, STANDARD_COLOR_BGR, 2, cv2.LINE_AA)
    return keyline_img


def render_landmark_scatter_image(
    verts_world: np.ndarray,
    faces: np.ndarray,
    image: np.ndarray,
    K: np.ndarray,
    Rt: np.ndarray,
    *,
    pot_rows: list[list[float]],
) -> np.ndarray:
    """Pure array in, array out -- the landmark-scatter half. Every index
    landmark_map.used_landmark_indices() actually sources (green=standard
    keypoints_3d-driven, orange=skin FLAME-vertex-driven), projected via a
    real camera's K/Rt."""
    indices = landmark_map.used_landmark_indices(len(pot_rows))
    xyz = np.array([mesh_utils.pot_row_world_xyz(verts_world, faces, pot_rows[i]) for i in indices])
    px = camera_utils.project_points(xyz, K, Rt)

    height, width = image.shape[:2]
    scatter_img = image.copy()
    for idx, (x, y) in zip(indices, px):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            color = SKIN_COLOR_BGR if idx >= landmark_map.NUM_STANDARD_LANDMARKS else STANDARD_COLOR_BGR
            cv2.circle(scatter_img, (xi, yi), POINT_RADIUS, color, -1, cv2.LINE_AA)
    return scatter_img


def render_wrapped_frame(
    capture_id: str,
    frame_id: str,
    camera_id: str,
    *,
    ava256_data_root: Path,
    ava256_output_root: Path,
    template_pot_path: Path = DEFAULT_TEMPLATE_POT_LIVE_PATH,
    landmark_lines_path: Path = DEFAULT_LANDMARK_LINES_PATH,
    output_root: Path,
    keylines: bool = True,
    landmarks: bool = True,
) -> tuple[Path | None, Path | None]:
    """THE consolidated render entry point -- formalizes the exact pipeline
    that's been run ad-hoc, by hand (inline heredoc scripts), for every
    landmark/keyline verification image this entire session, for both
    FXN596 and CDR970: load this frame's own wrapped_mesh.obj, undo the
    capture-level dynamic scale/translate transform back to real-world
    coordinates, project via a real camera's K/Rt, and save (a) a green
    keyline wireframe (landmark_lines.npy), (b) a green(standard)/
    orange(skin) landmark scatter (landmark_map.used_landmark_indices()).
    Works for ANY frame with its own wrapped_mesh.obj -- the neutral frame
    (always has one) or any expression frame run_pipeline.py --frame
    FRAME_ID has wrapped. Returns (keyline_path, landmark_path), either None
    if that half was disabled via keylines=False/landmarks=False."""
    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    verts_world, faces = load_wrapped_mesh_world(ava256_output_root, capture_id, frame_id)

    K, Rt = camera_utils.load_camera(actor_dir, camera_id)
    image = camera_utils.load_image(actor_dir, camera_id, frame_id)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    keyline_path = landmark_path = None

    if keylines:
        keyline_img = render_keyline_image(verts_world, faces, image, K, Rt, landmark_lines_path=landmark_lines_path)
        keyline_path = output_root / f"{capture_id}_{frame_id}_{camera_id}_keylines.jpg"
        cv2.imwrite(str(keyline_path), keyline_img)
        print(f"{capture_id}/{frame_id}/{camera_id}: wrote keylines -> {keyline_path}")

    if landmarks:
        with Path(template_pot_path).open("r", encoding="utf-8") as f:
            pot_rows = json.load(f)
        scatter_img = render_landmark_scatter_image(verts_world, faces, image, K, Rt, pot_rows=pot_rows)
        landmark_path = output_root / f"{capture_id}_{frame_id}_{camera_id}_landmarks.jpg"
        cv2.imwrite(str(landmark_path), scatter_img)
        print(f"{capture_id}/{frame_id}/{camera_id}: wrote landmark scatter -> {landmark_path}")

    return keyline_path, landmark_path


# --- keyline video (run_pipeline_batch.py's own post-capture step) ----------
#
# Same VideoWriter-then-ffmpeg-reencode pattern as wrap_script/render_
# landmarks.py's finalize_windows_compatible_mp4()/main() (lines ~210-365):
# write raw frames via cv2.VideoWriter(mp4v) to a _tmp.mp4, then re-encode
# with ffmpeg (libx264/yuv420p/+faststart, for actual cross-player/Windows
# compatibility -- mp4v alone is not reliably seekable/playable outside
# OpenCV itself), falling back to keeping the raw mp4v file if ffmpeg isn't
# on PATH or the re-encode fails.
#
# fps: Nersemble's render_landmarks.py derives displayed_fps from its own
# continuous camera .mp4's real frame rate (73fps raw / RAW_FRAME_STRIDE=5
# landmark-file stride = ~14.6fps) -- Ava-256 has no continuous video source
# to query (per-frame .avif stills only, no cv2.VideoCapture-readable fps
# metadata), so there's no equivalent real rate to derive from. 24fps is a
# deliberate choice (the standard cinematic/animation playback rate, not an
# arbitrary guess) rather than trying to fake a source frame rate this
# dataset doesn't expose.
DEFAULT_KEYLINE_VIDEO_FPS = 24.0


def finalize_mp4(temp_output_path: Path, output_path: Path) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        if output_path.exists():
            output_path.unlink()
        temp_output_path.rename(output_path)
        print("ffmpeg not found; kept OpenCV mp4v output.")
        return

    cmd = [
        ffmpeg_bin, "-y", "-i", str(temp_output_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as exc:
        if output_path.exists():
            output_path.unlink()
        temp_output_path.rename(output_path)
        print("ffmpeg conversion failed; kept OpenCV mp4v output.")
        stderr_text = exc.stderr.decode("utf-8", errors="ignore")
        if stderr_text:
            print(stderr_text)
        return
    temp_output_path.unlink(missing_ok=True)


def list_wrapped_frames_in_capture_order(
    actor_dir: Path, ava256_output_root: Path, capture_id: str, neutral_frame_id: str | None = None,
) -> list[str]:
    """Every frame_id under {ava256_output_root}/{capture_id}/ that has its
    own wrapped_mesh.obj, ordered by frame_list.csv's own row order (NOT
    glob/directory-listing order, which is lexicographic on strings and
    isn't guaranteed to match chronological/numeric frame order) -- the
    neutral frame first (if given and wrapped), then every other frame in
    frame_list.csv order."""
    frame_list_path = Path(actor_dir) / "decoder" / "frame_list.csv"
    with frame_list_path.open("r", encoding="utf-8", newline="") as f:
        all_frame_ids = [row["frame_id"].strip().zfill(6) for row in csv.DictReader(f)]

    capture_output_dir = Path(ava256_output_root) / capture_id
    def has_wrap(frame_id: str) -> bool:
        return (capture_output_dir / frame_id / "wrapped_mesh.obj").exists()

    ordered: list[str] = []
    if neutral_frame_id is not None and has_wrap(neutral_frame_id):
        ordered.append(neutral_frame_id)
    for frame_id in all_frame_ids:
        if frame_id != neutral_frame_id and has_wrap(frame_id):
            ordered.append(frame_id)
    return ordered


def render_keyline_video(
    capture_id: str,
    frame_ids: list[str],
    camera_id: str,
    *,
    ava256_data_root: Path,
    ava256_output_root: Path,
    landmark_lines_path: Path = DEFAULT_LANDMARK_LINES_PATH,
    line_renders_root: Path,
    fps: float = DEFAULT_KEYLINE_VIDEO_FPS,
) -> Path | None:
    """Keyline-only (no landmark scatter, matching "keyline video") render
    across frame_ids in the given order, one video frame per wrapped mesh --
    reuses render_keyline_image() per frame, so this is exactly the same
    math as the single-image CLI, just accumulated into a video instead of
    separate JPEGs. Returns None (no video written) if frame_ids is empty or
    every frame turned out unusable (e.g. no camera image for that frame)."""
    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    line_renders_root = Path(line_renders_root)
    line_renders_root.mkdir(parents=True, exist_ok=True)
    output_path = line_renders_root / f"{capture_id}_{camera_id}_keylines.mp4"
    temp_output_path = output_path.with_name(f"{output_path.stem}_tmp.mp4")

    K, Rt = camera_utils.load_camera(actor_dir, camera_id)
    writer: cv2.VideoWriter | None = None
    written = 0
    try:
        for frame_id in frame_ids:
            try:
                image = camera_utils.load_image(actor_dir, camera_id, frame_id)
            except (KeyError, FileNotFoundError) as exc:
                print(f"SKIP {capture_id}/{frame_id}: no image for cam {camera_id} ({exc})")
                continue
            verts_world, faces = load_wrapped_mesh_world(ava256_output_root, capture_id, frame_id)
            keyline_img = render_keyline_image(verts_world, faces, image, K, Rt, landmark_lines_path=landmark_lines_path)

            if writer is None:
                h, w = keyline_img.shape[:2]
                writer = cv2.VideoWriter(str(temp_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(keyline_img)
            written += 1
    finally:
        if writer is not None:
            writer.release()

    if written == 0:
        print(f"No usable frames for {capture_id}/{camera_id} keyline video -- nothing written")
        temp_output_path.unlink(missing_ok=True)
        return None

    finalize_mp4(temp_output_path, output_path)
    print(f"{capture_id}/{camera_id}: wrote {written}-frame keyline video -> {output_path}")
    return output_path


def render_frame(
    capture_id: str,
    frame_id: str,
    camera_id: str,
    *,
    ava256_data_root: Path,
    ava256_landmark_root: Path,
    output_root: Path,
) -> Path:
    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    K, Rt = camera_utils.load_camera(actor_dir, camera_id)
    image = camera_utils.load_image(actor_dir, camera_id, frame_id)

    points = load_frame_points(ava256_landmark_root, capture_id, frame_id)
    if not points:
        raise FileNotFoundError(
            f"No propagated landmark points found for {capture_id}/{frame_id} under {ava256_landmark_root}"
        )

    xyz = np.array([[p["x"], p["y"], p["z"]] for p in points], dtype=np.float64)
    px = camera_utils.project_points(xyz, K, Rt)

    height, width = image.shape[:2]
    drawn = 0
    for point, (x, y) in zip(points, px):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            color = SKIN_COLOR_BGR if point["index"] >= landmark_map.NUM_STANDARD_LANDMARKS else STANDARD_COLOR_BGR
            cv2.circle(image, (xi, yi), POINT_RADIUS, color, -1, cv2.LINE_AA)
            drawn += 1

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{capture_id}_{frame_id}_{camera_id}.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"{capture_id}/{frame_id}/{camera_id}: drew {drawn}/{len(points)} point(s) in-frame -> {output_path}")
    return output_path


def render_target_correspondence(
    capture_id: str,
    frame_id: str,
    camera_id: str,
    *,
    ava256_data_root: Path,
    output_root: Path,
    output_suffix: str = "_target_labeled",
) -> Path:
    """Projects the RAW keypoints_3d-derived positions actually used to
    DRIVE the wrap (the correspondence INPUT, not the wrapped mesh's
    resulting vertex position -- see render_wrap_direct() for that) onto a
    camera photo, with each point's standard landmark index printed next to
    it. Standard/no-skin indices only (skin never comes from keypoints_3d --
    see landmark_map.py). Useful for sanity-checking the input data itself
    when a wrap looks wrong, independent of anything the Wrap tool did with
    it."""
    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    kp_by_id = mesh_utils.keypoints_3d_by_id(actor_dir, frame_id)

    present = [
        (std_idx, kp_by_id[kp_id])
        for std_idx, kp_id in sorted(landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID.items())
        if kp_id in kp_by_id
    ]
    missing = sorted(
        std_idx for std_idx in landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID
        if landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID[std_idx] not in kp_by_id
    )
    if missing:
        print(f"NOTE: {len(missing)} standard index(es) have no keypoints_3d entry this frame, skipped: {missing}")

    K, Rt = camera_utils.load_camera(actor_dir, camera_id)
    image = camera_utils.load_image(actor_dir, camera_id, frame_id)
    height, width = image.shape[:2]

    xyz = np.array([xyz for _idx, xyz in present], dtype=np.float64)
    px = camera_utils.project_points(xyz, K, Rt)

    drawn = 0
    for (idx, _xyz), (x, y) in zip(present, px):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            cv2.circle(image, (xi, yi), 4, STANDARD_COLOR_BGR, -1, cv2.LINE_AA)
            cv2.putText(image, str(idx), (xi + 6, yi - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, str(idx), (xi + 6, yi - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            drawn += 1

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{capture_id}_{frame_id}_{camera_id}{output_suffix}.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"{capture_id}/{frame_id}/{camera_id}: drew {drawn}/{len(present)} target point(s) in-frame -> {output_path}")
    return output_path


def render_source_correspondence(
    capture_id: str,
    camera_id: str,
    *,
    ava256_data_root: Path,
    neutral_mesh_path: Path,
    output_root: Path,
    distance: float = 0.55,
    canvas_size: int = 700,
    focal: float = 900.0,
    output_suffix: str = "_source_labeled",
) -> Path:
    """The other half of render_target_correspondence(): shows where each
    standard landmark index sits on the SOURCE/template mesh (neutral.obj,
    generic FLAME topology, BEFORE wrapping -- the neutral (left) side of
    the correspondence the Wrap tool actually pairs against the target
    (right) side).

    neutral.obj lives in its own small (~0.2 units across), unregistered
    local coordinate frame -- NOT a rigid transform of Ava-256's world
    space, so camera_id's own R cannot be reused verbatim: an earlier
    version of this function tried exactly that and rendered the BACK of
    the template's head, since a rotation matrix's meaning is tied to the
    coordinate frame it was calibrated in, and reusing it across two
    physically unrelated frames doesn't carry over any actual "same viewing
    angle" -- confirmed by the resulting render (upside-down neck/collar
    shape with the point labels still landing in a recognizable face
    layout, since THEIR projection math was still correct -- only the
    borrowed camera basis was meaningless here). camera_id is accepted only
    for output-filename bookkeeping now, not for its rotation.

    Instead, this derives the template's OWN natural front-facing view
    directly from its own geometry: forward = the outward surface normal at
    the undernose-center landmark (index 61), negated (same technique
    face_ray_masking_ava256.classify_front_right_left already uses to find
    "true front" on a wrapped mesh) -- and up = world +Y, which the
    template's own vertex bounding box confirms is its natural up/down axis
    (head-top at max Y, chin/neck at min Y)."""
    import nvdiffrast.torch as dr
    import torch
    import trimesh as _trimesh

    mesh = _trimesh.load(str(neutral_mesh_path), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    with Path(DEFAULT_TEMPLATE_POT_LIVE_PATH).open("r", encoding="utf-8") as f:
        _pot_rows_for_forward = json.load(f)
    forward = -mesh_utils.pot_row_face_normal(verts, faces, _pot_rows_for_forward[61])
    forward = forward / np.linalg.norm(forward)
    up_hint = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up_hint)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)  # right-handed: right, down, forward
    R = np.stack([right, down, forward], axis=0)  # rows = camera's own axes in world space (same convention as camera_forward_world)

    centroid = verts.mean(axis=0)
    cam_center = centroid - forward * distance
    t_synth = -R @ cam_center
    W = H = canvas_size
    K_synth = np.array([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]], dtype=np.float64)

    with Path(DEFAULT_TEMPLATE_POT_LIVE_PATH).open("r", encoding="utf-8") as f:
        pot_rows = json.load(f)

    indices = sorted(landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID.keys())
    xyz = np.array([mesh_utils.pot_row_world_xyz(verts, faces, pot_rows[i]) for i in indices])
    px = camera_utils.project_points(xyz, K_synth, np.hstack([R, t_synth.reshape(3, 1)]))

    # Shaded nvdiffrast render of the template itself for anatomical context.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    verts_cam = verts @ R.T + t_synth
    near, far = 0.05, 5.0
    fx = fy = focal
    proj = np.array([
        [2 * fx / W, 0, 2 * (W / 2) / W - 1, 0],
        [0, 2 * fy / H, 2 * (H / 2) / H - 1, 0],
        [0, 0, (far + near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, 1, 0],
    ], dtype=np.float64)
    ones = np.ones((verts.shape[0], 1))
    clip = (proj @ np.concatenate([verts_cam, ones], axis=1).T).T
    clip_t = torch.from_numpy(clip).float().to(device)[None].contiguous()
    faces_t = torch.from_numpy(faces.astype(np.int32)).to(device).contiguous()
    normals_t = torch.from_numpy(normals.copy()).float().to(device)[None]

    global _source_glctx
    if _source_glctx is None:
        _source_glctx = dr.RasterizeCudaContext()

    rast_out, _ = dr.rasterize(_source_glctx, clip_t, faces_t, resolution=[H, W])
    normal_interp, _ = dr.interpolate(normals_t, rast_out, faces_t)
    normal_np = normal_interp[0].detach().cpu().numpy()
    ndotl = np.clip(np.sum(normal_np * (-forward), axis=-1), 0.0, 1.0)
    shade = np.clip(0.25 + 0.85 * ndotl, 0.0, 1.0)
    mesh_color = np.array([0.75, 0.72, 0.68])
    rendered = (shade[..., None] * mesh_color[None, None, :] * 255.0).astype(np.uint8)
    image = rendered[:, :, ::-1].copy()
    alpha = (rast_out[0, ..., 3] > 0).detach().cpu().numpy()
    image[~alpha] = 0

    drawn = 0
    for idx, (x, y) in zip(indices, px):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < W and 0 <= yi < H:
            cv2.circle(image, (xi, yi), 4, STANDARD_COLOR_BGR, -1, cv2.LINE_AA)
            cv2.putText(image, str(idx), (xi + 6, yi - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image, str(idx), (xi + 6, yi - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            drawn += 1

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{capture_id}_neutral_{camera_id}{output_suffix}.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"{capture_id}/neutral.obj/{camera_id}: drew {drawn}/{len(indices)} source point(s) -> {output_path}")
    return output_path


def render_wrap_direct(
    capture_id: str,
    frame_id: str,
    camera_id: str,
    *,
    ava256_data_root: Path,
    ava256_output_root: Path,
    template_pot_path: Path = DEFAULT_TEMPLATE_POT_LIVE_PATH,
    output_root: Path,
    output_suffix: str = "_direct",
) -> Path:
    """Projects a wrapped_mesh.obj's own POT-row positions directly onto a
    camera photo -- no propagation involved, so this only works for a frame
    that actually has a wrapped_mesh.obj (i.e. the neutral frame -- see
    run_pipeline.py). Useful for inspecting a specific wrap in isolation
    (e.g. comparing --no-skin vs. the skin-augmented pass, or before vs.
    after face_ray_masking), where going through propagation's own
    triangulation would be the wrong tool.

    Filters to landmark_map.used_landmark_indices() -- the same "only
    actually-sourced indices" whitelist render_frame()'s propagation-JSON
    path already gets for free (those files simply never contain the
    unsourced ones) -- NOT "every standard index except 2/9": that would
    include the ~27 standard indices with no keypoints_3d source at all,
    whose position on any wrap is just wherever ICP/blend-wrap interpolation
    happened to put that vertex, not a real landmark (see wrap_script/
    render_landmarks.py's own whitelist.json filter for the same principle
    in the Nersemble pipeline)."""
    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    wrapped_mesh_path = Path(ava256_output_root) / capture_id / frame_id / "wrapped_mesh.obj"
    if not wrapped_mesh_path.exists():
        raise FileNotFoundError(f"No wrapped_mesh.obj at {wrapped_mesh_path} -- run run_pipeline.py for this capture/frame first")

    mesh = trimesh.load(str(wrapped_mesh_path), process=False)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)

    with Path(template_pot_path).open("r", encoding="utf-8") as f:
        pot_rows = json.load(f)
    indices = landmark_map.used_landmark_indices(len(pot_rows))

    K, Rt = camera_utils.load_camera(actor_dir, camera_id)
    image = camera_utils.load_image(actor_dir, camera_id, frame_id)
    height, width = image.shape[:2]

    xyz = np.array([mesh_utils.pot_row_world_xyz(verts, faces, pot_rows[i]) for i in indices])
    px = camera_utils.project_points(xyz, K, Rt)

    drawn = 0
    for idx, (x, y) in zip(indices, px):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < width and 0 <= yi < height:
            color = SKIN_COLOR_BGR if idx >= landmark_map.NUM_STANDARD_LANDMARKS else STANDARD_COLOR_BGR
            cv2.circle(image, (xi, yi), POINT_RADIUS, color, -1, cv2.LINE_AA)
            drawn += 1

    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{capture_id}_{frame_id}_{camera_id}{output_suffix}.jpg"
    cv2.imwrite(str(output_path), image)
    print(f"{capture_id}/{frame_id}/{camera_id}: drew {drawn}/{len(indices)} point(s) in-frame -> {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_id")
    parser.add_argument("frame_id", nargs="?", default=None, help="Omit only when --video is given")
    parser.add_argument("camera_id")
    parser.add_argument("--video", action="store_true",
                         help="Render a keyline-only video across every frame of this capture that has its own "
                              "wrapped_mesh.obj (frame_list.csv order, neutral frame first), instead of a single "
                              "frame's images. frame_id is ignored/unnecessary in this mode.")
    parser.add_argument("--line-renders-root", default=str(DEFAULT_OUTPUT_ROOT),
                         help="--video only: output directory for the .mp4 (same convention as --output-root)")
    parser.add_argument("--fps", type=float, default=DEFAULT_KEYLINE_VIDEO_FPS, help="--video only")
    parser.add_argument("--ava256-data-root", default=str(DEFAULT_AVA256_DATA_ROOT))
    parser.add_argument("--ava256-landmark-root", default=str(DEFAULT_AVA256_LANDMARK_ROOT))
    parser.add_argument("--ava256-output-root", default=str(DEFAULT_AVA256_OUTPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--template-pot-path", default=str(DEFAULT_TEMPLATE_POT_LIVE_PATH),
                         help="The live-extended POT file (see mesh_utils.build_live_extended_pot)")
    parser.add_argument("--landmark-lines-path", default=str(DEFAULT_LANDMARK_LINES_PATH),
                         help="The curated keyline wireframe edge list")
    parser.add_argument("--no-keylines", action="store_true", help="Skip the keyline wireframe render")
    parser.add_argument("--no-landmarks", action="store_true", help="Skip the landmark scatter render")
    parser.add_argument("--legacy-direct", action="store_true",
                         help="[legacy, pre-dynamic_transform] Same as the default mode's landmark scatter, but "
                              "WITHOUT undoing the dynamic transform -- only correct for a wrap produced before "
                              "that integration. Prefer the default mode.")
    parser.add_argument("--legacy-propagated-points", action="store_true",
                         help="[legacy, pre-per-frame-wrapped_mesh.obj] Render the raw propagated-landmark JSON "
                              "points directly (no wrapped_mesh.obj needed) -- from when non-neutral frames had "
                              "no wrap of their own yet. Prefer the default mode now that every frame gets one.")
    parser.add_argument("--target-labeled", action="store_true",
                         help="Project the RAW keypoints_3d target positions (wrap correspondence INPUT, not the "
                              "wrap's output) with index-number labels -- standard/no-skin indices only")
    parser.add_argument("--source-labeled", action="store_true",
                         help="Show each standard index's position on the SOURCE/template mesh (neutral.obj), "
                              "labeled, viewed from camera_id's own orientation (frame_id is ignored -- the "
                              "template has no per-frame variant)")
    parser.add_argument("--neutral-mesh-path", default=str(SCRIPT_DIR / "neutral.obj"),
                         help="--source-labeled only")
    args = parser.parse_args()
    # Same zero-pad-on-input fix as run_pipeline.py's --frame -- a raw
    # user-typed 5-digit frame_id (e.g. "32939") silently fails to resolve
    # against Ava-256's always-6-digit on-disk file names otherwise.
    if args.frame_id is not None:
        args.frame_id = args.frame_id.strip().zfill(6)

    if args.video:
        import neutral_frame as neutral_frame_module
        actor_dir = mesh_utils.resolve_capture_dir(Path(args.ava256_data_root), args.capture_id)
        neutral = neutral_frame_module.resolve_neutral_frame(args.capture_id, actor_dir, Path(args.ava256_landmark_root))
        frame_ids = list_wrapped_frames_in_capture_order(
            actor_dir, Path(args.ava256_output_root), args.capture_id, neutral_frame_id=neutral["frame_id"],
        )
        render_keyline_video(
            args.capture_id, frame_ids, args.camera_id,
            ava256_data_root=Path(args.ava256_data_root),
            ava256_output_root=Path(args.ava256_output_root),
            landmark_lines_path=Path(args.landmark_lines_path),
            line_renders_root=Path(args.line_renders_root),
            fps=args.fps,
        )
    elif args.frame_id is None:
        parser.error("frame_id is required unless --video is given")
    elif args.source_labeled:
        render_source_correspondence(
            args.capture_id, args.camera_id,
            ava256_data_root=Path(args.ava256_data_root),
            neutral_mesh_path=Path(args.neutral_mesh_path),
            output_root=Path(args.output_root),
        )
    elif args.target_labeled:
        render_target_correspondence(
            args.capture_id, args.frame_id, args.camera_id,
            ava256_data_root=Path(args.ava256_data_root),
            output_root=Path(args.output_root),
        )
    elif args.legacy_direct:
        render_wrap_direct(
            args.capture_id, args.frame_id, args.camera_id,
            ava256_data_root=Path(args.ava256_data_root),
            ava256_output_root=Path(args.ava256_output_root),
            template_pot_path=Path(args.template_pot_path),
            output_root=Path(args.output_root),
        )
    elif args.legacy_propagated_points:
        render_frame(
            args.capture_id, args.frame_id, args.camera_id,
            ava256_data_root=Path(args.ava256_data_root),
            ava256_landmark_root=Path(args.ava256_landmark_root),
            output_root=Path(args.output_root),
        )
    else:
        render_wrapped_frame(
            args.capture_id, args.frame_id, args.camera_id,
            ava256_data_root=Path(args.ava256_data_root),
            ava256_output_root=Path(args.ava256_output_root),
            template_pot_path=Path(args.template_pot_path),
            landmark_lines_path=Path(args.landmark_lines_path),
            output_root=Path(args.output_root),
            keylines=not args.no_keylines,
            landmarks=not args.no_landmarks,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
