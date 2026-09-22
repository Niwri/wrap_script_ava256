#!/usr/bin/env python3
"""Ava-256 counterpart to wrap_script_facescape/run_pipeline.py -- wraps the
generic FLAME template onto one Ava-256 capture's neutral frame, using
keypoints_3d-derived landmark positions (no per-camera 2D detection/
triangulation at all -- see landmark_map.py) as the correspondence, and the
capture's own already-registered kinematic_tracking mesh (world-space, via
head_pose) as the wrap target (no photogrammetry reconstruction needed).

Two-phase wrap (verified against wrap_script_facescape/run_neutral_skin_
propagation.py:926-937, which confirm_labeled()s the FaceScape neutral
expression itself after writing its own skin ground truth -- i.e. FaceScape's
neutral wrap really is produced twice): the bootstrap pass (include_skin=
False, called in-process by run_neutral_skin_propagation.py when
wrapped_mesh.obj is missing) uses only the ~43 standard-index rows
keypoints_3d can source for this specific frame; the final pass
(include_skin=True, the standalone/batch CLI path, gated on label status
"unreviewed") additionally fills the skin rows from the bootstrap wrap's own
vertex positions (read directly off the correspondence file's own POT rows,
not a separately-recomputed region-file list -- see mesh_utils.pot_row_
vertex_index's docstring for why), then re-wraps and marks "unconfirmed".

Usage:
    python3 run_pipeline.py <CAPTURE_ID> [--ava256-data-root ...] [--no-skin] [--frame FRAME_ID] [--dry-run]
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import trimesh

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import landmark_map
import mesh_utils
import camera_utils
import dynamic_transform
from label_tracker_ava256 import Ava256LabelTracker, ensure_label_tracker_file


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DEFAULTS = {
    "ava256_data_root": Path("/scratch/thirty/irwinngo/ava-256"),
    "ava256_landmark_root": Path("/scratch/ondemand32/irwinngo/landmarks_ava-256/"),
    "ava256_output_root": Path("/scratch/ondemand32/irwinngo/ava_256_output/"),
    "label_tracker_path": Path("/scratch/ondemand32/irwinngo/label_tracker.json"),
    # Local, self-contained copy of the FLAME face topology (see the
    # directory-consolidation note below) -- was FaceView/ava-256/assets/
    # face_topology.obj.
    "ava256_mesh_topology_path": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/face_topology.obj"),
    # --- Everything below was consolidated directly into wrap_script_ava256/
    # (blendshape/, segmentation/ [segment.py+model/+data_loader.py],
    # keypoint_*.txt, neutral_correspondence_without_eyes_POT.json,
    # template.wrap, neutral.obj, facescape_mask.txt, face_topology.obj,
    # landmark_lines.npy, facescape_run_pipeline.py, and a real working
    # run_wrap_script.py replacing the old dead stub) so this directory is
    # self-contained rather than reaching into wrap_script_facescape/
    # wrap_script/ for its own defaults -- those two directories are still
    # used by FaceScape's own separate production pipeline and shouldn't be
    # a hard dependency for Ava-256 to run. All still fully overridable via
    # the same CLI flags if a caller wants to point back at shared copies.
    "wrap_script_path": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/run_wrap_script.py"),
    # Base 74-STANDARD-row file only -- deliberately NOT the cached "_all"
    # 107-row file (33 skin rows, confirmed stale against the live skin
    # region definition, which gives 66 today). Skin rows are appended live
    # at runtime by mesh_utils.build_live_extended_pot(), matching
    # FaceScape's own self-healing ensure_extended_correspondence_files()
    # semantics instead of trusting a frozen snapshot.
    "template_pot_path": Path(
        "/scratch/ondemand32/irwinngo/wrap_script_ava256/neutral_correspondence_without_eyes_POT.json"
    ),
    # The Wrap4D/R3DS node-graph project file wrap_mesh() loads and fills in
    # (NeutralInput/TargetInput/FaceMask/CorrespondancePointsNeutral/
    # CorrespondenceMapping/BlendWrapping/Wrapping/SaveWrap nodes) -- normally
    # a fixed module-level constant inside run_wrap_script.py
    # (TEMPLATE_PATH), overridden here the same way FACE_MASK_PATH is (see
    # _load_wrap_script's docstring).
    "template_wrap_path": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/template.wrap"),
    # The actual Faceform Wrap software installation wrap_mesh() shells out
    # to (WrapCmd binary + its license file) -- normally read from a dotenv
    # file (WrapCmd/WrapLicense env vars); overridden here by setting those
    # same env vars directly before calling wrap_mesh(), since python-dotenv
    # doesn't clobber an already-set env var by default.
    "faceform_wrap_cmd_path": Path("/scratch/ondemand32/irwinngo/faceform/Faceform_Wrap_2025.11.14_Linux/WrapCmd"),
    "faceform_wrap_license_path": Path(
        "/scratch/ondemand32/irwinngo/faceform/Faceform_Wrap_License_1035244_10483637.lic"
    ),
    # --- portability: everything below overrides a module-level constant
    # that has no CLI flag of its own upstream (segment.py's checkpoint
    # paths, face_ray_masking_ava256.py's WRAP_SCRIPT_DIR/FACESCAPE_MASK_PATH,
    # run_wrap_script.py's blendshape root, mesh_utils.py's facescape
    # run_pipeline.py path) -- see run_pipeline()'s monkeypatch block below.
    "sam_checkpoint_path": Path("/scratch/ondemand32/irwinngo/models/sam_vit_h_4b8939.pth"),
    "u2net_checkpoint_path": Path("/scratch/ondemand32/irwinngo/models/u2net_human_seg.pth"),
    "wrap_script_dir": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/segmentation"),
    "facescape_mask_path": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/facescape_mask.txt"),
    "blendshape_root": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/blendshape"),
    "facescape_run_pipeline_path": Path("/scratch/ondemand32/irwinngo/wrap_script_ava256/facescape_run_pipeline.py"),
}


def _load_wrap_script(wrap_script_path: Path):
    """Loads wrap_mesh()/DEFAULT_NEUTRAL_MESH_PATH from the given
    run_wrap_script.py (defaults to wrap_script_facescape's own, reused
    unmodified -- the WrapCmd/template.wrap driver is dataset-agnostic).
    Also returns the loaded module itself so callers can monkeypatch its
    FACE_MASK_PATH global (see run_pipeline()'s face-ray-masking step) --
    each call to this function loads a FRESH module instance (a new exec of
    the file via importlib), so mutating that instance's attribute never
    touches the file on disk or any other process's/import's copy."""
    module = _load_module("wrap_script_facescape_run_wrap_script", Path(wrap_script_path))
    return module.wrap_mesh, module.DEFAULT_NEUTRAL_MESH_PATH, module


def ensure_roots(*, landmark_root: Path, output_root: Path, label_tracker_path: Path) -> None:
    landmark_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    ensure_label_tracker_file(label_tracker_path)


@contextmanager
def _capture_pipeline_lock(output_root: Path, capture_id: str):
    """Exclusive, non-blocking per-capture flock -- output_root/<capture>/
    .pipeline.lock -- so concurrent run_pipeline.py invocations never
    double-wrap the same capture. Yields True if acquired (caller should
    proceed), False if another run already holds it (caller should skip)."""
    lock_dir = output_root / capture_id
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = open(lock_dir / ".pipeline.lock", "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock_file.close()
        yield False
        return
    try:
        yield True
    finally:
        fcntl.flock(lock_file, fcntl.LOCK_UN)
        lock_file.close()


def _load_template_pot(template_pot_path: Path) -> list[list[float]]:
    with Path(template_pot_path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _build_target_correspondence(
    actor_dir: Path,
    frame_id: str,
    pot_rows: list[list[float]],
    *,
    include_skin: bool,
    prior_wrap_verts: np.ndarray | None,
    faces: np.ndarray,
    transform_params: dict | None = None,
) -> tuple[list[dict[str, float]], list[int]]:
    """Returns (target_points, missing_indices). target_points is a flat
    list matching pot_rows' own length/position -- one {"x","y","z"} dict
    per row, same schema wrap_script/triangulate.py's own `formatted` output
    uses (so wrap_mesh()'s np.array(json.load(...))/np.delete(...,axis=0)
    round-trips correctly, since a 1-D object array of dicts deletes by
    position exactly like a plain list).

    transform_params (see dynamic_transform.py), if given, is applied ONLY to
    the standard-index rows sourced from keypoints_3d (real-world scale) --
    NOT to skin rows sourced from prior_wrap_verts below, which (once the
    caller transforms the scan/wraps the same way) already live in the
    transformed/FLAME-comparable frame."""
    kp_by_id = mesh_utils.keypoints_3d_by_id(actor_dir, frame_id)
    target_points: list[dict[str, float]] = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in pot_rows]
    missing_indices: list[int] = []

    for std_idx in range(landmark_map.NUM_STANDARD_LANDMARKS):
        if std_idx in landmark_map.ALWAYS_EXCLUDED_INDICES:
            missing_indices.append(std_idx)
            continue
        kp_id = landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID.get(std_idx)
        if kp_id is None or kp_id not in kp_by_id:
            missing_indices.append(std_idx)
            continue
        x, y, z = kp_by_id[kp_id]
        xyz = np.array([x, y, z], dtype=np.float64)
        if transform_params is not None:
            xyz = dynamic_transform.transform(xyz[None, :], transform_params)[0]
        target_points[std_idx] = {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}

    for skin_idx in range(landmark_map.NUM_STANDARD_LANDMARKS, len(pot_rows)):
        if include_skin and prior_wrap_verts is not None:
            vertex_idx = mesh_utils.pot_row_vertex_index(faces, pot_rows[skin_idx])
            xyz = prior_wrap_verts[vertex_idx]
            target_points[skin_idx] = {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}
        else:
            missing_indices.append(skin_idx)

    return target_points, missing_indices


def _load_propagated_points(landmark_root: Path, capture_id: str, frame_id: str) -> dict[int, tuple[float, float, float]]:
    """index -> (x,y,z), merged across every {group}_<frame_id>.json file
    run_neutral_skin_propagation.py wrote for this capture/frame (ear/
    nosebridge/undernose/eyebrow/chin/eyelid_lip/skin -- see its own
    write_frame_outputs())."""
    points: dict[int, tuple[float, float, float]] = {}
    capture_dir = Path(landmark_root) / capture_id
    if not capture_dir.is_dir():
        return points
    for path in capture_dir.glob(f"*_{frame_id}.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for p in payload.get("points", []):
            points[int(p["index"])] = (float(p["x"]), float(p["y"]), float(p["z"]))
    return points


def _build_propagated_target_correspondence(
    actor_dir: Path, ava256_landmark_root: Path, capture_id: str, frame_id: str, pot_rows: list[list[float]],
    transform_params: dict,
) -> tuple[list[dict[str, float]], list[int]]:
    """Non-neutral (propagated-expression) counterpart to
    _build_target_correspondence() above. keypoints_3d actually covers every
    frame (confirmed on disk for 20210818--1332--CDR970: 5039/5039 frames),
    but has no per-frame SKIN ground truth at all, so skin rows come from
    run_neutral_skin_propagation.py's per-(group,frame) output files as
    before. Standard rows are split by user directive: eyelid/lip indices
    (landmark_map.AVA256_STANDARD_INDEX_REGION_GROUPS["eyelid_lip"]) always
    come from keypoints_3d directly, every frame, never from propagation --
    confirmed by direct comparison on CDR970/070133 (an open-mouth
    expression) that CoTracker/RAFT propagation drifts badly on inner-lip
    points during large mouth deformation (~10-13 unit Y error on indices
    25/26/55, index 56 lost entirely) while keypoints_3d has real per-frame
    data with only 1/30 eyelid_lip indices missing. Every other standard
    group (ear/nosebridge/undernose/eyebrow/chin) still comes from
    propagation, matching run_neutral_skin_propagation.py's own per-group
    camera-selection/tracking work for those regions. All of it is
    real-world-scale, so transform_params is applied unconditionally here,
    same as before."""
    propagated = _load_propagated_points(ava256_landmark_root, capture_id, frame_id)
    kp_by_id = mesh_utils.keypoints_3d_by_id(actor_dir, frame_id)
    eyelid_lip_indices = set(landmark_map.AVA256_STANDARD_INDEX_REGION_GROUPS["eyelid_lip"])
    target_points: list[dict[str, float]] = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in pot_rows]
    missing_indices: list[int] = []
    for idx in range(len(pot_rows)):
        if idx in landmark_map.ALWAYS_EXCLUDED_INDICES:
            missing_indices.append(idx)
            continue
        if idx in eyelid_lip_indices:
            kp_id = landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID.get(idx)
            if kp_id is None or kp_id not in kp_by_id:
                missing_indices.append(idx)
                continue
            xyz = np.array(kp_by_id[kp_id], dtype=np.float64)
        elif idx in propagated:
            xyz = np.array(propagated[idx], dtype=np.float64)
        else:
            missing_indices.append(idx)
            continue
        xyz = dynamic_transform.transform(xyz[None, :], transform_params)[0]
        target_points[idx] = {"x": float(xyz[0]), "y": float(xyz[1]), "z": float(xyz[2])}
    return target_points, missing_indices


def run_pipeline(
    capture_id: str,
    *,
    ava256_data_root: Path,
    ava256_landmark_root: Path,
    ava256_output_root: Path,
    label_tracker_path: Path,
    ava256_mesh_topology_path: Path,
    wrap_script_path: Path = DEFAULTS["wrap_script_path"],
    template_pot_path: Path = DEFAULTS["template_pot_path"],
    template_wrap_path: Path = DEFAULTS["template_wrap_path"],
    faceform_wrap_cmd_path: Path = DEFAULTS["faceform_wrap_cmd_path"],
    faceform_wrap_license_path: Path = DEFAULTS["faceform_wrap_license_path"],
    sam_checkpoint_path: Path = DEFAULTS["sam_checkpoint_path"],
    u2net_checkpoint_path: Path = DEFAULTS["u2net_checkpoint_path"],
    wrap_script_dir: Path = DEFAULTS["wrap_script_dir"],
    facescape_mask_path: Path = DEFAULTS["facescape_mask_path"],
    blendshape_root: Path = DEFAULTS["blendshape_root"],
    facescape_run_pipeline_path: Path = DEFAULTS["facescape_run_pipeline_path"],
    include_skin: bool = True,
    frame_id_override: str | None = None,
    use_face_ray_masking: bool = True,
    face_ray_mask_path_override: Path | None = None,
    dry_run: bool = False,
    force: bool = False,
) -> Path:
    ensure_roots(landmark_root=ava256_landmark_root, output_root=ava256_output_root, label_tracker_path=label_tracker_path)
    wrap_mesh, default_neutral_mesh_path, wrap_script_module = _load_wrap_script(wrap_script_path)
    # Same monkeypatch pattern as FACE_MASK_PATH below -- a fresh module
    # instance from this call's own _load_wrap_script(), safe to mutate.
    wrap_script_module.TEMPLATE_PATH = Path(template_wrap_path)
    wrap_script_module.NERSEMBLE_BLENDSHAPE_ROOT = Path(blendshape_root)
    # Portability: module-level constants with no CLI flag upstream of this
    # script (see DEFAULTS' own comment above). face_ray_masking_ava256 is a
    # cheap import (SAM/U2Net themselves load lazily inside it) -- safe to
    # import + patch here even for --no-skin/bootstrap runs that never
    # actually call compute_face_ray_mask().
    import face_ray_masking_ava256
    face_ray_masking_ava256.WRAP_SCRIPT_DIR = Path(wrap_script_dir)
    face_ray_masking_ava256.FACESCAPE_MASK_PATH = Path(facescape_mask_path)
    face_ray_masking_ava256.SAM_CHECKPOINT_PATH = Path(sam_checkpoint_path) if sam_checkpoint_path else None
    face_ray_masking_ava256.U2NET_CHECKPOINT_PATH = Path(u2net_checkpoint_path) if u2net_checkpoint_path else None
    mesh_utils.FACESCAPE_RUN_PIPELINE_PATH = Path(facescape_run_pipeline_path)
    # python-dotenv's load_dotenv() (called inside wrap_mesh()) does not
    # override an already-set env var by default, so setting these here
    # wins over whatever .env/.env.example would otherwise provide.
    os.environ["WrapCmd"] = str(faceform_wrap_cmd_path)
    os.environ["WrapLicense"] = str(faceform_wrap_license_path)

    actor_dir = mesh_utils.resolve_capture_dir(ava256_data_root, capture_id)
    mesh_utils.ensure_capture_extracted(actor_dir)

    import neutral_frame  # local import: needs SCRIPT_DIR already on sys.path
    neutral = neutral_frame.resolve_neutral_frame(capture_id, actor_dir, ava256_landmark_root)
    frame_id = frame_id_override or neutral["frame_id"]
    is_neutral_frame = frame_id == neutral["frame_id"]

    output_dir = ava256_output_root / capture_id / frame_id

    with _capture_pipeline_lock(ava256_output_root, capture_id) as acquired:
        if not acquired:
            print(f"NOTE: {capture_id} already being wrapped by another run -- skipping")
            return output_dir / "wrapped_mesh.obj"

        output_dir.mkdir(parents=True, exist_ok=True)

        # Idempotency: skip the (re)compute entirely if this exact phase
        # already succeeded, so re-running the batch scripts (automatic
        # sweep OR explicit --captures, which bypasses discover_unreviewed_
        # captures()'s own status filter) never redoes finished work.
        # label_tracker.json only tracks the NEUTRAL frame's own two-phase
        # status (unreviewed = bootstrap+propagation done, not yet skin-
        # augmented; unconfirmed/confirmed = skin-augmented done) -- for a
        # non-neutral (expression) frame, wrapped_mesh.obj existing at all
        # is itself sufficient, since there's no separate per-frame status
        # to track a second time through. Not applied to the bootstrap
        # (include_skin=False) neutral call -- that path already has its own
        # "only bootstrap if missing" guard one level up, in
        # run_neutral_skin_propagation.py.
        wrapped_mesh_path = output_dir / "wrapped_mesh.obj"
        if not force and wrapped_mesh_path.exists():
            if not is_neutral_frame:
                print(f"NOTE: {capture_id}/{frame_id} already wrapped -- skipping (use force=True/--force to re-wrap)")
                return wrapped_mesh_path
            if include_skin:
                existing_status = Ava256LabelTracker(label_tracker_path).get_status(capture_id)
                if existing_status in ("unconfirmed", "confirmed"):
                    print(
                        f"NOTE: {capture_id}/{frame_id} already skin-augmented-wrapped "
                        f"(status={existing_status!r}) -- skipping (use force=True/--force to re-wrap)"
                    )
                    return wrapped_mesh_path

        # 1) Target mesh: the already-registered scan, world space, then
        # transformed into the FLAME template's own (small) scale/frame --
        # see dynamic_transform.py's docstring for why this is required
        # (kinematic_tracking world space is ~100-1000 units vs. the
        # template's ~0.2, and wrapping without correcting for that produces
        # badly deformed output). Params are computed once per capture, from
        # the NEUTRAL frame's own scan bbox, and reused for every other frame.
        verts_world, faces = mesh_utils.build_world_mesh(actor_dir, frame_id, ava256_mesh_topology_path)
        transform_params_path = dynamic_transform.params_path(ava256_output_root, capture_id)
        if transform_params_path.exists():
            transform_params = dynamic_transform.load_params(transform_params_path)
        elif is_neutral_frame:
            transform_params = dynamic_transform.compute_params(default_neutral_mesh_path, verts_world)
            if not dry_run:
                dynamic_transform.save_params(transform_params_path, transform_params)
        else:
            raise RuntimeError(
                f"No dynamic_transform_params.npz for {capture_id} yet -- wrap the neutral frame "
                f"({neutral['frame_id']}) first so its scan bbox can seed the capture-level transform."
            )
        verts_transformed = dynamic_transform.transform(verts_world, transform_params)
        scan_obj_path = output_dir / "scan.obj"
        if not dry_run:
            mesh_utils.write_obj(scan_obj_path, verts_transformed, faces)

        # 2) Target correspondence. template_pot_path is the base 74-row
        # file; live-extend it with today's actual skin vertex set (see
        # mesh_utils.build_live_extended_pot's own docstring for why the
        # cached "_all" file can't be trusted as-is).
        extended_pot_path = mesh_utils.build_live_extended_pot(template_pot_path, default_neutral_mesh_path)
        pot_rows = _load_template_pot(extended_pot_path)
        prior_wrap_verts = None
        if include_skin and wrapped_mesh_path.exists():
            prior_mesh = trimesh.load(str(wrapped_mesh_path), process=False)
            prior_wrap_verts = np.asarray(prior_mesh.vertices, dtype=np.float64)

        # POT rows pin to FLAME-template vertices, so skin-row lookups need
        # the FLAME template's own faces (3931v/7800f), not this capture's
        # kinematic_tracking faces -- the prior wrap shares that exact
        # topology (a pure vertex-position deformation, never re-triangulated).
        flame_faces = _flame_faces(default_neutral_mesh_path)

        if is_neutral_frame:
            target_points, missing_indices = _build_target_correspondence(
                actor_dir, frame_id, pot_rows,
                include_skin=include_skin, prior_wrap_verts=prior_wrap_verts, faces=flame_faces,
                transform_params=transform_params,
            )
        else:
            # Non-neutral expression frame: keypoints_3d has no per-frame
            # skin ground truth, so both standard and skin rows come from
            # run_neutral_skin_propagation.py's own tracked output instead
            # (see _build_propagated_target_correspondence's docstring).
            target_points, missing_indices = _build_propagated_target_correspondence(
                actor_dir, ava256_landmark_root, capture_id, frame_id, pot_rows, transform_params,
            )
        correspondence_path = output_dir / "target_correspondence.json"
        if not dry_run:
            with correspondence_path.open("w", encoding="utf-8") as f:
                json.dump(target_points, f)

        if dry_run:
            print(
                f"[dry-run] {capture_id}/{frame_id}: would wrap with "
                f"{len(pot_rows) - len(missing_indices)}/{len(pot_rows)} correspondence rows "
                f"(missing: {sorted(missing_indices)})"
            )
            return wrapped_mesh_path

        # 2.5) Face ray masking -- only for the subsequent (skin-augmented)
        # wrap, using the bootstrap wrap's own mesh, per the user's explicit
        # instruction: no need on the keypoints_3d-only first wrap, since
        # there's no "prior wrap" mesh to rasterize/ray-cast against yet.
        if include_skin and use_face_ray_masking and face_ray_mask_path_override is not None:
            # Reuse an already-computed face_ray_mask.json instead of
            # recomputing SAM/U2Net from scratch -- the visible/occluded face
            # region from this actor's camera rig is a property of their face
            # geometry and the rig, not of the specific expression, so it
            # doesn't meaningfully change frame to frame within one capture.
            # Typically the capture's own neutral-frame skin-augmented wrap's
            # face_ray_mask.json (already the EXCLUDED-polarity file wrap_mesh()
            # expects, written by the block below when it computes fresh).
            wrap_script_module.FACE_MASK_PATH = Path(face_ray_mask_path_override)
            print(f"Reusing precomputed face_ray_mask at {face_ray_mask_path_override} for {capture_id}/{frame_id}'s wrap.")
        elif include_skin and use_face_ray_masking and prior_wrap_verts is not None:
            import face_ray_masking_ava256

            camera_ids = camera_utils.load_all_camera_ids(actor_dir)
            camera_params = {cid: camera_utils.load_camera(actor_dir, cid) for cid in camera_ids}
            pot_rows_by_index = {i: row for i, row in enumerate(pot_rows)}

            # face_ray_masking rasterizes against REAL camera K/Rt (world
            # scale), so undo the dynamic transform on the prior wrap's
            # vertices first -- they're in the small FLAME-comparable frame
            # (wrap_mesh() was fed the transformed scan/correspondence above).
            prior_wrap_verts_world = dynamic_transform.untransform(prior_wrap_verts, transform_params)

            print(f"Running face_ray_masking for {capture_id}/{frame_id} (using the bootstrap wrap's mesh)...")
            front_cams, right_cams, left_cams = face_ray_masking_ava256.classify_front_right_left(
                prior_wrap_verts_world, flame_faces, pot_rows_by_index, camera_ids, camera_params,
            )
            face_mask_indices = face_ray_masking_ava256.compute_face_ray_mask(
                prior_wrap_verts_world, flame_faces, actor_dir, frame_id, camera_ids, camera_params,
                front_cams, right_cams, left_cams, pot_rows_by_index=pot_rows_by_index,
            )
            # FaceForm/Wrap4D's SelectPolygons (FaceMask) node treats its
            # selection/fileName content as the EXCLUDED set, not included
            # (confirmed this session against template.wrap's own node graph
            # and by the user directly) -- compute_face_ray_mask() returns
            # the INCLUDED set, so write the complement, matching the same
            # fix already applied to face_ray_masking_ava256._reference_hit_faces().
            num_flame_faces = flame_faces.shape[0]
            face_mask_excluded = sorted(set(range(num_flame_faces)) - set(face_mask_indices))
            face_ray_mask_path = output_dir / "face_ray_mask.json"
            with face_ray_mask_path.open("w", encoding="utf-8") as f:
                json.dump(face_mask_excluded, f)
            # Included-set kept too, purely for later inspection/rendering --
            # never read by wrap_mesh() itself.
            with (output_dir / "face_ray_mask_included.json").open("w", encoding="utf-8") as f:
                json.dump(face_mask_indices, f)
            # Overrides the module-level FACE_MASK_PATH global that wrap_mesh()'s
            # own FaceMask-node wiring reads at call time -- safe: wrap_script_module
            # is a fresh exec'd module instance from THIS call's _load_wrap_script(),
            # not the shared wrap_script_facescape/run_wrap_script.py file itself or
            # any other process's copy of it.
            wrap_script_module.FACE_MASK_PATH = face_ray_mask_path
            print(f"Using face-ray-masked FaceMask ({len(face_mask_indices)} included faces, "
                  f"{len(face_mask_excluded)} excluded) instead of the static facescape_mask.txt for this wrap.")

        # 3) Wrap.
        wrap_mesh(
            target_mesh_path=str(scan_obj_path),
            target_correspondence_path=str(correspondence_path),
            output_path=str(wrapped_mesh_path),
            extra_indices_to_remove=missing_indices,
            neutral_mesh_path=str(default_neutral_mesh_path),
            pot_template_path=str(extended_pot_path),
        )

        if include_skin:
            tracker = Ava256LabelTracker(label_tracker_path)
            result = tracker.mark_unconfirmed(capture_id)
            if result is None:
                current = tracker.get_status(capture_id)
                print(f"NOTE: not moving {capture_id} to 'unconfirmed' -- current status is {current!r}")

        return wrapped_mesh_path


_flame_faces_cache: dict[Path, np.ndarray] = {}


def _flame_faces(neutral_mesh_path: Path) -> np.ndarray:
    neutral_mesh_path = Path(neutral_mesh_path)
    if neutral_mesh_path not in _flame_faces_cache:
        faces = []
        with open(neutral_mesh_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("f "):
                    faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
        _flame_faces_cache[neutral_mesh_path] = np.array(faces, dtype=np.int64)
    return _flame_faces_cache[neutral_mesh_path]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("capture_id", help="Ava-256 capture directory name, e.g. 20210810--1306--FXN596")
    parser.add_argument("--ava256-data-root", default=str(DEFAULTS["ava256_data_root"]))
    parser.add_argument("--ava256-landmark-root", default=str(DEFAULTS["ava256_landmark_root"]))
    parser.add_argument("--ava256-output-root", default=str(DEFAULTS["ava256_output_root"]))
    parser.add_argument("--label-tracker-path", default=str(DEFAULTS["label_tracker_path"]))
    parser.add_argument("--ava256-mesh-topology-path", default=str(DEFAULTS["ava256_mesh_topology_path"]))
    parser.add_argument("--wrap-script-path", default=str(DEFAULTS["wrap_script_path"]),
                         help="Path to wrap_script_facescape's run_wrap_script.py (wrap_mesh()/DEFAULT_NEUTRAL_MESH_PATH), reused unmodified")
    parser.add_argument("--template-pot-path", default=str(DEFAULTS["template_pot_path"]),
                         help="Path to the shared FLAME landmark correspondence file (neutral_correspondence_without_eyes_POT_all.json)")
    parser.add_argument("--template-wrap-path", default=str(DEFAULTS["template_wrap_path"]),
                         help="Path to the Wrap4D/R3DS template.wrap node-graph project file")
    parser.add_argument("--faceform-wrap-cmd-path", default=str(DEFAULTS["faceform_wrap_cmd_path"]),
                         help="Path to the Faceform Wrap installation's WrapCmd binary")
    parser.add_argument("--faceform-wrap-license-path", default=str(DEFAULTS["faceform_wrap_license_path"]),
                         help="Path to the Faceform Wrap license file")
    parser.add_argument("--sam-checkpoint-path", default=str(DEFAULTS["sam_checkpoint_path"]),
                         help="Path to the SAM ViT-H checkpoint (sam_vit_h_4b8939.pth) used by face_ray_masking's segmentation")
    parser.add_argument("--u2net-checkpoint-path", default=str(DEFAULTS["u2net_checkpoint_path"]),
                         help="Path to the U2Net checkpoint (u2net_human_seg.pth) used by face_ray_masking's segmentation")
    parser.add_argument("--wrap-script-dir", default=str(DEFAULTS["wrap_script_dir"]),
                         help="Path to the wrap_script/ directory (segment.py, model/, data_loader.py) face_ray_masking imports from")
    parser.add_argument("--facescape-mask-path", default=str(DEFAULTS["facescape_mask_path"]),
                         help="Path to facescape_mask.txt, the curated face-region reference face_ray_masking intersects against")
    parser.add_argument("--blendshape-root", default=str(DEFAULTS["blendshape_root"]),
                         help="Path to the FLAME blendshape .obj directory (neutral.obj + shape_*/exp_*.obj) BlendWrapping retargets against")
    parser.add_argument("--facescape-run-pipeline-path", default=str(DEFAULTS["facescape_run_pipeline_path"]),
                         help="Path to wrap_script_facescape/run_pipeline.py, whose load_skin_vertex_indices()/_compute_skin_pot_entries() "
                              "mesh_utils.build_live_extended_pot() reuses read-only")
    parser.add_argument("--no-skin", action="store_true", help="Force include_skin=False (manual bootstrap-only rerun)")
    parser.add_argument("--no-face-ray-masking", action="store_true",
                         help="Skip face_ray_masking for the skin-augmented wrap and fall back to the static facescape_mask.txt "
                              "(face_ray_masking never runs on the --no-skin bootstrap wrap regardless of this flag)")
    parser.add_argument("--face-ray-mask-path", default=None,
                         help="Reuse an already-computed face_ray_mask.json (e.g. this capture's neutral frame's own, from its "
                              "output_dir/face_ray_mask.json) instead of recomputing SAM/U2Net for this frame -- the visible/"
                              "occluded face region is a property of the actor+rig, not the expression, so it's safe and much "
                              "faster to compute once per capture and reuse. Takes priority over --no-face-ray-masking.")
    parser.add_argument("--frame", default=None, help="Override the auto-resolved neutral frame (manual/debug)")
    parser.add_argument("--force", action="store_true",
                         help="Re-wrap even if this frame/phase already succeeded (skips the label_tracker.json / "
                              "wrapped_mesh.obj-existence idempotency check)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_path = run_pipeline(
        args.capture_id,
        ava256_data_root=Path(args.ava256_data_root),
        ava256_landmark_root=Path(args.ava256_landmark_root),
        ava256_output_root=Path(args.ava256_output_root),
        label_tracker_path=Path(args.label_tracker_path),
        ava256_mesh_topology_path=Path(args.ava256_mesh_topology_path),
        wrap_script_path=Path(args.wrap_script_path),
        template_pot_path=Path(args.template_pot_path),
        template_wrap_path=Path(args.template_wrap_path),
        faceform_wrap_cmd_path=Path(args.faceform_wrap_cmd_path),
        faceform_wrap_license_path=Path(args.faceform_wrap_license_path),
        sam_checkpoint_path=Path(args.sam_checkpoint_path),
        u2net_checkpoint_path=Path(args.u2net_checkpoint_path),
        wrap_script_dir=Path(args.wrap_script_dir),
        facescape_mask_path=Path(args.facescape_mask_path),
        blendshape_root=Path(args.blendshape_root),
        facescape_run_pipeline_path=Path(args.facescape_run_pipeline_path),
        include_skin=not args.no_skin,
        # Ava-256 frame ids are always 6-digit zero-padded on disk (kinematic_
        # tracking/<id>.ply, keypoints_3d/<id>.npy, cam<cid>/<id>.avif inside
        # the image zips, ...) -- frame_list.csv itself stores them unpadded
        # (e.g. "32939"), and every other frame_id source in this pipeline
        # already normalizes via .zfill(6) (neutral_frame.py, run_neutral_
        # skin_propagation.py, render_wrapped_landmarks.py) except this one,
        # a raw user-typed --frame value, which a 5-digit id would silently
        # fail to resolve against those on-disk names.
        frame_id_override=args.frame.strip().zfill(6) if args.frame else None,
        use_face_ray_masking=not args.no_face_ray_masking,
        face_ray_mask_path_override=Path(args.face_ray_mask_path) if args.face_ray_mask_path else None,
        dry_run=args.dry_run,
        force=args.force,
    )
    print(f"Wrapped mesh at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
