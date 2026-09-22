# Handoff: running this pipeline on a different cluster

This pipeline wraps a FLAME template onto Ava-256 capture data via R3DS
Wrap4D (FaceForm Wrap), using landmark correspondences sourced from
`keypoints_3d` (neutral frame) or RAFT/CoTracker propagation (every other
frame). `wrap_script_ava256/` is now self-contained: every FLAME/wrap-graph/
segmentation asset it needs (`blendshape/`, `segmentation/` [`segment.py` +
`model/` + `data_loader.py`], `keypoint_front.txt`/`keypoint_right.txt`/
`keypoint_left.txt`/`keypoint_under.txt`, `skin_landmarks.py` (self-resolving
-- its own `ROOT = Path(__file__).resolve().parent` just needed to BE this
directory to read the local `keypoint_*.txt` copies instead of `wrap_script/`'s
-- see below for why the file that reads these used to bypass them entirely),
`neutral_correspondence_without_eyes_POT.json`, `template.wrap`,
`neutral.obj`, `facescape_mask.txt`, `face_topology.obj`,
`landmark_lines.npy`, `facescape_run_pipeline.py` (stripped to only
`load_skin_vertex_indices()`/`_compute_skin_pot_entries()` -- the only two
things `mesh_utils.py` actually calls from it; this removed a real,
previously-missed FaceView dependency, see below), `triangulate.py` (stripped
of its original `backend.app.core.config` dependency -- see its own module
docstring), a local git clone of CoTracker (`co-tracker/`), and a real
working `run_wrap_script.py`) lives directly in this directory, copied from
`wrap_script_facescape/`/`wrap_script/` -- those two directories are
FaceScape's own separate production pipeline and are no longer a hard
dependency for Ava-256 to run, and there is no longer any dependency on the
FaceView repo at all -- audited directly (every hardcoded
`/scratch/ondemand32/irwinngo/(wrap_script|wrap_script_facescape|FaceView)`
reference grepped across every file actually reachable from
`run_wrapping.sh`/`run_neutral_propagation.sh`'s own entry points, not just
spot-checked) as of this pass. The only genuinely-external dependencies left
are the Ava-256 dataset itself, the FaceForm Wrap application + license, and
model checkpoints (SAM/U2Net/RAFT/CoTracker's `scaled_offline.pth` -- code for
CoTracker is the local clone above, but its checkpoint is handed off like
every other model weight, not bundled into the git repo) -- every path/flag
for these is in the table below. All of them are still fully overridable via
CLI flag (Python) / environment variable (shell) in case a caller wants to
point back at shared/external copies instead.

## Primary entry points

Two shell scripts, both in `wrap_script_ava256/`:

- **`run_wrapping.sh <CAPTURE_ID> [extra flags]`** -- wraps `run_pipeline.py`.
  Bootstrap (keypoints_3d-only) wrap of the capture's neutral frame, then the
  skin-augmented wrap (face_ray_masking + ear/nose force-include). Pass
  `--no-skin` for bootstrap-only, `--frame FRAME_ID` to wrap a specific
  non-neutral frame (requires that frame's landmarks already propagated --
  run `run_neutral_propagation.sh` first), `--dry-run` to check correspondence
  coverage without invoking WrapCmd.
- **`run_neutral_propagation.sh <CAPTURE_ID> [extra flags]`** -- wraps
  `run_neutral_skin_propagation.py`. Bootstrap-wraps the neutral frame
  in-process if missing, selects cameras per-landmark, then
  RAFT/CoTracker-propagates every landmark to every other frame in the
  capture. Pass `--segments SEG_ID ...` / `--max-frames-per-segment N` to
  restrict to a subset (full-capture propagation is thousands of frames and
  expensive) or `--dry-run` to preview scope/camera selection.

Both scripts read every path below as an environment variable, falling back
to this cluster's current value if unset:

```
MODELS_PATH=/my/models \
AVA256_DATA_ROOT=/my/ava-256 \
./run_wrapping.sh 20210810--1306--FXN596
```

or export them once at the top of a copy of the script.

## External dependencies and their override flag/variable

| Dependency | What it is | Shell var | Python CLI flag |
|---|---|---|---|
| Ava-256 dataset | Root containing `<capture_id>/decoder/...` per capture | `AVA256_DATA_ROOT` | `--ava256-data-root` |
| Landmark output root | Where `neutral_frame.json` + propagated `{group}_{frame}.json` files are written | `AVA256_LANDMARK_ROOT` | `--ava256-landmark-root` |
| Wrap output root | Where `scan.obj`/`target_correspondence.json`/`wrapped_mesh.obj`/`dynamic_transform_params.npz` are written, per capture/frame | `AVA256_OUTPUT_ROOT` | `--ava256-output-root` |
| Label tracker | `label_tracker.json`, tracks each capture's `unlabeled -> unreviewed -> unconfirmed` status | `LABEL_TRACKER_PATH` | `--label-tracker-path` |
| FLAME mesh topology | `face_topology.obj` (local copy), the faces used to interpret Ava-256's `kinematic_tracking` vertex-only data | `AVA256_MESH_TOPOLOGY_PATH` (default `${SCRIPT_DIR}/face_topology.obj`) | `--ava256-mesh-topology-path` |
| Wrap driver | `run_wrap_script.py` (local copy -- `wrap_mesh()`/`DEFAULT_NEUTRAL_MESH_PATH`) | `WRAP_SCRIPT_PATH` (default `${SCRIPT_DIR}/run_wrap_script.py`) | `--wrap-script-path` |
| FLAME landmark correspondence (template side) | `neutral_correspondence_without_eyes_POT.json` (local copy), the 74-standard-row base file | `TEMPLATE_POT_PATH` (default `${SCRIPT_DIR}/neutral_correspondence_without_eyes_POT.json`) | `--template-pot-path` |
| Wrap4D project graph | `template.wrap` (local copy) -- the node graph (NeutralInput/TargetInput/FaceMask/BlendWrapping/Wrapping/SaveWrap) WrapCmd computes against | `TEMPLATE_WRAP_PATH` (default `${SCRIPT_DIR}/template.wrap`) | `--template-wrap-path` |
| FaceForm Wrap binary | `WrapCmd` executable from the R3DS Wrap4D install -- genuinely external, not copyable | `FACEFORM_WRAP_PATH` | `--faceform-wrap-cmd-path` |
| FaceForm Wrap license | The `.lic` file for the above | `FACEFORM_WRAP_LICENSE_PATH` | `--faceform-wrap-license-path` |
| SAM checkpoint | `sam_vit_h_4b8939.pth`, face_ray_masking's segmentation model | `SAM_CHECKPOINT_PATH` (default `${MODELS_PATH}/sam_vit_h_4b8939.pth`) | `--sam-checkpoint-path` |
| U2Net checkpoint | `u2net_human_seg.pth`, finds face_ray_masking's SAM point-prompt | `U2NET_CHECKPOINT_PATH` (default `${MODELS_PATH}/u2net_human_seg.pth`) | `--u2net-checkpoint-path` |
| RAFT checkpoint | `raft_large_C_T_SKHT_V2-ff5fadd5.pth`, seeds propagation across segment boundaries | `RAFT_CHECKPOINT_PATH` (default `${MODELS_PATH}/raft_large_...pth`) | `--raft-checkpoint-path` |
| Segmentation directory | `segmentation/` (local copy of `segment.py` + `model/` [U2NET arch] + `data_loader.py`) | `WRAP_SCRIPT_DIR` (default `${SCRIPT_DIR}/segmentation`) | `--wrap-script-dir` |
| facescape_mask.txt | `facescape_mask.txt` (local copy) -- curated face-region reference face_ray_masking's own hit-set is intersected against | `FACESCAPE_MASK_PATH` (default `${SCRIPT_DIR}/facescape_mask.txt`) | `--facescape-mask-path` |
| Blendshape basis | `blendshape/` (local copy) -- FLAME `shape_*.obj`/`exp_*.obj`/`neutral.obj` BlendWrapping retargets against | `BLENDSHAPE_ROOT` (default `${SCRIPT_DIR}/blendshape`) | `--blendshape-root` |
| Skin-region vertex list source | `facescape_run_pipeline.py` (local copy, rewritten minimal -- only `load_skin_vertex_indices()`/`_compute_skin_pot_entries()`, no FaceView/`backend.app.*` dependency) | `FACESCAPE_RUN_PIPELINE_PATH` (default `${SCRIPT_DIR}/facescape_run_pipeline.py`) | `--facescape-run-pipeline-path` |
| Skin region definitions | `skin_landmarks.py` (local copy) + `keypoint_front.txt`/`keypoint_right.txt`/`keypoint_left.txt`/`keypoint_under.txt` (local copies, read relative to wherever `skin_landmarks.py` itself lives -- no separate flag needed) -- which FLAME vertices count as "skin" landmarks, per region | (not a flag -- resolves via `skin_landmarks.py`'s own location) | (not a flag) |
| triangulate.py | `triangulate.py` (local copy, stripped of its original FaceView/`backend.app.core.config` dependency) -- multi-camera 3D triangulation used to fuse propagated per-camera 2D tracks | `TRIANGULATE_SCRIPT_PATH` (default `${SCRIPT_DIR}/triangulate.py`) | `--triangulate-script-path` |
| CoTracker code | `co-tracker/` -- local `git clone` of `https://github.com/facebookresearch/co-tracker.git` (has its own nested `.git/`; decide submodule-vs-plain when folding into the outer repo) | `COTRACKER_ROOT` (default `${SCRIPT_DIR}/co-tracker`) | `--cotracker-root` |
| CoTracker checkpoint | `scaled_offline.pth` -- genuinely external, handed off like every other model weight (not part of the git clone) | `COTRACKER_CHECKPOINT_PATH` (default `${MODELS_PATH}/scaled_offline.pth`) | `--cotracker-checkpoint-path` |
| Keyline video output root | Ava-256's own equivalent of Nersemble's `LINE_RENDERS_ROOT` -- `run_pipeline_batch.py`'s automatic post-capture keyline video | (batch script only, no shell wrapper yet) | `--line-renders-root` |

`${SCRIPT_DIR}` above means "this script's own directory" (`wrap_script_ava256/`
itself) -- these defaults are self-contained, no path outside this directory.
`MODELS_PATH` (default `/scratch/ondemand32/irwinngo/models`) is a convenience
root the SAM/U2Net/RAFT checkpoint variables default under -- set it once to
relocate all three, or override any individual checkpoint path separately.

## Known gaps (lower priority, not yet wired to a flag)

- `render_overlay_mesh.py` (a standalone debug/verification script) still
  hardcodes `sys.path.insert(0, "/scratch/ondemand32/irwinngo/FaceView")` and
  a `FaceView/ava-256/assets/face_topology.obj` default. Confirmed NOT
  reachable from either real entry point (`run_wrapping.sh`/
  `run_neutral_propagation.sh` and everything they import) -- grepped every
  file in that actual reachable set, nothing imports this one -- so it does
  not affect the pipeline's self-containment, only this one debug tool if run
  directly.

Everything else previously listed here is resolved: `run_pipeline_batch.py`/
`run_neutral_skin_propagation_batch.py` and `render_overlay_frame.py` forward
all the same flags as the single-capture scripts; `wrap_script_ava256/
run_wrap_script.py` is a real working local copy, not dead code; `render_
wrapped_landmarks.py`'s `DEFAULT_LANDMARK_LINES_PATH`/`--neutral-mesh-path`
default locally; `extract_obj.py`/`render_overlay.py`'s own `FACE_TOPOLOGY_
PATH` module defaults (`mesh_utils.build_world_mesh()` already overrode
`extract_obj.py`'s per-call, but the bare default was still external, and
`render_overlay.py`'s copy was on a function neither `camera_utils.py` nor
this pipeline ever actually calls) now default locally too; `face_ray_
masking_ava256.py`'s own `WRAP_SCRIPT_DIR`/`FACESCAPE_MASK_PATH` module
defaults (previously overridden correctly via `run_pipeline.py`'s monkeypatch
at runtime, but still external as a bare default) now default locally.
`facescape_run_pipeline.py`'s old `from render_overlay_frame import
render_overlay_frame` landmine no longer exists -- that file was rewritten
from scratch as a minimal, ~70-line file containing only the two functions
`mesh_utils.py` actually calls, with no `render_overlay_frame`/`wrap_mesh`/
`triangulate`/`backend.app.*` imports at all (previously a genuine, previously-
missed `sys.path.insert(0, ".../FaceView")` + three-way `backend.app.*`
dependency -- the actual explanation for why `keypoint_front.txt`'s local
copy was never being read: it went through this file's hardcoded external
`wrap_script/skin_landmarks.py` reference instead). `skin_landmarks.py`
itself is now a local copy too (byte-identical to `wrap_script/`'s -- already
self-contained via its own `Path(__file__).resolve().parent`, it just needed
to physically live here).
`FACEVIEW_ROOT`/`--faceview-root` has been removed entirely (was only ever
needed for `triangulate.py`'s now-removed `get_settings` import) -- there is
no FaceView dependency left anywhere in this pipeline.

## Verifying a new setup

```
./run_wrapping.sh <CAPTURE_ID> --no-skin --dry-run
```

prints how many of the 45 standard-landmark correspondence rows resolved
without invoking WrapCmd at all -- a quick way to confirm `AVA256_DATA_ROOT`
and the keypoints_3d extraction path are right before spending any GPU/Wrap
license time.
