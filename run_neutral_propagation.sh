#!/usr/bin/env bash
# Runs Ava-256 landmark propagation (run_neutral_skin_propagation.py) for one
# capture: bootstrap-wraps the neutral frame in-process if missing, selects
# cameras per-landmark, then RAFT/CoTracker-propagates every landmark to
# every other frame (or a --segments/--max-frames-per-segment subset). Run
# run_wrapping.sh afterward with --frame FRAME_ID for any non-neutral frame
# you want an actual wrap of -- this script only produces the propagated
# landmark JSON files (and the neutral frame's own bootstrap wrap).
#
# HANDOFF: every path below is an environment variable with the value
# currently used on THIS cluster as its default -- override any of them by
# exporting the var (or prefixing the invocation) before running this script
# on a different cluster. See HANDOFF.md for what each one is and where to
# get it.
#
# Usage:
#   ./run_neutral_propagation.sh <CAPTURE_ID> [extra run_neutral_skin_propagation.py flags,
#       e.g. --segments EXP_jaw001 EXP_lip001 --max-frames-per-segment 5 --dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <CAPTURE_ID> [extra run_neutral_skin_propagation.py flags...]" >&2
    exit 1
fi
CAPTURE_ID="$1"
shift

# --- Ava-256 dataset ---------------------------------------------------------
: "${AVA256_DATA_ROOT:=/scratch/thirty/irwinngo/ava-256}"
: "${AVA256_LANDMARK_ROOT:=/scratch/ondemand32/irwinngo/landmarks_ava-256/}"
: "${AVA256_OUTPUT_ROOT:=/scratch/ondemand32/irwinngo/ava_256_output/}"
: "${LABEL_TRACKER_PATH:=/scratch/ondemand32/irwinngo/label_tracker.json}"
: "${AVA256_MESH_TOPOLOGY_PATH:=${SCRIPT_DIR}/face_topology.obj}"

# --- FaceScape wrap driver (needed for the in-process bootstrap wrap) -------
# All consolidated directly into this directory (self-contained -- see
# HANDOFF.md) rather than reaching into wrap_script_facescape/wrap_script/.
: "${WRAP_SCRIPT_PATH:=${SCRIPT_DIR}/run_wrap_script.py}"
: "${TEMPLATE_POT_PATH:=${SCRIPT_DIR}/neutral_correspondence_without_eyes_POT.json}"
: "${TEMPLATE_WRAP_PATH:=${SCRIPT_DIR}/template.wrap}"
: "${BLENDSHAPE_ROOT:=${SCRIPT_DIR}/blendshape}"
: "${FACESCAPE_MASK_PATH:=${SCRIPT_DIR}/facescape_mask.txt}"
: "${FACESCAPE_RUN_PIPELINE_PATH:=${SCRIPT_DIR}/facescape_run_pipeline.py}"
: "${WRAP_SCRIPT_DIR:=${SCRIPT_DIR}/segmentation}"

# --- FaceForm Wrap (R3DS Wrap4D) application ---------------------------------
: "${FACEFORM_WRAP_PATH:=/scratch/ondemand32/irwinngo/faceform/Faceform_Wrap_2025.11.14_Linux/WrapCmd}"
: "${FACEFORM_WRAP_LICENSE_PATH:=/scratch/ondemand32/irwinngo/faceform/Faceform_Wrap_License_1035244_10483637.lic}"

# --- Model checkpoints --------------------------------------------------------
# SAM/U2Net are only used by the in-process bootstrap wrap's face_ray_masking
# step, which never actually runs during propagation (bootstrap is
# include_skin=False) -- kept here anyway so this script's flags stay a
# strict superset of run_wrapping.sh's, in case that changes.
: "${MODELS_PATH:=/scratch/ondemand32/irwinngo/models}"
: "${SAM_CHECKPOINT_PATH:=${MODELS_PATH}/sam_vit_h_4b8939.pth}"
: "${U2NET_CHECKPOINT_PATH:=${MODELS_PATH}/u2net_human_seg.pth}"
: "${RAFT_CHECKPOINT_PATH:=${MODELS_PATH}/raft_large_C_T_SKHT_V2-ff5fadd5.pth}"

# --- CoTracker -----------------------------------------------------------------
# Local git clone (see HANDOFF.md) -- code lives here, self-contained; the
# checkpoint itself lives under MODELS_PATH like SAM/U2Net/RAFT's (handed off
# via scp separately, not part of the git-cloned code).
: "${COTRACKER_ROOT:=${SCRIPT_DIR}/co-tracker}"
: "${COTRACKER_CHECKPOINT_PATH:=${MODELS_PATH}/scaled_offline.pth}"
: "${TRIANGULATE_SCRIPT_PATH:=${SCRIPT_DIR}/triangulate.py}"

# --- Camera-selection tuning ---------------------------------------------------
: "${MIN_CAMERAS_PER_LANDMARK:=5}"
: "${ANGLE_THRESHOLD_DEG:=80.0}"

: "${PYTHON_BIN:=/scratch/ondemand32/irwinngo/envs/faceview_venv/bin/python}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_neutral_skin_propagation.py" "${CAPTURE_ID}" \
    --ava256-data-root "${AVA256_DATA_ROOT}" \
    --ava256-landmark-root "${AVA256_LANDMARK_ROOT}" \
    --ava256-output-root "${AVA256_OUTPUT_ROOT}" \
    --label-tracker-path "${LABEL_TRACKER_PATH}" \
    --ava256-mesh-topology-path "${AVA256_MESH_TOPOLOGY_PATH}" \
    --wrap-script-path "${WRAP_SCRIPT_PATH}" \
    --template-pot-path "${TEMPLATE_POT_PATH}" \
    --template-wrap-path "${TEMPLATE_WRAP_PATH}" \
    --faceform-wrap-cmd-path "${FACEFORM_WRAP_PATH}" \
    --faceform-wrap-license-path "${FACEFORM_WRAP_LICENSE_PATH}" \
    --sam-checkpoint-path "${SAM_CHECKPOINT_PATH}" \
    --u2net-checkpoint-path "${U2NET_CHECKPOINT_PATH}" \
    --wrap-script-dir "${WRAP_SCRIPT_DIR}" \
    --facescape-mask-path "${FACESCAPE_MASK_PATH}" \
    --blendshape-root "${BLENDSHAPE_ROOT}" \
    --facescape-run-pipeline-path "${FACESCAPE_RUN_PIPELINE_PATH}" \
    --triangulate-script-path "${TRIANGULATE_SCRIPT_PATH}" \
    --cotracker-root "${COTRACKER_ROOT}" \
    --cotracker-checkpoint-path "${COTRACKER_CHECKPOINT_PATH}" \
    --raft-checkpoint-path "${RAFT_CHECKPOINT_PATH}" \
    --min-cameras-per-landmark "${MIN_CAMERAS_PER_LANDMARK}" \
    --angle-threshold-deg "${ANGLE_THRESHOLD_DEG}" \
    "$@"
