#!/usr/bin/env bash
# Runs the Ava-256 FLAME-wrap pipeline (run_pipeline.py) for one capture:
# bootstrap (keypoints_3d-only) wrap of the neutral frame, then the
# skin-augmented wrap (face_ray_masking + ear/nose force-include). For a
# non-neutral frame (--frame FRAME_ID), sources correspondence from
# run_neutral_skin_propagation.py's own tracked output instead -- run
# run_neutral_propagation.sh for that capture FIRST if you're wrapping an
# expression frame.
#
# HANDOFF: every path below is an environment variable with the value
# currently used on THIS cluster as its default -- override any of them by
# exporting the var (or prefixing the invocation, e.g.
# `AVA256_DATA_ROOT=/my/data ./run_wrapping.sh 20210810--1306--FXN596`)
# before running this script on a different cluster. See HANDOFF.md for what
# each one is and where to get it.
#
# Usage:
#   ./run_wrapping.sh <CAPTURE_ID> [extra run_pipeline.py flags, e.g. --no-skin --frame 041323 --dry-run]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <CAPTURE_ID> [extra run_pipeline.py flags...]" >&2
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

# --- FaceScape wrap driver (template.wrap graph + FLAME assets) --------------
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

# --- Model checkpoints (face_ray_masking's segmentation) ---------------------
: "${MODELS_PATH:=/scratch/ondemand32/irwinngo/models}"
: "${SAM_CHECKPOINT_PATH:=${MODELS_PATH}/sam_vit_h_4b8939.pth}"
: "${U2NET_CHECKPOINT_PATH:=${MODELS_PATH}/u2net_human_seg.pth}"

: "${PYTHON_BIN:=/scratch/ondemand32/irwinngo/envs/faceview_venv/bin/python}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_pipeline.py" "${CAPTURE_ID}" \
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
    "$@"
