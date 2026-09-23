#!/usr/bin/env bash
# Batch counterpart to run_wrapping.sh: runs run_pipeline_batch.py, which
# sweeps every capture at label status "unreviewed" (or an explicit
# --captures list), wrapping each capture's neutral frame (bootstrap +
# skin-augmented) FIRST, then -- unless --neutral-only is passed -- every
# other decoder/frame_list.csv frame that run_neutral_propagation_batch.sh
# has already produced landmark data for (frames without landmarks yet are
# skipped, not failed). One run_pipeline.py subprocess per FRAME (not per
# capture) -- pass --segments/--max-frames-per-segment to restrict the sweep.
# Also fires the automatic keyline-video step (run_wrapping.sh's
# single-capture path doesn't -- that's a batch-only feature since it renders
# every wrapped frame for a capture once all of them are done).
#
# HANDOFF: every path below is an environment variable with the value
# currently used on THIS cluster as its default -- override any of them by
# exporting the var (or prefixing the invocation) before running this script
# on a different cluster. See HANDOFF.md for what each one is and where to
# get it. Same variable names as run_wrapping.sh -- set them once and both
# scripts pick them up.
#
# Usage:
#   ./run_wrapping_batch.sh [--captures ID1 ID2 ...] [--parallel N] [--gpu-ids 0 1 2 3]
#       [extra run_pipeline_batch.py flags, e.g. --no-video --no-face-ray-masking --log-dir /path --poll --dry-run]
#   ./run_wrapping_batch.sh                          # sweep every capture at 'unreviewed', one at a time
#   ./run_wrapping_batch.sh --captures A B C --parallel 3 --gpu-ids 0 1 2
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# --- Automatic keyline-video output (batch-only feature) ---------------------
: "${LINE_RENDERS_ROOT:=/scratch/ondemand32/irwinngo/line_renders_ava256}"

# --- Parallelism ---------------------------------------------------------------
# Leave PARALLEL=1/GPU_IDS unset to run one capture at a time on the default
# GPU. Set both to fan out across captures -- e.g. PARALLEL=4 GPU_IDS="0 1 2 3"
# round-robins each capture's subprocess onto CUDA_VISIBLE_DEVICES 0/1/2/3.
: "${PARALLEL:=1}"
GPU_IDS_ARGS=()
if [ -n "${GPU_IDS:-}" ]; then
    # shellcheck disable=SC2206
    GPU_IDS_ARGS=(--gpu-ids ${GPU_IDS})
fi

: "${PYTHON_BIN:=/scratch/ondemand32/irwinngo/envs/faceview_venv/bin/python}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/run_pipeline_batch.py" \
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
    --line-renders-root "${LINE_RENDERS_ROOT}" \
    --parallel "${PARALLEL}" \
    "${GPU_IDS_ARGS[@]}" \
    "$@"
