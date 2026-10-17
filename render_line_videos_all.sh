#!/usr/bin/env bash
# Renders the keyline overlay video (+ frame CSV) of every Ava-256 capture in
# WRAPS_ROOT via render_capture_line_video.py, PARALLEL captures at a time.
# Already-rendered/archived captures and captures without raw data are skipped
# by the Python script, so this is safe to rerun (e.g. after downloading more
# raw captures).
#
# Usage: ./render_line_videos_all.sh [PARALLEL]   (default 8)
set -uo pipefail

PARALLEL="${1:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/scratch/ondemand32/irwinngo/envs/faceview_venv/bin/python}"
WRAPS_ROOT="/scratch/ondemand32/irwinngo/ava256_killarney_hz"
LOG_DIR="/scratch/ondemand32/irwinngo/line_renders_ava256/logs"
mkdir -p "$LOG_DIR"

export PYTHON SCRIPT_DIR LOG_DIR
find "$WRAPS_ROOT" -mindepth 1 -maxdepth 1 -type d -name '*--*' -printf '%f\n' | sort |
  xargs -P "$PARALLEL" -I{} bash -c '
    if "$PYTHON" "$SCRIPT_DIR/render_capture_line_video.py" "$1" > "$LOG_DIR/$1.log" 2>&1; then
      echo "$(tail -n 1 "$LOG_DIR/$1.log")"
    else
      echo "FAIL $1 (see $LOG_DIR/$1.log)"
    fi
  ' _ {}
echo "All captures processed."
