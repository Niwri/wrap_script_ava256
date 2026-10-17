#!/usr/bin/env bash
# Runs render_capture_line_video.py for every Ava-256 capture whose raw data is
# downloaded: DATA_ROOT/<CAPTURE>/decoder/image/cam*.zip present (plus
# decoder/camera_calibration.json) and a wraps folder in WRAPS_ROOT.
# Captures already rendered or already Done are skipped by the Python script,
# so rerun this after each download batch to pick up only the new captures.
#
# Usage: ./render_line_videos_downloaded.sh [PARALLEL]   (default 8)
#        DRY_RUN=1 ./render_line_videos_downloaded.sh    (list the captures only)
set -uo pipefail

PARALLEL="${1:-8}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/scratch/ondemand32/irwinngo/envs/faceview_venv/bin/python}"
DATA_ROOT="/scratch/thirty/irwinngo/ava-256"
WRAPS_ROOT="/scratch/ondemand32/irwinngo/ava256_killarney_hz"
LOG_DIR="/scratch/ondemand32/irwinngo/line_renders_ava256/logs"
mkdir -p "$LOG_DIR"

captures=()
for capture_dir in "$DATA_ROOT"/*--*/; do
  capture="$(basename "$capture_dir")"
  compgen -G "$capture_dir/decoder/image/cam*.zip" > /dev/null || continue
  [[ -s "$capture_dir/decoder/camera_calibration.json" ]] || { echo "WAIT $capture: camera zips but camera_calibration.json missing or empty"; continue; }
  [[ -d "$WRAPS_ROOT/$capture" ]] || { echo "NOTE $capture: no wraps in $WRAPS_ROOT"; continue; }
  captures+=("$capture")
done
echo "${#captures[@]} capture(s) with camera zips and wraps"

if [[ -n "${DRY_RUN:-}" ]]; then
  printf '  %s\n' "${captures[@]}"
  exit 0
fi

export PYTHON SCRIPT_DIR LOG_DIR
printf '%s\n' "${captures[@]}" |
  xargs -r -P "$PARALLEL" -I{} bash -c '
    if "$PYTHON" "$SCRIPT_DIR/render_capture_line_video.py" "$1" > "$LOG_DIR/$1.log" 2>&1; then
      echo "$(tail -n 1 "$LOG_DIR/$1.log")"
    else
      echo "FAIL $1 (see $LOG_DIR/$1.log)"
    fi
  ' _ {}
echo "All downloaded captures processed."
