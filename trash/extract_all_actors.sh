#!/usr/bin/env bash
# Runs extract_ava256_zips.py (head_pose.zip + registration_vertices.zip) for
# every actor directory under ACTOR_ROOT, one at a time. Uses --actor-dir
# directly (not --actor <code>) since we're already iterating full paths from
# ACTOR_ROOT itself -- no need to re-resolve the code back through it.
#
# Doesn't use `set -e`: one actor's zip being missing/corrupt shouldn't abort
# the rest, so failures are collected and reported in a summary at the end
# instead.
#
# Skips an actor if both zips are already FULLY extracted (every member
# present at its destination path with a matching file size -- a quick
# central-directory check, not a full re-extraction, so it's cheap even on
# slow NFS) so re-running this script after a partial/interrupted prior run,
# or just to pick up newly-added actors, doesn't redo already-finished work.
#
# Usage:
#   ./extract_all_actors.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTOR_ROOT="/scratch/thirty/irwinngo/ava-256"

# Exits 0 if every member of $1 (a zip) already exists under $2 (that zip's
# own destination dir) with a matching size; exits 1 otherwise (missing,
# short, or the zip itself is missing/unreadable).
zip_fully_extracted() {
    python3 - "$1" "$2" <<'EOF'
import sys, zipfile, os

zip_path, dest = sys.argv[1], sys.argv[2]
try:
    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()
except OSError:
    sys.exit(1)

for info in infos:
    if info.is_dir():
        continue
    path = os.path.join(dest, info.filename)
    if not os.path.isfile(path) or os.path.getsize(path) != info.file_size:
        sys.exit(1)
sys.exit(0)
EOF
}

succeeded=()
skipped=()
failed=()

for actor_dir in "$ACTOR_ROOT"/*/; do
    actor_dir="${actor_dir%/}"
    actor="$(basename "$actor_dir")"
    head_pose_zip="$actor_dir/decoder/head_pose/head_pose.zip"
    kinematic_zip="$actor_dir/decoder/kinematic_tracking/registration_vertices.zip"

    if zip_fully_extracted "$head_pose_zip" "$(dirname "$head_pose_zip")" \
        && zip_fully_extracted "$kinematic_zip" "$(dirname "$kinematic_zip")"; then
        echo "=== $actor: already extracted, skipping ==="
        skipped+=("$actor")
        continue
    fi

    echo "=== $actor ==="
    if python3 "$SCRIPT_DIR/extract_ava256_zips.py" --actor-dir "$actor_dir"; then
        succeeded+=("$actor")
    else
        echo "FAILED: $actor" >&2
        failed+=("$actor")
    fi
    echo
done

echo "=== Summary ==="
echo "Succeeded: ${#succeeded[@]}"
echo "Skipped (already extracted): ${#skipped[@]}"
echo "Failed: ${#failed[@]}"
if [ "${#failed[@]}" -gt 0 ]; then
    printf '  %s\n' "${failed[@]}"
    exit 1
fi
