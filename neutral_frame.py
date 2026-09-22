"""Neutral-frame resolution + cache for the Ava-256 pipeline.

Among a capture's EXP_neutral_peak rows in decoder/frame_list.csv, picks
whichever frame's keypoints_3d/<frame>.npy has the most of the landmarks
needed for propagation actually present (confirmed with the user: coverage
of the specific ~45 landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID ids, not
total keypoint count), tie-broken by lowest frame_id. Cached once to
{AVA_256_LANDMARK_ROOT}/<CAPTURE>/neutral_frame.json so repeat runs don't
re-scan frame_list.csv + every candidate's keypoints_3d.
"""
from __future__ import annotations

import csv
import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path

import landmark_map
import mesh_utils

NEUTRAL_SEG_ID = "EXP_neutral_peak"


def _read_frame_list(frame_list_path: Path) -> list[dict[str, str]]:
    with frame_list_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _compute_neutral_frame(capture_id: str, actor_dir: Path) -> dict:
    frame_list_path = actor_dir / "decoder" / "frame_list.csv"
    if not frame_list_path.exists():
        raise FileNotFoundError(f"No frame_list.csv at {frame_list_path}")

    rows = _read_frame_list(frame_list_path)
    candidates = sorted(
        (r["frame_id"].strip().zfill(6) for r in rows if r["seg_id"] == NEUTRAL_SEG_ID),
        key=int,
    )
    if not candidates:
        raise RuntimeError(f"No {NEUTRAL_SEG_ID} rows in {frame_list_path}")

    needed_ids = set(landmark_map.STANDARD_INDEX_TO_KEYPOINT3D_ID.values())
    best_frame_id: str | None = None
    best_coverage = -1
    for frame_id in candidates:  # ascending order -> first-seen wins ties
        present_ids = set(mesh_utils.keypoints_3d_by_id(actor_dir, frame_id).keys())
        coverage = len(needed_ids & present_ids)
        if coverage > best_coverage:
            best_frame_id, best_coverage = frame_id, coverage

    return {
        "capture_id": capture_id,
        "frame_id": best_frame_id,
        "seg_id": NEUTRAL_SEG_ID,
        "coverage": best_coverage,
        "needed_total": len(needed_ids),
        "candidates_considered": candidates,
        "tie_break": "lowest_frame_id",
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }


def _locked_atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            tmp_path = path.with_name(path.name + ".tmp")
            tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp_path.replace(path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def resolve_neutral_frame(capture_id: str, actor_dir: Path, ava256_landmark_root: Path) -> dict:
    """Read-cache-compute-write-once. Returns the full cached payload;
    frame_id = payload['frame_id']."""
    cache_path = Path(ava256_landmark_root) / capture_id / "neutral_frame.json"
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass  # fall through and recompute

    payload = _compute_neutral_frame(capture_id, actor_dir)
    _locked_atomic_write(cache_path, payload)
    return payload
