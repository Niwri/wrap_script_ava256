"""Landmark-index bookkeeping shared across the Ava-256 pipeline.

STANDARD_INDEX_TO_KEYPOINT3D_ID bridges FaceScape's "standard" 0-73 landmark
index space (the same space neutral_correspondence_without_eyes_POT_all.json's
first 74 rows use) to Ava-256's own decoder/keypoints_3d/<frame>.npy keypoint
IDs (0-273). Derived from SAPIENS_LANDMARK_MAP in
/scratch/ondemand32/irwinngo/FaceView/scripts/preprocess/process_expression_sapiens.py:60-98
(FaceScape index -> sapiens2 keypoints308 index), composed with a
user-confirmed offset from sapiens2 index space to keypoints_3d index space:
    keypoints_3d_id = sapiens2_id - 70   for sapiens2_id in [70, 220)
    keypoints_3d_id = sapiens2_id - 34   for sapiens2_id >= 220
(the >=220 branch went through -35, then -36, then settled on -34 -- this
only affects the 4 ear indices below, the only entries with a sapiens2_id
>=220: 1 (206->207), 4 (195->196), 6 (232->233), 14 (221->222), tracking the
-35/-36/-34 revisions in that order).
Indices 2, 8, 9, 29 have no sapiens2 source at all (None in
SAPIENS_LANDMARK_MAP) and therefore no keypoints_3d source either -- absent
from this table. Verified collision-free across all 45 resolved entries.

Indices 2 and 9 (ear-top points) are additionally excluded from the wrap
correspondence itself even though a hypothetical sapiens2 source existed for
neither -- confirmed with the user as always-excluded, same as FaceScape's own
"index 2/9 have no automatic source" special case (see
run_neutral_skin_propagation.py's own docstring on FACESCAPE_STANDARD_INDEX_REGION_GROUPS).
"""
from __future__ import annotations

STANDARD_INDEX_TO_KEYPOINT3D_ID: dict[int, int] = {
    1: 207, 4: 196, 6: 233, 11: 88, 13: 86, 14: 222, 15: 38, 16: 35, 17: 40,
    18: 119, 19: 116, 20: 128, 21: 144, 23: 123, 25: 125, 26: 148, 27: 132,
    30: 23, 32: 105, 33: 103, 34: 62, 35: 59, 36: 64, 37: 118, 38: 113,
    39: 127, 40: 143, 42: 122, 44: 124, 45: 147, 46: 131, 47: 13, 50: 7,
    54: 5, 55: 121, 56: 137, 57: 2, 58: 136, 59: 120, 60: 1, 61: 109,
    65: 50, 69: 8, 71: 26, 72: 17,
}

# Standard indices with no keypoints_3d source at all (no sapiens2 source in
# the first place) -- always dropped from the correspondence.
NO_SAPIENS_SOURCE_INDICES: frozenset[int] = frozenset({2, 8, 9, 29})

# Ear-top indices confirmed always excluded (redundant with
# NO_SAPIENS_SOURCE_INDICES for 2/9, listed explicitly for clarity/searchability).
ALWAYS_EXCLUDED_INDICES: frozenset[int] = frozenset({2, 9})


def used_landmark_indices(num_pot_rows: int) -> list[int]:
    """The actual "whitelist" of landmark indices that have a real source --
    STANDARD_INDEX_TO_KEYPOINT3D_ID's 45 keys (NOT "every standard index
    except 2/9": the other ~27 standard indices were never in
    SAPIENS_LANDMARK_MAP at all, so their position on any wrap is just
    wherever ICP/blend-wrap interpolation happened to put that vertex, not a
    real landmark -- see run_neutral_skin_propagation.py's own landmark_
    indices computation, which this mirrors) plus every skin index (74+,
    FLAME-vertex-anchored, same as FaceScape). Same role as Nersemble's own
    wrap_script/render_landmarks.py's whitelist.json filter -- "only render
    indices that are actually used," not every index that merely isn't
    explicitly excluded."""
    return sorted(
        set(STANDARD_INDEX_TO_KEYPOINT3D_ID.keys()) | set(range(NUM_STANDARD_LANDMARKS, num_pot_rows))
    )

NUM_STANDARD_LANDMARKS = 74  # matches wrap_script/skin_landmarks.py's own constant

# Naming-only groups for output file organization (AVA_256_LANDMARK_ROOT/
# <CAPTURE>/{GROUP}_<FRAME>.json) -- camera selection itself is fully
# per-landmark (see camera_selection.py), not driven by these buckets.
# Reuses FaceScape's FACESCAPE_STANDARD_INDEX_REGION_GROUPS index sets
# (run_neutral_skin_propagation.py:152-186) plus an "eyelid_lip" bucket for
# every other index STANDARD_INDEX_TO_KEYPOINT3D_ID resolves that FaceScape
# never gave its own dedicated group file.
AVA256_STANDARD_INDEX_REGION_GROUPS: dict[str, list[int]] = {
    "ear": [1, 2, 4, 6, 9, 14],
    "nosebridge": [54, 57, 60],
    "undernose": [19, 38, 61],
    "eyebrow": [30, 47, 69, 72],
    "chin": [50],
    "eyelid_lip": [
        11, 13, 15, 16, 17, 18, 20, 21, 23, 25, 26, 27, 32, 33, 34, 35, 36,
        37, 39, 40, 42, 44, 45, 46, 55, 56, 58, 59, 65, 71,
    ],
}


def group_for_standard_index(index: int) -> str | None:
    """Which AVA256_STANDARD_INDEX_REGION_GROUPS bucket a standard (0-73)
    landmark index belongs to, or None if it's not in any (e.g. 2, 8, 9, 29,
    or any other index outside SAPIENS_LANDMARK_MAP entirely)."""
    for group, indices in AVA256_STANDARD_INDEX_REGION_GROUPS.items():
        if index in indices:
            return group
    return None
