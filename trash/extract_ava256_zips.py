#!/usr/bin/env python3
"""Unzips one ava-256 actor's head_pose.zip and registration_vertices.zip
in place, since every other script in this directory (render_overlay.py,
render_overlay_mesh.py, extract_obj.py) reads per-frame files directly off
disk rather than out of the zips:

    decoder/head_pose/head_pose.zip                -> decoder/head_pose/<frame>.txt
    decoder/kinematic_tracking/registration_vertices.zip -> decoder/kinematic_tracking/<frame>.ply

Both zips are flat (no nested directory inside them -- confirmed by
inspection), so this is a plain extractall() into the zip's own parent
directory for each.

Usage:
    python3 extract_ava256_zips.py <ACTOR> [--actor-dir ...]
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ACTOR_ROOT = Path("/scratch/thirty/irwinngo/ava-256")
DEFAULT_ACTOR_DIR = ACTOR_ROOT / "20210810--1306--FXN596"


def resolve_actor_dir(actor: str) -> Path:
    """Resolves a 6-character actor code (e.g. 'FXN596') to its full
    directory under ACTOR_ROOT (e.g. .../20210810--1306--FXN596), by
    globbing for that suffix -- the leading date/session segments aren't
    something a caller would otherwise know."""
    matches = sorted(ACTOR_ROOT.glob(f"*--{actor}"))
    if not matches:
        raise FileNotFoundError(f"No actor directory matching '*--{actor}' found under {ACTOR_ROOT}")
    if len(matches) > 1:
        raise ValueError(f"Multiple actor directories matched '*--{actor}' under {ACTOR_ROOT}: {matches}")
    return matches[0]


def extract_zip(zip_path: Path, dest_dir: Path) -> int:
    """Extracts zip_path into dest_dir (zip_path's own parent). Returns the
    number of members extracted. Overwrites any already-extracted files with
    the same (unchanged) content, so this is safe to re-run."""
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} not found")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        zf.extractall(dest_dir)
    return len(members)


def extract_ava256_zips(actor_dir: Path) -> None:
    head_pose_zip = actor_dir / "decoder" / "head_pose" / "head_pose.zip"
    kinematic_tracking_zip = actor_dir / "decoder" / "kinematic_tracking" / "registration_vertices.zip"

    count = extract_zip(head_pose_zip, head_pose_zip.parent)
    print(f"Extracted {count} files from {head_pose_zip} to {head_pose_zip.parent}")

    count = extract_zip(kinematic_tracking_zip, kinematic_tracking_zip.parent)
    print(f"Extracted {count} files from {kinematic_tracking_zip} to {kinematic_tracking_zip.parent}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    actor_group = parser.add_mutually_exclusive_group(required=True)
    actor_group.add_argument("actor", nargs="?", default=None,
                              help="6-character actor code, e.g. FXN596 (resolved against ACTOR_ROOT="
                                   f"{ACTOR_ROOT})")
    actor_group.add_argument("--actor-dir", default=None, help="Full ava-256 actor directory (overrides <ACTOR>)")
    args = parser.parse_args()

    actor_dir = Path(args.actor_dir) if args.actor_dir is not None else resolve_actor_dir(args.actor)
    extract_ava256_zips(actor_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
