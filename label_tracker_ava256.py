"""Standalone Ava-256 label tracker.

Deliberately NOT a subclass/user of backend.app.services.label_tracker's
LabelTracker -- that class is bound to the backend's global, env-configured
Settings singleton, whereas this pipeline's label_tracker_path is a plain
CLI flag that may point anywhere and may not exist yet. Shares the exact
same physical label_tracker.json file and the same fcntl-lock +
temp-file-replace write pattern (see backend/app/services/label_tracker.py's
LabelTracker._write() docstring for why the lock is a separate sibling file,
never the data file itself), just under a new top-level "Ava256" key.

Key shape is FLAT: {"Ava256": {"<capture_id>": "<status>"}} -- one level,
not FaceScape/Nersemble's {actor: {sequence: status}} two-level shape.
Ava-256's unit of work is one capture directory; there's no second
(expression/sequence) axis to key on, so a redundant nested key would only
exist to match a shape convention with no real second dimension behind it.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
from typing import Literal

AvaLabelStatus = Literal["unlabeled", "unreviewed", "unconfirmed", "confirmed"]

_CONFIRM_LABELED_FROM = "unlabeled"
_MARK_UNCONFIRMED_FROM = "unreviewed"
_CONFIRM_FROM = "unconfirmed"
_UNCONFIRM_LABELED_FROM = "unreviewed"
_UNCONFIRM_FROM = "unconfirmed"
_UNCONFIRM_CONFIRMED_FROM = "confirmed"


class Ava256LabelTracker:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _load(self) -> dict[str, dict[str, str]]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def get_all(self) -> dict[str, str]:
        return self._load().get("Ava256", {})

    def get_status(self, capture_id: str, *, all_statuses: dict[str, str] | None = None) -> AvaLabelStatus:
        data = all_statuses if all_statuses is not None else self.get_all()
        return data.get(capture_id, "unlabeled")  # implicit default -- never itself written

    def _write(self, capture_id: str, status: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                data = self._load()
                data.setdefault("Ava256", {})[capture_id] = status
                tmp_path = self.path.with_name(self.path.name + ".tmp")
                tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
                tmp_path.replace(self.path)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def confirm_labeled(self, capture_id: str) -> AvaLabelStatus | None:
        """unlabeled -> unreviewed. Called directly by
        run_neutral_skin_propagation.py on full success -- there is no prior
        explicit "unlabeled" write (unlike FaceScape's
        process_expression_sapiens.py), since the implicit default already
        satisfies this guard. Returns None (no-op, does not raise) if the
        current status isn't "unlabeled" -- a rerun on an
        already-unreviewed/unconfirmed/confirmed capture is left alone."""
        if self.get_status(capture_id) != _CONFIRM_LABELED_FROM:
            return None
        self._write(capture_id, "unreviewed")
        return "unreviewed"

    def mark_unconfirmed(self, capture_id: str) -> AvaLabelStatus | None:
        """unreviewed -> unconfirmed. Called by run_pipeline.py's
        standalone/batch (include_skin=True) invocation once the
        skin-augmented final wrap succeeds."""
        if self.get_status(capture_id) != _MARK_UNCONFIRMED_FROM:
            return None
        self._write(capture_id, "unconfirmed")
        return "unconfirmed"

    def confirm(self, capture_id: str) -> AvaLabelStatus | None:
        """unconfirmed -> confirmed. Not exercised by the batch scripts --
        mirrored 1:1 from the real LabelTracker for a future Ava-256 review UI."""
        if self.get_status(capture_id) != _CONFIRM_FROM:
            return None
        self._write(capture_id, "confirmed")
        return "confirmed"

    def unconfirm_labeled(self, capture_id: str) -> AvaLabelStatus | None:
        if self.get_status(capture_id) != _UNCONFIRM_LABELED_FROM:
            return None
        self._write(capture_id, "unlabeled")
        return "unlabeled"

    def unconfirm(self, capture_id: str) -> AvaLabelStatus | None:
        if self.get_status(capture_id) != _UNCONFIRM_FROM:
            return None
        self._write(capture_id, "unreviewed")
        return "unreviewed"

    def unconfirm_confirmed(self, capture_id: str) -> AvaLabelStatus | None:
        if self.get_status(capture_id) != _UNCONFIRM_CONFIRMED_FROM:
            return None
        self._write(capture_id, "unconfirmed")
        return "unconfirmed"


def ensure_label_tracker_file(label_tracker_path: Path) -> None:
    """Auto-creates an empty label_tracker.json if missing, same
    lock-and-atomic-replace pattern as every other write here, so two
    concurrent first-ever invocations can't race a torn file into existence."""
    path = Path(label_tracker_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if not path.exists():
                tmp_path = path.with_name(path.name + ".tmp")
                tmp_path.write_text("{}", encoding="utf-8")
                tmp_path.replace(path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
