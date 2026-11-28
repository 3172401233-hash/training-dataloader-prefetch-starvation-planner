"""Recoverable state recorder for the prefetch scheduler.

Persists confirmed samples and unfinished prefetch work to disk so that
a process restart can resume without re-reading confirmed samples and
without losing unconfirmed batches.

Storage layout inside the recorder directory:

- ``checkpoint.json``: periodic full snapshot of the scheduler state.
- ``journal.jsonl``: append-only event log for everything that happens
  after the last snapshot.  Truncated every time a snapshot is written.

``restore()`` rebuilds the state by loading the latest snapshot (if any)
and replaying the journal on top of it.
"""

import json
import os

JOURNAL_FILE = "journal.jsonl"
CHECKPOINT_FILE = "checkpoint.json"


def _empty_state():
    return {
        "confirmed_samples": [],
        "cursors": {},
        "epoch": 0,
        "step_index": 0,
        "confirmed_steps": 0,
        "in_flight": {},
        "buffered": {},
        "retained": {},
        "retries": {},
    }


def _apply_event(state, entry):
    etype = entry["type"]
    payload = entry["payload"]
    if etype == "issue":
        info = payload["info"]
        state["in_flight"][payload["batch"]] = info
        state["cursors"][info["shard"]] = payload["cursor"]
        state["retained"].pop(payload["batch"], None)
    elif etype == "ready":
        info = state["in_flight"].pop(payload["batch"], None)
        if info is not None:
            state["buffered"][payload["batch"]] = info
    elif etype == "cancel":
        info = state["in_flight"].pop(payload["batch"], None)
        if info is not None:
            state["retained"][payload["batch"]] = info
        state["retries"][payload["batch"]] = payload["retries"]
    elif etype == "confirm_step":
        confirmed = set(state["confirmed_samples"])
        for batch_key in payload["batches"]:
            info = state["buffered"].pop(batch_key, None)
            if info is not None:
                confirmed.update(info["samples"])
        state["confirmed_samples"] = sorted(confirmed)
        state["step_index"] = payload["step_index"]
        state["confirmed_steps"] = payload["confirmed_steps"]
    elif etype == "epoch_start":
        state["epoch"] = payload["epoch"]
        state["step_index"] = 0
        state["cursors"] = {}
    return state


class StateRecorder:
    """Append-only journal plus periodic snapshots on disk."""

    def __init__(self, directory):
        self.directory = directory
        if not os.path.isdir(directory):
            os.makedirs(directory)
        self.journal_path = os.path.join(directory, JOURNAL_FILE)
        self.checkpoint_path = os.path.join(directory, CHECKPOINT_FILE)

    def record(self, event_type, payload):
        """Append one event to the journal."""
        entry = {"type": event_type, "payload": payload}
        with open(self.journal_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")

    def snapshot(self, state):
        """Atomically write a full checkpoint and truncate the journal."""
        tmp_path = self.checkpoint_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, sort_keys=True)
        os.replace(tmp_path, self.checkpoint_path)
        with open(self.journal_path, "w", encoding="utf-8"):
            pass

    def restore(self):
        """Rebuild scheduler state from checkpoint + journal replay."""
        state = _empty_state()
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "r", encoding="utf-8") as fh:
                state = json.load(fh)
        for key, default in _empty_state().items():
            state.setdefault(key, default)
        if os.path.exists(self.journal_path):
            with open(self.journal_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        _apply_event(state, json.loads(line))
        return state

    def recovery_manifest(self):
        """Summary of what a restart would resume from.

        Returns the already-confirmed samples and the batch keys that
        were still unfinished (in-flight, buffered or retained) when the
        state was last persisted.
        """
        state = self.restore()
        pending = sorted(
            list(state["in_flight"]) + list(state["buffered"]) + list(state["retained"])
        )
        return {
            "confirmed_samples": list(state["confirmed_samples"]),
            "pending_batches": pending,
        }
