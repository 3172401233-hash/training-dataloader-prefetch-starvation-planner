"""Discrete-event prefetch scheduler for a sharded training DataLoader.

This is a simulation: shard contents are plain lists of sample ids
supplied by the caller, no real dataset is ever read.

Model
-----
- Each shard owns an ordered list of batches and a constant read
  latency.  A training step needs one batch from *every* shard
  (cross-rank barrier), so one slow shard stalls the whole step while
  the other shards keep prefetching.
- Workers fetch one batch at a time.  A worker can be lost for a
  configured down interval; its unconfirmed in-flight batches are
  retained (never dropped) and retried once a worker is available.
- Memory is bounded: in-flight plus buffered samples never exceed the
  memory budget, so fast shards cannot grow memory without limit while
  a slow shard blocks the barrier.  The prefetch depth additionally
  caps the number of outstanding batches.
- Confirmed (consumed) samples are remembered; duplicates are filtered
  at issue time and at confirm time, so a repeated sample never enters
  the next epoch.
- Every ``checkpoint_interval`` confirmed steps the scheduler writes a
  snapshot through a ``StateRecorder``; all other transitions are
  journaled, so a restarted process can resume exactly.
"""

import heapq
import itertools


class ShardSpec:
    """A data shard: read latency plus its batches of sample ids."""

    def __init__(self, shard_id, latency, batches):
        self.shard_id = shard_id
        self.latency = float(latency)
        self.batches = [list(batch) for batch in batches]


class WorkerSpec:
    """A prefetch worker, optionally lost during ``down_intervals``."""

    def __init__(self, worker_id, down_intervals=()):
        self.worker_id = worker_id
        self.down_intervals = [tuple(iv) for iv in down_intervals]


class SchedulerConfig:
    def __init__(self, prefetch_depth=4, memory_budget=64, max_retries=2,
                 checkpoint_interval=4, step_compute_time=1.0, epochs=1):
        self.prefetch_depth = prefetch_depth
        self.memory_budget = memory_budget  # in samples
        self.max_retries = max_retries
        self.checkpoint_interval = checkpoint_interval
        self.step_compute_time = float(step_compute_time)
        self.epochs = epochs


def _batch_key(shard_id, epoch, index):
    return "%s:%d:%d" % (shard_id, epoch, index)


class PrefetchScheduler:
    def __init__(self, shards, workers, config, recorder=None,
                 restored_state=None):
        if not shards:
            raise ValueError("at least one shard is required")
        if not workers:
            raise ValueError("at least one worker is required")
        self.shards = list(shards)
        self.shard_map = {s.shard_id: s for s in self.shards}
        self.workers = list(workers)
        self.config = config
        self.recorder = recorder
        self.steps_per_epoch = min(len(s.batches) for s in self.shards)

        # Recoverable state (mirrored by the recorder).
        self.confirmed = set()
        self.cursors = {s.shard_id: 0 for s in self.shards}
        self.epoch = 0
        self.step_index = 0
        self.confirmed_steps = 0
        self.in_flight = {}
        self.buffered = {}
        self.retained = {}
        self.retries = {}
        if restored_state is not None:
            self._load_state(restored_state)

        # Transient simulation state.
        self.now = 0.0
        self.trainer_free = 0.0
        self.idle_since = None
        self.idle_intervals = []
        self.prefetch_order = []
        self.duplicates_skipped = 0
        self.stalled = False
        self.outstanding_samples = sum(
            len(i["samples"]) for i in self.in_flight.values()
        ) + sum(len(i["samples"]) for i in self.buffered.values())
        self.memory_peak = self.outstanding_samples
        self.worker_status = {w.worker_id: "free" for w in self.workers}
        self._tokens = itertools.count(1)
        self._seq = itertools.count()
        self._events = []
        for worker in self.workers:
            for start, end in worker.down_intervals:
                self._push(start, "worker_down", worker.worker_id)
                self._push(end, "worker_up", worker.worker_id)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def run(self, max_confirm_steps=None):
        """Run the simulation; stop early after ``max_confirm_steps``
        confirmed steps to emulate a crash (used for recovery tests)."""
        self._try_dispatch()
        self._try_train()
        self._try_dispatch()
        while not self._done():
            if not self._events:
                self.stalled = True
                break
            self.now, _, kind, payload = heapq.heappop(self._events)
            self._apply_event(kind, payload)
            self._try_dispatch()
            self._try_train()
            # Training may have freed memory or rolled the epoch over,
            # which unblocks issuing without any further event.
            self._try_dispatch()
            if (max_confirm_steps is not None
                    and self.confirmed_steps >= max_confirm_steps):
                break
        return self._report()

    # ------------------------------------------------------------------
    # event handling
    # ------------------------------------------------------------------
    def _push(self, time, kind, payload):
        heapq.heappush(self._events, (time, next(self._seq), kind, payload))

    def _apply_event(self, kind, payload):
        if kind == "fetch_done":
            batch_key, token = payload
            info = self.in_flight.get(batch_key)
            if info is None or info["token"] != token:
                return  # cancelled while the worker was lost
            del self.in_flight[batch_key]
            self.buffered[batch_key] = info
            self.worker_status[info["worker"]] = "free"
            self._record("ready", {"batch": batch_key})
        elif kind == "worker_down":
            self.worker_status[payload] = "down"
            for batch_key, info in list(self.in_flight.items()):
                if info["worker"] == payload:
                    # Worker lost: keep the unconfirmed batch for retry.
                    del self.in_flight[batch_key]
                    self.outstanding_samples -= len(info["samples"])
                    self.retained[batch_key] = info
                    self.retries[batch_key] = self.retries.get(batch_key, 0) + 1
                    self._record("cancel", {
                        "batch": batch_key,
                        "worker": payload,
                        "retries": self.retries[batch_key],
                    })
        elif kind == "worker_up":
            self.worker_status[payload] = "free"

    # ------------------------------------------------------------------
    # prefetch dispatch
    # ------------------------------------------------------------------
    def _try_dispatch(self):
        while True:
            outstanding = len(self.in_flight) + len(self.buffered)
            if outstanding >= self.config.prefetch_depth:
                return
            worker_id = self._free_worker()
            if worker_id is None:
                return
            candidate = self._peek_candidate()
            if candidate is None:
                return
            kind, batch_key, info = candidate
            size = len(info["samples"])
            over_budget = (self.outstanding_samples + size
                           > self.config.memory_budget)
            if over_budget and self.outstanding_samples > 0:
                return  # backpressure: slow shard may not hog memory
            if kind == "retry":
                del self.retained[batch_key]
            info = dict(info)
            info["worker"] = worker_id
            info["token"] = next(self._tokens)
            self.in_flight[batch_key] = info
            self.worker_status[worker_id] = "busy"
            self.outstanding_samples += size
            self.memory_peak = max(self.memory_peak, self.outstanding_samples)
            latency = self.shard_map[info["shard"]].latency
            self._push(self.now + latency, "fetch_done",
                       (batch_key, info["token"]))
            self.prefetch_order.append({
                "time": self.now,
                "worker": worker_id,
                "shard": info["shard"],
                "epoch": info["epoch"],
                "index": info["index"],
                "samples": list(info["samples"]),
                "retry": kind == "retry",
            })
            self._record("issue", {
                "batch": batch_key,
                "info": info,
                "cursor": self.cursors[info["shard"]],
                "retry": kind == "retry",
            })

    def _free_worker(self):
        for worker in self.workers:
            if self.worker_status[worker.worker_id] == "free":
                return worker.worker_id
        return None

    def _peek_candidate(self):
        # Retained (unconfirmed) batches from lost workers come first.
        for batch_key in sorted(self.retained):
            if self.retries.get(batch_key, 0) <= self.config.max_retries:
                return ("retry", batch_key, self.retained[batch_key])
        if self.epoch >= self.config.epochs:
            return None
        # Then the shard that is least far ahead, so cross-rank steps
        # complete together instead of one shard racing ahead.
        while True:
            chosen = None
            for shard in self.shards:
                if self.cursors[shard.shard_id] >= self.steps_per_epoch:
                    continue
                if (chosen is None or self.cursors[shard.shard_id]
                        < self.cursors[chosen.shard_id]):
                    chosen = shard
            if chosen is None:
                return None
            index = self.cursors[chosen.shard_id]
            raw = chosen.batches[index]
            samples = [s for s in raw if s not in self.confirmed]
            self.cursors[chosen.shard_id] = index + 1
            self.duplicates_skipped += len(raw) - len(samples)
            batch_key = _batch_key(chosen.shard_id, self.epoch, index)
            info = {
                "batch": batch_key,
                "shard": chosen.shard_id,
                "epoch": self.epoch,
                "index": index,
                "samples": samples,
                "worker": None,
                "token": None,
            }
            return ("fresh", batch_key, info)

    # ------------------------------------------------------------------
    # training consumption (cross-rank barrier)
    # ------------------------------------------------------------------
    def _try_train(self):
        while self.epoch < self.config.epochs:
            if self.step_index >= self.steps_per_epoch:
                self.epoch += 1
                self.step_index = 0
                self.cursors = {s.shard_id: 0 for s in self.shards}
                self._record("epoch_start", {"epoch": self.epoch})
                continue
            keys = [_batch_key(s.shard_id, self.epoch, self.step_index)
                    for s in self.shards]
            if not all(key in self.buffered for key in keys):
                if self.idle_since is None:
                    self.idle_since = max(self.trainer_free, self.now)
                return
            start = max(self.trainer_free, self.now)
            if self.idle_since is not None and start > self.idle_since:
                self._add_idle(self.idle_since, start)
            self.idle_since = None
            for key in keys:
                info = self.buffered.pop(key)
                self.outstanding_samples -= len(info["samples"])
                self.confirmed.update(
                    s for s in info["samples"] if s not in self.confirmed)
            self.step_index += 1
            self.confirmed_steps += 1
            self.trainer_free = start + self.config.step_compute_time
            self._record("confirm_step", {
                "batches": keys,
                "step_index": self.step_index,
                "confirmed_steps": self.confirmed_steps,
            })
            if (self.recorder is not None
                    and self.confirmed_steps % self.config.checkpoint_interval
                    == 0):
                self.recorder.snapshot(self._export_state())

    def _add_idle(self, start, end):
        if self.idle_intervals and self.idle_intervals[-1][1] >= start:
            last = self.idle_intervals[-1]
            self.idle_intervals[-1] = (last[0], max(last[1], end))
        else:
            self.idle_intervals.append((start, end))

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def _record(self, event_type, payload):
        if self.recorder is not None:
            self.recorder.record(event_type, payload)

    def _export_state(self):
        return {
            "confirmed_samples": sorted(self.confirmed),
            "cursors": dict(self.cursors),
            "epoch": self.epoch,
            "step_index": self.step_index,
            "confirmed_steps": self.confirmed_steps,
            "in_flight": dict(self.in_flight),
            "buffered": dict(self.buffered),
            "retained": dict(self.retained),
            "retries": dict(self.retries),
        }

    def _load_state(self, state):
        self.confirmed = set(state["confirmed_samples"])
        self.cursors = {s.shard_id: 0 for s in self.shards}
        self.cursors.update(state["cursors"])
        self.epoch = state["epoch"]
        self.step_index = state["step_index"]
        self.confirmed_steps = state.get("confirmed_steps", 0)
        self.buffered = dict(state["buffered"])
        self.retained = dict(state["retained"])
        # After a restart every previously in-flight batch belongs to a
        # dead worker: keep it as retained so it is retried, not lost.
        self.retained.update(state["in_flight"])
        self.in_flight = {}
        self.retries = dict(state["retries"])

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def _done(self):
        return (self.epoch >= self.config.epochs
                and not self.in_flight
                and not self.buffered
                and not self.retained)

    def _report(self):
        if self.recorder is not None:
            manifest = self.recorder.recovery_manifest()
        else:
            state = self._export_state()
            manifest = {
                "confirmed_samples": sorted(self.confirmed),
                "pending_batches": sorted(
                    list(state["in_flight"]) + list(state["buffered"])
                    + list(state["retained"])),
            }
        return {
            "prefetch_order": list(self.prefetch_order),
            "memory_peak": self.memory_peak,
            "idle_intervals": list(self.idle_intervals),
            "confirmed_samples": sorted(self.confirmed),
            "confirmed_steps": self.confirmed_steps,
            "duplicates_skipped": self.duplicates_skipped,
            "stalled": self.stalled,
            "recovery_manifest": manifest,
        }
