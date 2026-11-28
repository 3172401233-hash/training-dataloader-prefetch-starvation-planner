"""Tests for the prefetch scheduler and the recoverable state recorder.

Run with:  python -m unittest
"""

import json
import os
import shutil
import tempfile
import unittest

from prefetch_scheduler import (
    PrefetchScheduler,
    SchedulerConfig,
    ShardSpec,
    WorkerSpec,
)
from state_recorder import StateRecorder


def make_shards(latencies, num_batches, batch_size, prefix="s"):
    shards = []
    for idx, latency in enumerate(latencies):
        batches = [
            ["%s%d_b%d_%d" % (prefix, idx, b, k) for k in range(batch_size)]
            for b in range(num_batches)
        ]
        shards.append(ShardSpec("shard%d" % idx, latency, batches))
    return shards


def all_sample_ids(shards):
    return sorted({s for shard in shards for b in shard.batches for s in b})


class SlowShardTest(unittest.TestCase):
    def test_single_slow_shard_blocks_barrier_but_memory_is_bounded(self):
        shards = make_shards([0.5, 0.5, 5.0], num_batches=4, batch_size=4)
        workers = [WorkerSpec("w0"), WorkerSpec("w1"), WorkerSpec("w2")]
        config = SchedulerConfig(prefetch_depth=6, memory_budget=24,
                                 checkpoint_interval=2)
        report = PrefetchScheduler(shards, workers, config).run()

        self.assertFalse(report["stalled"])
        # Slow shard stalls the cross-rank barrier -> trainer idles.
        self.assertTrue(report["idle_intervals"])
        for start, end in report["idle_intervals"]:
            self.assertLess(start, end)
        # Fast shards keep prefetching but never exceed the memory budget.
        self.assertLessEqual(report["memory_peak"], 24)
        self.assertGreater(report["memory_peak"], 4 * 3)  # > one step
        # Everything is eventually confirmed exactly once.
        self.assertEqual(report["confirmed_samples"], all_sample_ids(shards))
        self.assertEqual(report["confirmed_steps"], 4)
        # Prefetch order interleaves shards instead of racing one ahead.
        first_three = report["prefetch_order"][:3]
        self.assertEqual({p["shard"] for p in first_three},
                         {"shard0", "shard1", "shard2"})
        self.assertEqual(report["recovery_manifest"]["pending_batches"], [])


class WorkerRestartTest(unittest.TestCase):
    def test_lost_worker_keeps_unconfirmed_batches_for_retry(self):
        shards = make_shards([1.0, 1.0], num_batches=4, batch_size=2)
        # w0 is lost from t=0.5 to t=3.0, mid-fetch.
        workers = [WorkerSpec("w0", down_intervals=[(0.5, 3.0)]),
                   WorkerSpec("w1")]
        config = SchedulerConfig(prefetch_depth=4, memory_budget=16,
                                 max_retries=2, checkpoint_interval=100)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = StateRecorder(tmp)
            report = PrefetchScheduler(shards, workers, config,
                                       recorder=recorder).run()

            self.assertFalse(report["stalled"])
            self.assertEqual(report["confirmed_samples"],
                             all_sample_ids(shards))
            # The cancelled batch was retried, not dropped.
            retries = [p for p in report["prefetch_order"] if p["retry"]]
            self.assertTrue(retries)
            # Journal records the cancellation of the unconfirmed batch.
            with open(recorder.journal_path) as fh:
                events = [json.loads(line) for line in fh if line.strip()]
            cancelled = [e for e in events if e["type"] == "cancel"]
            self.assertTrue(cancelled)
            self.assertEqual(cancelled[0]["payload"]["worker"], "w0")
            # Nothing left unfinished at the end.
            self.assertEqual(report["recovery_manifest"]["pending_batches"],
                             [])


class DuplicateSampleTest(unittest.TestCase):
    def test_duplicate_samples_never_reenter_next_epoch(self):
        batches_a = [["a", "b"], ["a", "c"], ["d", "e"]]
        batches_b = [["x", "y"], ["x", "z"], ["w", "a"]]
        shards = [ShardSpec("sa", 1.0, batches_a),
                  ShardSpec("sb", 1.0, batches_b)]
        workers = [WorkerSpec("w0"), WorkerSpec("w1")]
        config = SchedulerConfig(prefetch_depth=4, memory_budget=8,
                                 checkpoint_interval=2, epochs=2)
        report = PrefetchScheduler(shards, workers, config).run()

        self.assertFalse(report["stalled"])
        # Each unique sample is confirmed exactly once across both epochs.
        self.assertEqual(report["confirmed_samples"],
                         sorted({"a", "b", "c", "d", "e",
                                 "w", "x", "y", "z"}))
        # Duplicates were filtered instead of re-prefetched in epoch 2.
        self.assertGreater(report["duplicates_skipped"], 0)
        # No duplicate sample ids were ever prefetched again in epoch 2.
        epoch_two = [p for p in report["prefetch_order"] if p["epoch"] == 1]
        self.assertTrue(epoch_two)
        epoch_one_samples = {s for p in report["prefetch_order"]
                             if p["epoch"] == 0 for s in p["samples"]}
        for entry in epoch_two:
            self.assertTrue(set(entry["samples"]).isdisjoint(
                epoch_one_samples))


class CheckpointRecoveryTest(unittest.TestCase):
    def test_restart_resumes_confirmed_and_unfinished_work(self):
        shards = make_shards([1.0, 2.0], num_batches=6, batch_size=2)
        workers = [WorkerSpec("w0"), WorkerSpec("w1")]
        config = SchedulerConfig(prefetch_depth=4, memory_budget=12,
                                 checkpoint_interval=2)
        expected = all_sample_ids(shards)

        with tempfile.TemporaryDirectory() as tmp:
            recorder = StateRecorder(tmp)
            # First process "crashes" after 2 confirmed steps.
            first = PrefetchScheduler(shards, workers, config,
                                      recorder=recorder)
            report1 = first.run(max_confirm_steps=2)
            self.assertEqual(report1["confirmed_steps"], 2)

            manifest = recorder.recovery_manifest()
            self.assertEqual(manifest["confirmed_samples"],
                             report1["confirmed_samples"])
            # In-flight / buffered work survives the crash as pending.
            self.assertTrue(manifest["pending_batches"])

            # New process restores from disk and finishes the epoch.
            restored = recorder.restore()
            self.assertTrue(restored["in_flight"] or restored["buffered"]
                            or restored["retained"])
            second = PrefetchScheduler(shards, workers, config,
                                       recorder=recorder,
                                       restored_state=restored)
            # In-flight batches of the dead process became retained.
            self.assertFalse(second.in_flight)
            report2 = second.run()

            self.assertFalse(report2["stalled"])
            combined = set(report1["confirmed_samples"])
            combined.update(report2["confirmed_samples"])
            self.assertEqual(sorted(combined), expected)
            # No sample was confirmed twice across the restart.
            self.assertEqual(len(combined), len(expected))
            self.assertEqual(
                report2["recovery_manifest"]["confirmed_samples"], expected)
            self.assertEqual(
                report2["recovery_manifest"]["pending_batches"], [])

    def test_checkpoint_file_is_written_and_journal_replayed(self):
        shards = make_shards([1.0], num_batches=4, batch_size=2)
        workers = [WorkerSpec("w0")]
        config = SchedulerConfig(prefetch_depth=2, memory_budget=4,
                                 checkpoint_interval=2)
        with tempfile.TemporaryDirectory() as tmp:
            recorder = StateRecorder(tmp)
            PrefetchScheduler(shards, workers, config,
                              recorder=recorder).run()
            self.assertTrue(os.path.exists(recorder.checkpoint_path))
            with open(recorder.checkpoint_path) as fh:
                snapshot = json.load(fh)
            self.assertEqual(snapshot["confirmed_steps"], 4)
            self.assertEqual(sorted(snapshot["confirmed_samples"]),
                             all_sample_ids(shards))


if __name__ == "__main__":
    unittest.main()
