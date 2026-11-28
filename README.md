# training-dataloader-prefetch-starvation-planner
规划 DataLoader 预取、慢 shard 和恢复状态

## 说明

这是一个**纯模拟**项目：所有 shard 数据都是调用方传入的样本 id 列表，
**不读取任何真实数据集**、不访问磁盘数据文件、不联网。代码仅使用标准库，
兼容 Python 3.8.2。

## 组成

- `prefetch_scheduler.py` — 离散事件预取调度器。
  - 输入：shard 延迟与批次（`ShardSpec`）、worker 及其失联区间
    （`WorkerSpec`）、prefetch 深度、内存预算（按样本数）、重试上限、
    checkpoint 间隔、epoch 数（`SchedulerConfig`）。
  - 每个训练 step 需要所有 shard 各一个批次（跨 rank 同步屏障），
    慢 shard 会阻塞 step，其他 shard 继续预取，但内存预算与 prefetch
    深度会施加背压，慢 shard 不会导致内存无限增长。
  - worker 失联时，其未确认批次被保留（retained）并在有可用 worker
    后重试，不会丢失。
  - 已确认样本在签发与确认两处去重，重复样本不会进入下一个 epoch。
  - 输出（`run()` 返回的 dict）：`prefetch_order`（预取顺序）、
    `memory_peak`（内存峰值，单位：样本数）、`idle_intervals`
    （训练空闲区间）、`recovery_manifest`（恢复清单）等。
- `state_recorder.py` — 可恢复状态记录器。`journal.jsonl` 记录全部状态
  变迁，`checkpoint.json` 定期快照；`restore()` 通过快照 + 日志重放
  在进程重启后恢复已确认样本与未完成预取（in-flight 批次在重启后
  自动转为 retained 等待重试）；`recovery_manifest()` 输出恢复清单。
- `test_prefetch_scheduler.py` — 单元测试，覆盖：单 shard 慢、
  worker 重启、重复样本、checkpoint 恢复。

## 运行测试

```bash
python -m unittest
```

## 简单示例

```python
from prefetch_scheduler import (
    PrefetchScheduler, SchedulerConfig, ShardSpec, WorkerSpec)
from state_recorder import StateRecorder

shards = [ShardSpec("s0", 0.5, [["a", "b"], ["c", "d"]]),
          ShardSpec("s1", 5.0, [["e", "f"], ["g", "h"]])]  # s1 是慢 shard
workers = [WorkerSpec("w0"), WorkerSpec("w1", down_intervals=[(0.4, 2.0)])]
config = SchedulerConfig(prefetch_depth=4, memory_budget=8,
                         max_retries=2, checkpoint_interval=1)

recorder = StateRecorder("/tmp/prefetch-state")
report = PrefetchScheduler(shards, workers, config, recorder=recorder).run()
print(report["prefetch_order"])    # 预取顺序
print(report["memory_peak"])       # 内存峰值（样本数）
print(report["idle_intervals"])    # 训练空闲区间
print(report["recovery_manifest"]) # 恢复清单
```
