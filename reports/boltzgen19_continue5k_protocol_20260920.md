# 19 个 BoltzGen 片段：1,000 → 5,000 步续训协议

目的：在固定数据、架构、损失和采样配置下，检查额外训练是否改善随机几何去噪与
自由生成。该实验是探索性训练，不是正式 Joint-v2 训练或新结构泛化评估。

- 保留原 19 个样本、原顺序，full batch=19；源数据身份不变。
- 从 `boltzgen19_cpu_overfit1000_20260920/checkpoint_00001000.pt` 恢复权重、Adam 状态、
  独立噪声 RNG 和 torch RNG。父 checkpoint SHA256：
  `47d7d04f4914cb39cc04b48047af1dac729809d864e02a905e30b38a1ef607f0`。
- 124,976 参数 tiny 模型、FP32、CPU 4 线程、dropout=0，几何与序列三项等权损失不变。
- 原 cosine 日程已在 1,000 步结束；新增 4,000 步保持终点 LR=1e-4，无 warmup 重启，
  AdamW weight decay=0、gradient clip=10。
- 总步数固定为 5,000，每 500 步保存 checkpoint 和固定去噪面板。
- 在 3,000、5,000 步进行四个独立噪声基底的去噪面板及 38 个自由生成候选评估。
  种子与父运行一致，以便配对比较；评估随机流不影响训练随机流。
- 固定面板：每样本一个基底、10 个分层时间，另计 t=0、0.99 边界；新面板四个基底。
- rollout 20 步、两个基底，不做几何修复或共价链重建。位置/旋转和骨架代理必须单独
  报告，不用序列恢复率代表结构成功。
- 最终结论使用固定 5,000 步模型；中途面板不参与选择训练预算或调整超参数。

运行前核对四张 H100 均已被其他任务占满，因此继续使用 CPU。
恢复测试中 228 条面板记录与原 1,000 步模型逐值一致，最大差为 0；1,001 步 smoke 已完成
实际优化、保存、重载和自由采样。随机更新回放及恢复相关 5 项测试通过。

入口：`scripts/experiments/continue_joint_v2_boltzgen_overfit.py`。
运行目录：`artifacts/experiments/boltzgen19_cpu_continue5000_20260920/`。
日志：`artifacts/tmp/boltzgen19_cpu_continue5000_20260920.log`。

```bash
source scripts/project_tmp_env.sh
python scripts/experiments/continue_joint_v2_boltzgen_overfit.py \
  --fit-run artifacts/experiments/boltzgen19_cpu_overfit1000_20260920 \
  --output artifacts/experiments/boltzgen19_cpu_continue5000_20260920 \
  --total-steps 5000 --eval-every 500 --milestones 3000 5000
```
