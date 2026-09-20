# 官方 BoltzGen 19 个界面片段：过拟合测试协议

本轮检验已经打通的数据接口是否支持实际优化，以及当前 simplex 模型能否记住已见条件下
的原生序列和结构。不是新结构泛化验证，也不预设固定步数后一定充分过拟合。

数据：`artifacts/datasets/boltzgen_native_four_20260920/dataset/`，split=`smoke`。
四个 PDB 来源 `5hga, 5k9s, 4ep3, 7yxn` 共 19 个连续界面片段，长度 4–9，合计 106 个
target 残基；每段独立样本，不拼接。全部原生 frames 的既有骨架几何代理和已表示原子
碰撞检查均通过（19/19），这不补足此前记录的完整源环境质控限制。

当前四张 H100 均被其他任务占满，采用 CPU。实测 full batch=19 时，1/2/4 线程约为
2.28/1.26/1.01 秒每步，因此选 4 线程。资源和速度记录在 `artifacts/tmp/boltzgen*`。

配置固定为已经用于接口验证的
`configs/joint_v2/experiments/sequence_structure_tiny_encoder_bottleneck_v1.yaml`，
即 tiny 配置，**不是 base 配置的完整规模过拟合结果**。随机初始化，不读取旧训练模型；
dropout=0，旋转在 decoder 各块间完整反传。实际运行遵循 simplex v3 契约：
Dirichlet alpha(t)=1+7t，t∼Uniform[0,1)，三项原始损失为平移终点 MSE、旋转切空间误差、
序列 CE，权重 1:1:1，无额外键长/碰撞损失或共价结构重建。

预算固定 1,000 个 AdamW 更新，每步使用所有 19 个样本，每个样本暴露 1,000 次；
FP32、TF32 禁用、weight decay=0、梯度裁剪=10。50 步 warmup 到 1e-3，随后 cosine
降到最终 1e-4。这是此次 tiny 配置的诊断协议，不声称与历史大模型实验是严格配对比较。

每 250 步保存 checkpoint，并在固定独立噪声面板上评估：每样本 1 个几何噪声基底 ×
10 个分层均匀时间，共 190 个状态。额外 t=0 / 0.99 边界单列，不混入均匀时间期望。
初始与最终使用相同评估噪声；评估 RNG 不推进训练 RNG。

最终固定采用第 1,000 步 checkpoint，重载并核对权重、代码、数据身份后，以新 4 个噪声
基底评估（760 个内部时间状态）。另在训练前后用配对的 2 个纯噪声基底、20 步积分，
各生成 38 个候选；记录每个样本的序列恢复、CA 误差、旋转角、C–N 键误差和骨架几何代理。
保存带 context 的生成骨架 NPZ，坐标处于模型中心化坐标系，site_origin 可恢复源坐标。
仅有 N/CA/C 生成骨架，未生成完整侧链或做外部 Boltz/AF3 共同折叠。

5 步 CPU 预运行已完成，验证实际更新、评估、保存和重载路径；这部分仅用来确认流程和
运行速度，不用于判断过拟合或选择 checkpoint。主试验重新从随机初始化开始。

入口：`scripts/experiments/run_joint_v2_boltzgen_overfit.py`。
输出：`artifacts/experiments/boltzgen19_cpu_overfit1000_20260920/`。
运行日志：`artifacts/tmp/boltzgen19_cpu_overfit1000_20260920.log`。

```bash
source scripts/project_tmp_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python scripts/experiments/run_joint_v2_boltzgen_overfit.py \
  --dataset artifacts/datasets/boltzgen_native_four_20260920/dataset \
  --output artifacts/experiments/boltzgen19_cpu_overfit1000_20260920 \
  --steps 1000 --eval-every 250 --cpu-threads 4
```
