# ApexGen Joint-v2：在新服务器运行

本仓库提供 BoltzGen NPZ → 连续界面片段 → LMDB → 条件序列/结构生成模型的可运行核心，
以及 4 个 PDB 结构、19 个片段组成的 example。当前是探索性 Simplex 基线，不是正式训练锁。

## 安装

```bash
git clone https://github.com/XiaoqiongXia/apexgen-joint-v2.git
cd apexgen-joint-v2
source scripts/project_tmp_env.sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c requirements/portable-tested-py312.txt -e '.[dev]'
apexgen-simplex --help
```

依赖范围见 `pyproject.toml`，上述 constraints 固定本次 CPython 3.12 验证使用的直接依赖版本，
包括 PyTorch 2.10.0；不是完整传递依赖锁。GPU 上先确认
`python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'`。
以下命令默认为 CPU；支持 CUDA 的环境可将 `--device cpu` 换成 `--device cuda:0`。
这套入口使用单进程、单设备；不支持直接用 torchrun 启动多卡。

每个新 shell 先运行 `source scripts/project_tmp_env.sh`，让缓存和临时文件留在项目内。

## 直接使用 example

`examples/boltzgen19/dataset/` 已包含 manifest、LMDB、mapping 和 4 个来源 NPZ，无需再次下载。
预处理后数据约 2 MB，所有 19 个样本的 split 都是 `smoke`。来源、许可与哈希见
`examples/boltzgen19/README.md` 和 `SHA256SUMS`。

先跑两步验证环境：

```bash
apexgen-simplex train \
  --config configs/joint_v2/portable/simplex_tiny.yaml \
  --dataset examples/boltzgen19/dataset --split smoke \
  --steps 2 --device cpu --output runs/example-smoke
```

随后训练 1,000 步：

```bash
apexgen-simplex train \
  --config configs/joint_v2/portable/simplex_tiny.yaml \
  --dataset examples/boltzgen19/dataset --split smoke \
  --device cuda:0 --output runs/example-fit
```

这是新的小批次实验：batch size 4、固定学习率 0.001、AdamW、梯度裁剪 10。
它沿用之前的架构与三项 loss，但训练日程不同于历史 19 条 full-batch、warmup/cosine 的实验；
不应把新结果当成历史 checkpoint 的直接延续。

输出包含 `run.json`、逐步 `training.jsonl`、定期 checkpoint、带 SHA256 的 `latest.json`。
所有输出目录必须是新目录。训练入口只训练并保存；评估和采样由下面的独立命令运行。

## 模型与配置

唯一运行配置是 `configs/joint_v2/portable/simplex_tiny.yaml`，不要使用旧实验 YAML 的 flow/loss 字段替代它。

| 部分 | 当前配置 |
|---|---|
| 静态 context encoder | 1 block，single 32、pair 16，4 heads |
| 几何 decoder | single 64、pair 32，2 次共享参数的 IPA refinement |
| IPA | 4 heads，hidden 8，2 个 query/key 点、4 个 value 点 |
| 几何状态 | 每残基位置和旋转；context 固定，target 加噪 |
| 序列状态 | 20 维 simplex，模型输出 categorical logits |
| 时间 | Uniform[0,1)，每个训练步重新抽样时间和噪声 |
| 几何更新 | 局部 quaternion/translation 更新乘以 1−t；完整旋转反向传播 |
| 序列路径 | Dirichlet，alpha(t)=1+7t |
| loss | CA 终点位置 MSE + 旋转切空间误差 + native AA CE，默认等权 |
| 采样 | 20 步；位置 endpoint Euler、旋转 geodesic、序列 exponential midpoint |

`model.architecture` 可调 encoder/decoder 宽度、block 次数、IPA heads/points；decoder block 参数共享。
改变架构需重新训练。角度 head 保留但冻结，当前不训练侧链、FAPE、键长、键角或 clash 辅助 loss。
`training` 管理 batch、固定学习率、权重、精度、保存间隔与 alpha；`sampling.steps` 管理默认采样步数。
实际运行契约和有效配置写入 run/checkpoint。CPU 默认 FP32；可显式选择 `bfloat16` 网络精度，几何保持 FP32。

## 评估与采样

```bash
apexgen-simplex evaluate \
  --checkpoint runs/example-fit/checkpoint_00001000.pt \
  --dataset examples/boltzgen19/dataset --split smoke \
  --bases 2 --device cuda:0 --output runs/example-evaluation

apexgen-simplex sample \
  --checkpoint runs/example-fit/checkpoint_00001000.pt \
  --dataset examples/boltzgen19/dataset --split smoke \
  --bases 2 --device cuda:0 --output runs/example-samples
```

评估逐样本运行，分别报告自由 rollout 和 t=0、0.05、0.25、0.5、0.9 的加噪监督预测。
输出 `metrics.jsonl` 和 `summary.json`：CA RMSD、平均旋转误差、序列准确率、C–N 键误差及几何通过比例。
位置误差在固定 context 坐标系中计算，不进行结构对齐。各样本/噪声等权平均。
`--limit N` 取 manifest 前 N 个样本用于小规模检查；`--sampling-steps` 可覆盖配置中的积分步数。

采样只把 `batch.condition` 交给模型；目标的 native 序列/结构不进入采样器。
当前 CLI 从带监督的 LMDB 获取条件和目标长度，是给定原生界面位点/长度的生成测试，
不是任意新受体的独立输入文件接口。
每个输出 NPZ 包含 `generated_backbone`（L×3×3，N/CA/C，Å）、`generated_aatype`、
context 的 atom38 坐标/mask/AA 及 `site_origin`。生成与 context 坐标同为中心化坐标，
恢复原始平移坐标时加上 `site_origin`。AA/atom 顺序由 example 的 `mapping.json` 定义。

19 条 smoke 样本属于训练集上的记忆测试，没有独立验证集。换用独立数据可通过
`--dataset OTHER --split validation` 评估，但该入口不自动构建同源分组划分。
历史过拟合实验的骨架几何尚未全部通过，程序成功运行不代表设计已合格。

## checkpoint 迁移与续训

```bash
apexgen-simplex train \
  --config configs/joint_v2/portable/simplex_tiny.yaml \
  --dataset examples/boltzgen19/dataset --split smoke \
  --resume runs/example-fit/checkpoint_00001000.pt \
  --steps 5000 --device cuda:0 --output runs/example-continued
```

`--steps 5000` 指总步数，到 5,000 步结束。恢复模型、Adam 状态、训练噪声、CPU/CUDA RNG 和下一批次位置。
续训保持原配置、数据内容、split、设备类型和 PyTorch 版本一致；跨 GPU 硬件不承诺逐位复现。
模型/优化器状态可移动到新的服务器路径，数据身份按 manifest/metadata/shard 内容哈希校验。
评估和采样可以更换设备或数据集。checkpoint 内含模型配置，不需要原服务器的 config 或 run 文件路径。
建议使用同一 Git commit；入口不提供正式运行锁级别的全源码/依赖验证。
旧 `boltzgen_overfit.v1` checkpoint 仍使用旧脚本，不会被这个新格式静默加载。

## 导入新的 BoltzGen 数据

已解压官方数据后：

```bash
python scripts/data/prepare_joint_v2_boltzgen.py \
  --structures-dir /path/to/training_data/targets/structures \
  --structure-ids 5hga 5k9s 4ep3 7yxn \
  --output artifacts/datasets/new-panel --model-smoke
```

转换结果的 `dataset/` 可直接传入上述 train/evaluate/sample。
每个蛋白链对分别考虑 A→B、B→A；目标取长度至少 4 的每个最大连续接触段，分别作为样本。
不会把不连续片段拼接。原子按名称、AA 按残基语义映射，原始 NPZ 的整数编号不直接复用。
完整转换细节见 `docs/boltzgen_training_pipeline.md` 和 `docs/joint_v2_boltz_npz_adapter.md`。

## 核心代码导航与验证

| 功能 | 文件 |
|---|---|
| NPZ 读取、链对、连续片段、映射 | `src/apexgen/joint_v2/data/boltz_*.py` |
| 流式转换与清单 | `scripts/data/prepare_joint_v2_boltzgen.py` |
| LMDB、组 batch | `src/apexgen/joint_v2/data/dataset.py`、`batch.py` |
| 模型入口 | `src/apexgen/joint_v2/model/simplex_codesign.py` |
| encoder、decoder、IPA | `src/apexgen/joint_v2/model/encoder.py`、`decoder.py`、`structure_module.py` |
| 加噪、loss、积分采样 | `src/apexgen/joint_v2/sampling/simplex_runtime.py`、`dirichlet.py` |
| 可迁移训练/评估/采样 CLI | `src/apexgen/joint_v2/runtime/portable.py` |
| 配置 | `configs/joint_v2/portable/simplex_tiny.yaml` |

```bash
python -m pytest --basetemp="$TMPDIR/pytest" -q tests/joint_v2/test_portable_simplex.py
```

该测试覆盖 19 条 example 的来源映射审计、跨目录加载、训练/恢复的一致性，以及独立评估和采样。
