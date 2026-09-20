# 官方 BoltzGen NPZ → Joint-v2 四例修复验证（2026-09-20）

## 结果

官方 `boltzgen/boltzgen1_train/targets.zip` 中的 `5hga, 5k9s, 4ep3, 7yxn`
已通过原始 NPZ → 界面清单 → 双向连续片段 → 统一 LMDB record → JointV2Dataset /
DataLoader → collate → SimplexCodesignModel 前向和反向的完整路径。
本次只读取四个指定 NPZ，没有进行全量处理，也没有运行优化器或正式训练。

| 项目 | 结果 |
|---|---:|
| 原始结构 | 4 |
| 蛋白链对 | 6 |
| 展开方向 | 12 |
| 最大连续接触段 | 132 |
| 长度不足 4、跳过 | 113 |
| 实际尝试转换 | 19 |
| 接受 / 拒绝 | 19 / 0 |
| CPU 模型前向、反向 | 19 / 19 通过，分成 8、8、3 三批 |
| 原子坐标回查最大误差 | 9.5367431640625e-7 Å |
| 独立 DataLoader 读取 | 19 条记录，5 个 batch |
| 自动测试 | 146 项通过 |

各方向的片段仍与之前四例的规则一致：每个最大连续接触段单独构成样本，至少 4 个残基；
不连接分隔的片段，不补入中间非接触残基。目标保留全部观察到的 N/CA/C，源位置和 C–N
几何必须连续。context 来自对侧单条链，按当前片段重算 5 Å 接触核心及 11 Å CB 环境。

## 代码修复

- `data/boltz_schema.py` 统一处理字符串原子名和旧版四整数编码；界面统计适配无 element
  的蛋白原子表，保持氢原子和缺失原子不参与重原子接触统计。
- `data/boltz_npz.py` 支持官方字符串原子名及扩展 bonds 表，无需独立 connections。
  对选中的标准蛋白残基，先核对残基名、Boltz token 和合法原子名模板，再识别元素。
  源文件若有 element 列仍强制交叉校验。
- 连接端点 chain 字段按 **chains 表行号**解释，分别核对全局 residue/atom 行号的归属；
  不再把 chain row 当作 asym_id。显式边界上的普通相邻 C–N 肽键允许裁断，其他目标交联拒绝。
- 若输入额外提供 cyclic_period，源目标链明确声明为环状时拒绝；本次官方四例没有该字段。
- 适配器映射契约更新为 `apexgen.boltz_structure_adapter.v2`。Dataset 和 collate 均强制
  检查 Boltz 映射哈希及目标 Joint-v2 契约哈希。旧 v1 Boltz 转换产物需要重新构建，
  不能只替换版本字符串或哈希。其他来源的非 Boltz 数据读取路径没有增加此来源专属限制。
- 新入口 `scripts/data/prepare_joint_v2_boltzgen.py` 从已解压 NPZ 直接构建小样本清单及数据集。
  必须显式提供结构 ID；独立 NPZ 以自身路径和 offset=0 作为字节来源，不使用 ZIP header offset。

AA 仍按名称映射，例如 Boltz ALA=2 → 模型 ALA=0；原子按残基对应的名称放入 target
atom14 / context atom38。源数组原子顺序和全局行号都不充当模型槽位编号。

## 验证细节

自动测试分为：旧 Boltz / 链对 / 分段回归 78 项，新增官方格式及运行时契约检查 35 项，
共享 Dataset / 原生监督 / 模型回归 33 项。涵盖所有 20 种标准 AA、打乱原子顺序、
缺失侧链、错误 token、错误原子名、连接端点 row/asym 区分、边界连接、非法交联、
环状声明、映射版本不一致以及多段独立样本。

四例全部 19 个样本还做了以下实际验证：

1. 逐原子回查官方 NPZ，核对残基、原子名称、源索引、掩码和坐标。
2. 离线读取历史旧格式对应记录，确认新旧样本 ID 集合、片段范围、AA、atom14/atom38
   坐标和 mask、目标 frames、context 连接 mask 逐项相同。历史 v1 记录未进入当前模型。
3. 用保留的 context 坐标直接计算距离，每个 target 残基都仍有 ≤5 Å 接触。
4. DataLoader 使用真实 collate 入口，变长 batch 通过模型输入契约校验。
5. 当前 tiny 配置的 SimplexCodesignModel 得到有限 loss 和有限非零梯度，条件侧坐标保持固定。
   修改 native target 的序列和坐标后，静态 condition 不变。
6. 默认零初始化输出头的首步非零梯度集中在输出层；额外对临时模型输出头施加小扰动后，
   45 个 encoder 参数张量有非零梯度，确认梯度能传回编码器。未保存该临时模型或 checkpoint。

## 输出与复现

结果目录：`artifacts/datasets/boltzgen_native_four_20260920/`。

- `interfaces.parquet` / `interfaces.csv`：四例六个原始链对及接触信息。
- `summary.json`：处理和模型验证汇总。
- `tensor_equivalence_and_contacts.json`：19 例张量对照与最终 context 接触检查。
- `validation.json`：验证计数和当前相关代码哈希。
- **`dataset/`**：实际供 JointV2Dataset 读取的目录，内含 manifest、LMDB shards、原始 NPZ 副本、
  mapping、完整分段结果及模型检查报告。

已执行命令（复跑需换一个尚不存在的输出目录）：

```bash
source scripts/project_tmp_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/data/prepare_joint_v2_boltzgen.py \
  --structures-dir artifacts/datasets/boltzgen1_train_official_20260920/training_data/targets/structures \
  --structure-ids 5hga 5k9s 4ep3 7yxn \
  --output artifacts/datasets/boltzgen_native_four_20260920 \
  --model-smoke
```

模型读取示例见 `docs/boltz_interface_fragment_training.md`。当前 split 为 `smoke`。
日志：`artifacts/tmp/boltzgen_native_four_20260920.log`、`boltzgen_adapter_tests.log`、
`boltzgen_additional_tests.log`、`boltzgen_shared_tests.log`。

## 适用范围

本轮解决官方格式兼容、连接编号和运行时映射校验，建立了四例数据到当前模型的可验证路径。
非标准残基、多坐标模型、缺失目标主链和不支持的交联仍会拒绝；历史旧文件中的连接字段
互换异常不做猜测修复。全结构的严重碰撞和未标注共价连接筛查尚未统一接入 NPZ 路径，
相关限制没有因格式兼容而消失。尚未建立正式训练集的同源去重/划分或验证生成质量。
