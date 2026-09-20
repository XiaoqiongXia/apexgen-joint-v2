# Official BoltzGen 19-fragment overfitting protocol

This experiment tests whether the working data interface supports actual optimization and
whether the Simplex model can memorize native sequences/structures under seen conditions.
It is not an unseen-structure generalization test, and a fixed budget does not guarantee
complete overfitting.

Data: `artifacts/datasets/boltzgen_native_four_20260920/dataset/`, split=`smoke`.
The four PDB sources `5hga, 5k9s, 4ep3, 7yxn` yield 19 independent continuous interface
fragments of length 4–9, totaling 106 target residues. Fragments are never concatenated.
Native frames pass the existing backbone and represented-atom clash proxies in all 19 cases;
this does not resolve the documented full-source-environment QC limitations.

All four H100 GPUs were occupied by other jobs, so execution used CPU. Full-batch=19
benchmarks at 1/2/4 threads measured approximately 2.28/1.26/1.01 seconds per step;
four threads were selected. Resource/timing records are under `artifacts/tmp/boltzgen*`.

Use the previously interface-tested tiny configuration:
`configs/joint_v2/experiments/sequence_structure_tiny_encoder_bottleneck_v1.yaml`.
This is **not an overfitting result for the full-size base configuration**. Initialize
randomly without loading an older model; dropout=0 and full rotation backpropagation
across decoder blocks. Runtime follows Simplex v3: Dirichlet alpha(t)=1+7t,
t∼Uniform[0,1), raw endpoint translation MSE, rotation tangent error, and sequence CE,
weighted 1:1:1. No extra bond/clash losses or covalent structure reconstruction are used.

Fix the budget at 1,000 AdamW updates, using all 19 samples each step: 1,000 exposures
per sample. Use FP32, disable TF32, weight decay=0, gradient clip=10, 50 warmup steps to
1e-3, then cosine decay to 1e-4. This diagnostic tiny-model protocol is not claimed to
be a strictly matched comparison with historical larger-model experiments.

Save every 250 steps and evaluate a fixed independent-noise panel: one geometry base per
sample × ten stratified uniform times = 190 states. Additional t=0/0.99 boundaries are
reported separately from the uniform-time expectation. Initial/final evaluation uses
paired noise and does not advance training RNG.

Use the fixed step-1,000 checkpoint for final evaluation. Reload and verify weights,
source code, and data identities, then evaluate four fresh bases (760 internal-time
states). Before and after training, generate 38 candidates from two paired pure-noise
bases using 20 integration steps. Record per-sample sequence recovery, CA error, rotation
angle, C–N error, and backbone proxy. Save backbone/context NPZ arrays in the model's
centered frame; `site_origin` restores source coordinates. Generate only N/CA/C, not
complete side chains, and do not run external Boltz/AF3 cofolding.

A five-step CPU prerun verified updates, evaluation, save/reload, and speed. It was not
used to judge fitting or select a checkpoint. The main experiment starts again from
random initialization.

Historical development entry: `scripts/experiments/run_joint_v2_boltzgen_overfit.py`.
Output: `artifacts/experiments/boltzgen19_cpu_overfit1000_20260920/`.
Log: `artifacts/tmp/boltzgen19_cpu_overfit1000_20260920.log`.

```bash
source scripts/project_tmp_env.sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
python scripts/experiments/run_joint_v2_boltzgen_overfit.py \
  --dataset artifacts/datasets/boltzgen_native_four_20260920/dataset \
  --output artifacts/experiments/boltzgen19_cpu_overfit1000_20260920 \
  --steps 1000 --eval-every 250 --cpu-threads 4
```
