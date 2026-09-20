# BoltzGen 19-fragment continuation protocol: steps 1,000–5,000

Purpose: test whether additional training improves random-geometry denoising and free
generation with fixed data, architecture, losses, and sampling settings. This is an
exploratory experiment, not formal Joint-v2 training or unseen-structure evaluation.

- Preserve all 19 samples in their original order, full batch=19, and unchanged data identity.
- Restore weights, Adam state, independent noise RNG, and torch RNG from
  `boltzgen19_cpu_overfit1000_20260920/checkpoint_00001000.pt`.
  Parent checkpoint SHA256:
  `47d7d04f4914cb39cc04b48047af1dac729809d864e02a905e30b38a1ef607f0`.
- Keep the 124,976-parameter tiny model, FP32, four CPU threads, dropout=0, and the
  three equally weighted geometry/sequence losses.
- The original cosine schedule ended at step 1,000. Run 4,000 additional updates at
  constant endpoint LR=1e-4, with no warmup restart; AdamW weight decay=0, gradient clip=10.
- Fix the total budget at 5,000 steps; save a checkpoint and fixed denoising panel every 500 steps.
- At steps 3,000 and 5,000, evaluate denoising with four independent noise bases and free
  generation with 38 candidates. Reuse the parent's evaluation seeds for paired comparisons;
  evaluation randomness does not advance training RNG streams.
- Fixed panel: one base per sample, ten stratified times, plus separate t=0/0.99 boundaries.
  The fresh panel uses four bases.
- Rollout: 20 integration steps and two bases; no geometry repair or covalent-chain rebuilding.
  Report position, orientation, and backbone geometry separately from sequence recovery.
- Use the fixed step-5,000 checkpoint for the final conclusion. Intermediate panels do not
  select the budget or change hyperparameters.

All four H100 GPUs were occupied by other jobs, so this run continued on CPU.
At restoration, all 228 panel rows matched the parent step-1,000 model exactly, maximum
difference zero. A step-1,001 smoke completed optimization, saving, reloading, and sampling.
Five random-update replay/restoration tests passed.

Historical development entry: `scripts/experiments/continue_joint_v2_boltzgen_overfit.py`.
Run: `artifacts/experiments/boltzgen19_cpu_continue5000_20260920/`.
Log: `artifacts/tmp/boltzgen19_cpu_continue5000_20260920.log`.

```bash
source scripts/project_tmp_env.sh
python scripts/experiments/continue_joint_v2_boltzgen_overfit.py \
  --fit-run artifacts/experiments/boltzgen19_cpu_overfit1000_20260920 \
  --output artifacts/experiments/boltzgen19_cpu_continue5000_20260920 \
  --total-steps 5000 --eval-every 500 --milestones 3000 5000
```
