# Official BoltzGen 19 fragments: 1,000-step overfitting results

The data interface supports actual optimization. The tiny configuration learned native
sequences and CA positions, but **did not fully overfit sequence and backbone structure**.
Rotation and backbone geometry errors remain substantial. High sequence recovery does
not imply a correct generated backbone, and these results do not establish the capacity
of the full-size base model.

See `reports/boltzgen19_overfit_protocol_20260920.md`. With all four H100 GPUs occupied,
the run used four CPU threads, a randomly initialized 124,976-parameter tiny model, and
full batch=19. Each sample received 1,000 exposures. Training plus intermediate/final
evaluation took about 1,065 seconds (17.75 minutes), excluding initial evaluation.
The samples, architecture, three losses, AdamW settings, and scheduled LR were unchanged.

## Fixed-noise denoising panel

The same 19 training samples, one geometry base per sample, and ten stratified times give
190 states. Evaluation has no gradients; separate endpoint boundaries are excluded here.

| Metric | Initialization | Step 1,000 |
|---|---:|---:|
| Translation MSE, Å² | 47.5930 | 0.29659 |
| Rotation tangent loss, rad² | 1.92591 | 1.56107 |
| Sequence CE | 2.99573 | 0.02923 |
| Raw total loss | 52.51469 | 1.88689 |
| CA RMS, Å | 6.89877 | 0.54460 |
| Sequence recovery | 6.97% | 99.47% |
| Mean residue rotation error | 67.22° | 58.64° |
| Backbone proxy pass | 3/190 | 6/190 |

Mean training losses over the last 100 steps were translation 0.28671 Å², rotation
1.40195 rad², CE 0.03630, and total 1.72496. Training logs mix changing weights and
noise, so they are reported separately from fixed-model panels.

After reloading the final checkpoint, four fresh bases across 760 internal-time states
gave CA RMS 0.52504 Å, sequence recovery 99.47%, mean rotation error 53.87°, and
26/760 backbone passes. The 76 additional t=0 states gave sequence recovery 95.24%,
CA RMS 0.94086 Å, rotation error 112.11°, and 0/76 passes. Rotation prediction from
pure-noise initial states therefore remained poor.

## Free generation

Initial and final models used the same two sets of starting noise: 38 candidates,
20 integration steps. Only conditions were inputs; native targets were not supplied.
No covalent-chain reconstruction or geometry repair was applied.

| Metric | Initialization | Step 1,000 |
|---|---:|---:|
| Mean candidate CA RMSE, Å | 11.76915 | 1.04851 |
| Sequence recovery | 6.97% | 96.99% |
| Exact complete sequences | 0/38 | 32/38 |
| Mean residue rotation error | 126.79° | 110.11° |
| C–N bond-length MAE, Å | 10.00499 | 1.89100 |
| Backbone proxy pass | 0/38 | 0/38 |
| Represented-atom clash/violation-free metric | 2/38 | 12/38 |

Native frames passed the same backbone and represented-atom clash checks in 19/19 cases.
This is a reference for generation, not a chemical audit of every omitted source atom.
Condition tensor hashes differed among fragments; no identical static condition with
conflicting supervision was found.

Rollout CA values are arithmetic means of per-candidate RMSE. Denoising-panel CA RMS is
the square root of mean translation loss. Sequence recovery averages per-sample residue
fractions, rather than pooling all residues into a different weighting scheme.

## Artifacts

- Run: `artifacts/experiments/boltzgen19_cpu_overfit1000_20260920/`.
- Final model: `checkpoint_00001000.pt`; SHA256:
  `47d7d04f4914cb39cc04b48047af1dac729809d864e02a905e30b38a1ef607f0`.
- `manifest.json` binds data, model, runtime contract, seeds, and source hashes. Source
  checks ran at all four checkpoint saves and after final sampling. Reloaded weights
  matched by hash and data identity rechecks passed.
- `completion.json`: execution results; status=completed means execution finished, not quality passed.
- `rollout_final/cases.json`: metrics and NPZ provenance for 38 candidates.
- `rollout_final/*.npz`: generated N/CA/C, predicted sequence, native reference, and context;
  centered coordinates with `site_origin` for restoration. Side chains were not generated.
- Summary: `artifacts/diagnostics/boltzgen19_cpu_overfit1000_20260920/`, including
  `summary.json`, `per_sample.csv`, `training_and_recovery.png`, and PDF.

Further overfitting work should first diagnose rotation prediction and backbone errors
from pure-noise sampling before expanding the data. This run did not change weights,
extend its budget, or select a better checkpoint to improve the reported result.
No formal training was started.
