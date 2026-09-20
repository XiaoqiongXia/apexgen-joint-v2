# BoltzGen 19 fragments: results after 5,000 training steps

The fixed-budget continuation from step 1,000 to step 5,000 completed. Additional training
improved orientation and position prediction in the tiny configuration. Final free-rollout
rotation error was 64.07°, with 0/38 candidates passing the backbone proxy. Geometry metrics,
not completion or sequence recovery, determine whether quality criteria are met. This
experiment does not establish that further training is ineffective or fitting is impossible.

See `boltzgen19_continue5k_protocol_20260920.md`. The run retained the 19 records/order,
124,976-parameter model, equal loss weights, FP32, four CPU threads, Adam moments, and
random streams. After the original cosine schedule, continuation kept LR=1e-4 without
restarting warmup, increasing LR, or changing the rotation target. The 4,000 additional
updates and evaluation took 69.99 minutes.

## Free generation with the same two noise bases

Each checkpoint produced 19 samples × 2 bases = 38 candidates using 20 integration steps.
Only conditions were supplied; native targets were not inputs. Sample IDs and seed order
matched across checkpoints. N/CA/C was reconstructed directly from per-residue R/t,
without covalent-chain rebuilding or geometry repair.

| Total training steps | Mean CA RMSE Å | Mean rotation error | C–N MAE Å | Sequence recovery | Backbone proxy pass |
|---|---:|---:|---:|---:|---:|
| 1000 | 1.0485 | 110.11° | 1.8910 | 96.99% | 0/38 |
| 3000 | 0.7276 | 92.22° | 1.4605 | 100.00% | 0/38 |
| 5000 | 0.5792 | 64.07° | 1.0313 | 100.00% | 0/38 |

The proxy includes C–N MAE≤0.15 Å and represented-atom clash checks. It does not establish
all-atom validity or binding ability. The 19 fragments come from four PDB structures;
the 38 candidates are not 38 independent complexes.

## Fresh-noise denoising panel

Reuse the original run's four evaluation bases, independent of training randomness.
Ten stratified internal times per sample/base give 760 states. The 76 t=0 boundary
states are reported separately. Rollout CA values average candidate RMSE; panel CA RMS
is the square root of mean translation MSE.

| Total training steps | Internal-time CA RMS Å | Internal-time rotation error | t=0 CA RMS Å | t=0 rotation error |
|---|---:|---:|---:|---:|
| 1000 | 0.5250 | 53.87° | 0.9409 | 112.11° |
| 3000 | 0.4380 | 43.61° | 0.8106 | 97.52° |
| 5000 | 0.4358 | 34.03° | 0.7984 | 82.90° |

Per-time and per-sample results are in `fresh_panel_*.json`, `panel_*.json`, and
`rollout_*/cases.json`. Training curves use a 50-step moving average and do not replace
fixed-noise evaluation.

## Interpretation and next experiments

The paired improvement shows learning is occurring rather than rotation gradients being
absent. Nevertheless, final orientation and bond geometry remain underfit at this budget.
Low-time inputs require large rotation corrections; high-time ground-truth path inputs
already approach native geometry and cannot establish free-generation ability.

Suggested follow-ups are a finite four-structure, multiple-noise/time panel to separate
seen-noise fitting from unseen-noise performance, then separate tests of low-time
oversampling and decoder refinement depth 2→4. Orientation auxiliary supervision and
rotation-loss weighting require their own controls. The development report
`boltzgen_rotation_learning_options_20260920.md` contains the proposed objectives and
reference implementations.

None of these strategies was introduced during this run, and intermediate results did
not replace the scheduled final checkpoint. These results alone do not isolate capacity,
LR, noise coverage, time gating, or loss scale as an independent root cause.

## Verification and artifacts

- All 228 restored fixed-panel rows exactly matched the parent, maximum difference zero;
  weights, Adam state, noise RNG, and torch RNG passed exact hash checks.
- The one-step smoke saved, reloaded, and sampled; five replay/restoration tests passed.
- Step-3,000 and step-5,000 evaluation models were reloaded from disk, with hashes matching
  the presave weights.
- Source/data identities matched before and after the run. No formal training was started.
- Final checkpoint: `artifacts/experiments/boltzgen19_cpu_continue5000_20260920/checkpoint_00005000.pt`.
- SHA256: `c8ef26ce3db7e4f999db2029d9b5510a87e01b53214b7a803e3ce5d5f2d976fc`.
- Run directory: `artifacts/experiments/boltzgen19_cpu_continue5000_20260920/`.
- Summary: `artifacts/diagnostics/boltzgen19_continue5000_20260920/`, containing summary.json,
  milestones.csv, per_sample.csv, continuation_results.png, and PDF.
- Summary script: `artifacts/tmp/summarize_boltzgen19_continue5k.py`.

![Training and structure recovery](../artifacts/diagnostics/boltzgen19_continue5000_20260920/continuation_results.png)
