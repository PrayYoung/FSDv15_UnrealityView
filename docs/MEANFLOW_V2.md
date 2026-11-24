# Mean-Flow v2 fixes and verification

This change follows the attached review and preserves the failed
[original 5,000-step pilot](results/pandaset-pilot-5000/README.md). No LoRA training,
OpenCV style caching or 3DGS reconstruction is part of the base verification job.

## Implemented

- Configurable `DiTConfig.time_scale`, default 10 for new models. Time, interval
  and freshness embeddings all use it. Loading an old checkpoint without the field
  selects 1000, preserving its numerical behavior. Archived training configurations
  also retain the original scale and loss settings; `configs/pandaset_pilot.json`
  is unchanged. New settings are explicit in `configs/pandaset_pilot_v2.json`.
- Exact and finite JVP paths use FP32 inputs with autocast and TF32 disabled.
  The exact path uses math SDPA. The derivative target is detached. Finite-difference
  probes stay inside the valid interval; invalid zero-width corner probes fail.
- New objective defaults: 75% boundary flow matching, 25% interior Mean-Flow;
  logit-normal ordered time pairs with mean -0.4/std 1; detached adaptive weighting
  p=1/c=0.01; configurable JVP weight. The v2 trainer ramps JVP weight from 0 to 1
  over 300 steps. Uniform sampling and the old adaptive parameters remain available.
- Weighted per-sample MSE is bounded by 1 for p=1. This bound does **not** imply
  bounded raw MSE, derivatives or good image quality. Raw residual diagnostics are
  still required. FP64 residual reduction protects extreme CUDA/CPU values; MPS
  uses FP32 because that backend has no FP64 support.
- Training records raw/boundary/interior MSE, derivative/target/prediction RMS,
  interior count, JVP weight/backend, precision, independent pre-clipping DiT and
  policy gradient norms, and separate clipping flags. Gradient accumulation records
  microbatch averages. Policy weight zero skips its training forward/backward.
- `--init-weights` transfers weights into model/EMA with a fresh optimizer and step
  counter; it is distinct from exact `--resume`, and mutually exclusive with it.
  Architecture changes other than an explicit time-scale transfer are rejected.
- Evaluation now reports TAESD round-trip and previous-chunk-copy baselines against
  reference JPEGs across every evaluated window. The codec score is an empirical
  reference, not a mathematical upper bound. An optional single-window raw/EMA
  comparison uses the same fixed noise. Existing visual diagnostics remain saved.

Adaptive weighting and ordered logit-normal sampling follow the MeanFlow design;
see the [official implementation](https://github.com/Gsunshine/meanflow) and
[paper](https://papers.nips.cc/paper_files/paper/2025/file/6d13e085b79d454da5910e4ca82a3d9d-Paper-Conference.pdf).
The chosen c=0.01, time scale and warmup are this experiment's settings, not a claim
that every value is prescribed by the paper.

## What the tests establish

Exact FP32 JVP matches finite differences on a nonzero-gate model under an outer
BF16 autocast context. At identical sinusoidal phases and weights, changing scale
1000 to 10 reduces the pure time derivative by exactly 100. At identical wall-clock
times the phases differ, and the full derivative also contains a spatial term;
there is no universal claim of a 100x reduction in the trained model's total JVP.
The former high frequency is a plausible instability mechanism, not a proven sole
cause of the failed pilot.

Tests also cover constant-velocity sampling at 1/2/4 NFE, extreme bounded weighted
loss, zero-JVP warmup, independent clipping, disabled policy training, warm-start
step reset, old-checkpoint output equivalence, numerical evaluation baselines and
visual artifact writing. They establish implementation behavior, not model quality.

## Real-data gate before the full pilot

`scripts/train_pandaset_v2_overfit_job.sh` runs CUDA checks and two matched 500-step
probes: corrected Mean-Flow versus boundary-only control. Both use the same first
training window, random seed, model, batch and training schedule, without jitter.
These are explicitly overfit diagnostics, not held-out quality scores. A separate
original validation scene remains in its original partition. Use preserved encoded
caches; no full archive download is needed. A requested two-minute duration is an
estimate, not a hard expectation for two full-size models plus evaluation.

Original submitted gate (review 1): both probes >=22 dB at 1 NFE, strict
nondecreasing PSNR, and derivative median 0.5–3/max <=10. The user supplied an
additional review after submission; that original run and verdict are preserved.

Amended gate (review 2): corrected Mean-Flow must reach >=22 dB at **1 NFE**;
the boundary-only control is scored separately at **4 NFE**, with both curves
recorded for both controls. Weighted loss remains <=1. Allow up to 0.5 dB decline
between evaluations, and also cap drawdown from the previous peak at 0.5 dB so
many small drops cannot silently accumulate into a large regression. Corrected
Mean-Flow post-step-100 derivative RMS median must be **0 < median <=3**, max <=10.
A zero directional derivative can be mathematically valid for constant velocity;
this positivity check is a diagnostic heuristic, not proof that zero means a bug.
Boundary-only training sees interval zero; even four-step inference uses unseen
nonzero intervals. Its four-step score is a comparison diagnostic, not proof of
matched train/inference conditioning or one-step competence. These heuristic derivative limits are diagnostic gates,
not mathematical properties of MeanFlow. The run writes `gate.json`; a numeric
pass still requires manual inspection of crisp road/vehicle geometry. No full
pilot starts automatically. Any criteria amendment is versioned and reported
alongside the original verdict; thresholds are not silently changed after seeing results.

Keep final/best probe weights, metrics and images; omit large optimizer-state
files from the diagnostic return payload. The managed runner retrieves outputs
and destroys the rental. No periodic agent log polling is needed. Cache preflight loads both train and
validation episodes and checks complete reference JPEG inventories before CUDA
checks or training. The local staging bundle already included both partitions.

Only after numeric and visual checks pass, run the non-LoRA front-target v2 pilot
for 5,000 steps on the original frozen 60/8/8 scenes, keeping test reserved and
using the original quality gate. Report both new baselines and a clearly labeled
32-frame closed-loop GIF in `docs/results/pandaset-pilot-v2-5000/`. If the small gate
fails, publish that outcome and diagnose before spending on the full pilot.

## Future teacher hooks, code only

Safe style slugs now work in adapter artifacts, style datasets and LoRA CLI.
Built-in OpenCV shaders still accept only their two implemented shader styles.
Future LoRA jobs inherit the stabilized objective defaults instead of forcing 50%
boundary samples. Neither shader caching nor LoRA training is run by this work.

Import externally rendered RGB appearance teachers with:

```bash
uv run --no-sync python prepare_style_cache.py import-teacher \
  --data outputs/real_dataset --teacher /path/to/export \
  --output outputs/styles --style jets_aliens --device cuda
```

Each `<split>/<episode>/` input contains exactly `00000.png` through the last frame
and `metadata.json` declaring `frames` and `base_sha256` (the SHA-256 of that base
`episode.safetensors`). RGB mode, target dimensions, complete inventory, source
fingerprint and safe paths are validated before encoding. Outputs include safe
latents, copied reference PNGs, per-frame hashes, metadata and a sanity image.
This importer executes no external generative model, OpenCV filter or pickle code.
Synthetic unit fixtures are the only teacher import executed during local tests.

The existing calibrated `prepare_real_data.py --target teacher --targets ...`
PNG/JSON contract remains unchanged for future dynamic 3DGS exports. DriveStudio
and OmniRe are possible external render sources, not installed or executed here.

## Regrading a submitted job without repeating GPU work

Training weights and metrics are unaffected by these gate-only changes. The existing
job keeps its original source snapshot. Once its outputs return, run:

```bash
uv run --no-sync python -m scripts.check_pandaset_v2_overfit \
  --output outputs/pandaset_v2_overfit --regrade-only
```

This writes a separately versioned `gate_review2.json` and never overwrites the
original `gate.json` or retrains a model. Report both verdicts and explain any
difference. A regrade still requires manual image review before the full pilot.

The first submitted verification attempt ended after successful CUDA/JVP checks
with SSH exit 255 (broken pipe), and returned no training or gate artifacts. The
runner labeled it `WORKLOAD_FAILED`; the observable failure is a connection loss,
not a negative model-quality result. No active rental remained. The revised gate
was recorded before inspecting any training metrics (none were returned).
