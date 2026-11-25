# Prompt A: Arm E pilot on the preserved 4x8 cache

User scope: complete all Prompt A, publish results, then proceed to Prompt B.
Prompt B remains an independent scene040 test; no76-scene batch or automatic
Difix. Each prompt permits one paid attempt capped at$1, retaining the existing
hourly ceiling. No paid retry on an undesirable or failed result.

## Frozen treatment and controls

The local one-episode parity diagnosis is complete and remains linked to its
original episode fingerprint. No cache rewrite or threshold change is needed for
this experiment: use the original384-channel cached tokens and append6 geometric
ray values at dataset access time. Reserved test scenes are neither uploaded nor
evaluated; train60/validation8 retain their original identities and benchmark.

The bundled treatment in `configs/pandaset_pilot_v3_arme.json` changes v2 only by
`dit.qk_logit_scale=true`, `dit.geometry_conditioning=true`, and
`prev_chunk_dropout=0.25`. All new defaults are false/zero for old models.
QK scaling applies to cross-attention only: exp(log-scale), initialized2, capped8.
Geometry mode initializes learned query positions with deterministic3D sin/cos,
normalizes cached channels and fuses Fourier rays with a trainable MLP and gains.
The cached384 channels already contain the historical fixed ray projection; this
proposal deliberately retains that existing contribution and adds learned fusion.
It is not a decomposition into perfectly pure semantic features.

Mask unavailable cameras before ray nonlinearities and after freshness addition.
Keep policy inputs384-wide. Sample previous-history dropout once before JVP so
both the derivative and main prediction see exactly the same dropped history.
No geometry modules/QK parameters exist when their flags are off, preserving
strict old checkpoint loading and the legacy forward path.

CPU checks: frozen v1/v2 source oracle bitwise forward equality, strict checkpoint
loads, unavailable-camera ray corruption invariance, exact-JVP finite gradients,
deepcopy, stage packing, LoRA manager/injection, and app/evaluation compatibility.
Run the complete suite before provisioning. Original v2 configuration is unchanged.

## Execution and budget

Use `scripts/train_pandaset_pilot_v3_arme.sh` through the global managed runner.
CUDA preflight, training-only cache validation and local parity-report fingerprint
check precede training. Train5k steps from scratch under the original20k cosine
schedule, batch16, seed42, policyweight0; no new dataset/model-encoder download.
The main decoder may use its existing public pretrained weights as before.
Then run the unchanged numeric front gate, select the actual validation-best
checkpoint, reproduce its40-window result, and diagnose40training scenes plus the
same40validation windows/seed917 and conditioning controls. Return all metrics,
initial/best/final previews, best weights and full exact-resume state; omit only
the verified duplicate export. Return no dataset or cached camera features.

Use `--max-cost-usd 1 --max-infra-retries 0` with the opt-in owner budget watchdog.
It starts after instance creation, estimates spend using the selected hourly rate,
and reserves15% for cleanup/other billing variation. The same runner owns destruction;
no manual provider APIs or SSH. The workload has a6,000-second training timeout
inside a bounded whole-command wrapper. A cutoff/failure is reported honestly;
it does not authorize a second paid attempt for Prompt A.

## Predeclared targets and report

Against v2 best21.884630dB, report all1/2/4NFE PSNR/pixelL1/latentL1, train/validation
gap, f0..f7 error profiles, reference/copy baselines and diagnostic visuals.

Conditioning targets: zero-history PSNR>=19.5dB; frozen-stage PSNR delta<=-0.15dB;
wrong-scene UVE delta<=-3dB; normalized attention entropy<=0.70 and effective
exp(logit-scale)>2 in all four stage means. Report per-stage min/mean/max scales,
raw log-parameter means, entropy normalized by log(valid keys), and mean maximum
attention probability. Collect health only in validation1NFE baseline inference,
never during training/JVP. Do not claim previously measured attention collapse:
the baseline has no direct attention-health measurement.

Quality targets: validation1NFE PSNR>21.885dB, less f0-to-f7 decay, and visibly
sharper reconstruction/rollout. Numeric target checks never substitute for visual
review. A bundled result cannot uniquely attribute changes to one component;
wrong-scene/zero-history controls remain out of distribution.

Publish under `docs/results/pandaset-v3-arme-5000/`. No automatic appearance
features, CFG, extended training or style adapters. Following the user's latest
instruction, complete reporting and then proceed to the separately bounded
Prompt B raw3DGS scene040 verification.
