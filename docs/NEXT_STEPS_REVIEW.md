# Follow-up suggestions and current scope

The review has two workstreams. The user subsequently clarified: **complete all
Prompt A, publish its result, then complete Prompt B.** The local Part 1 remains
completed; [the Arm E protocol](ARM_E_PROTOCOL.md) records the newly authorized
pilot and budget. The earlier limited scope below is historical; parity thresholds remain RMS<=0.20/cosine>=0.98, and the existing cache,
checkpoints, split allocation and reserved test scenes stay untouched.

## Active: local parity diagnostic

`scripts/diagnose_uve_parity.py` reads one non-test 4x8 episode and writes JSON only.
By default sample 8 evenly spaced frames, all 6 camera slots. Compare original
cached tokens with fresh JPEG encodings with/without ray projections, and compare
a ray-subtracted cached semantic proxy with newly encoded semantics. Compare batch 1
and 8, FP32/FP16/BF16 where supported. Report unavailable precision/backend modes
explicitly; do not silently fall back to a different precision. Record SHA256 for
the current ray projection, first/last backbone weights and entire backbone state.

A ray-subtracted cache is only a proxy: the original cache already rounded the
sum of semantics and projected rays. Historical encoder fingerprints were not
saved, so current fingerprints cannot establish historical identity. Local MPS/CPU
results are not bitwise CUDA reproductions; an 8-frame sample is not the previous
all 80-frame episode aggregate. Same-JPEG comparisons isolate current batch and
precision effects but cannot directly measure damage relative to missing decoded
source pixels. Do not infer a unique cause from similarity alone.

## Deferred Prompt A: Arm E on the existing 4x8 cache

The proposed bundle, not implemented in this step:

- Add opt-in per-head cross-attention QK scaling, initialized log(2), exponential
  scale capped 8; leave self-attention and all legacy defaults unchanged.
- Add opt-in deterministic 3D sin/cos query initialization, balanced semantic
  LayerNorm, and learnable Fourier Plucker-ray fusion. Proposed packed input is
 384 semantic channels plus 6 rays; keep planner inputs at 384.
- Mask unavailable cameras before normalization/fusion and after freshness addition;
  verify corrupting masked rays cannot affect outputs.
- Share geometry-aware stage packing across training, evaluation/rollout, diagnosis,
  style training and app rendering without changing stored legacy cache tensors.
- Add optional previous-chunk dropout, default 0; proposed experiment 0.25. Sample
  its mask before JVP so derivative and prediction see identical history.
- Add optional cross-attention entropy, maximum attention and learned-scale
  diagnostics without changing frozen metrics schemas.
- Verify strict legacy checkpoint/output compatibility, missing-camera invariance,
  deepcopy, exact JVP/backprop, LoRA and app/evaluation compatibility on CPU first.
- Only after explicit continuation: one managed 5k pilot, existing 4x8 cache, unchanged
  validation/test allocation, comparative diagnosis and compact return. Suggested
  limit is one paid run <=$1; keep the existing hourly cap and stronger runner caps.

Treat the proposed readouts as predeclared targets for that future experiment,
not established capabilities: zero-history PSNR>=19.5 dB, frozen-stage delta<=-0.15 dB,
wrong-scene UVE delta<=-3 dB, normalized attention entropy<=0.70, learned scale > 2;
quality should exceed 21.884630 dB with improved frame decay and visible detail.
Attention collapse has not yet been directly measured. Existing cached semantic
channels already contain a frozen ray projection: decide explicitly how the new
ray fusion handles that contribution. The proposed bundle changes multiple
variables, so its result would not isolate one causal fix. Stop after reporting
for user review; no automatic 10x16 or appearance-feature experiment.

## Deferred Prompt B: archive-path fix and one-scene 3DGS

First, local-only fixes: reject a directory-valued archive-record destination
before disk/network work; unit-test that the downloader is never called. Keep
working outputs separate and collect minimal artifacts on EXIT, including failures,
into `outputs/chase_teacher_return/`. Use that DIRECTORY as the first sync path.
Whitelist archive/quality/calibration/manifest JSON, inspection PNG, raw GIF,
optional already-created Difix GIF, 80 RGB targets and small ingestion JSONs.
Exclude alpha sequences, dataset/archive, build environments and fit checkpoints.

Only after explicit continuation: one managed scene 040 / 15k iteration verification,
raw 3DGS first, Difix off, <=$1 budget and existing hourly cap; verify ingestion,
report stage timings/cost, held-out PSNR, whole/ground/ego coverage and an honest
visual review of floaters, sky holes and ego gaps. Stop for review before Difix
or 76-scene batching. This future work is not started by the active parity check.

Unrelated cloud-housekeeping questions appearing in the screenshots are outside
this request; no other cloud resources are inspected or changed.
