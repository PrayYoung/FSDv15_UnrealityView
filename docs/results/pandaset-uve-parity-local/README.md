# Part 1: free local one-episode parity diagnosis

**Completed locally; no paid GPU, training, cache rewrite or threshold change.**
The original first training episode (PandaSet 040) was inspected on 8 evenly spaced
frames: 0, 11, 22, 33, 45, 56, 67, 79, all six camera slots. The full episode has 80 frames;
this sample is not the prior all 80-frame CUDA aggregate. Only JSON was produced.
The source episode tensor SHA256 stayed unchanged during each probe.

[Summary and fingerprints](summary.json), [native MPS measurements](mps_native.json),
[native CPU results including failed modes](cpu_native.json),
[controlled CPU FP32-ray measurements](cpu_fp32_ray_control.json).
[Recorded later suggestions, held for review](../../NEXT_STEPS_REVIEW.md).

## Findings

| Comparison | Relative RMS difference (range or maximum) |
|---|---:|
| MPS FP32: original cache vs JPEG re-encoding with rays | **23.75–29.68% range** |
| MPS FP32: batch 1 vs 8, stored-FP16 tokens | **0%** |
| MPS FP32 vs FP16, same JPEGs and batch 1 | **0.494%** |
| Controlled CPU FP16 vs BF16, batch 1 | **3.988%** |
| Controlled CPU FP32: batch 1 vs 8 | **0.0044%** |
| Ray-subtracted cached proxy vs fresh semantics, MPS FP32 | **24.51–30.07% range** |

The frozen parity limits remain relative RMS <= 20% and cosine >= 0.98. Every camera
still fails them on this sample. MPS FP32 cache/re-encoding cosine ranges from
0.9571 to 0.9723. Comparing original combined tokens with new semantics *without*
rays is worse (31.77–37.35% relative RMS); those are different representations,
so this is an explanatory comparison rather than a valid parity pass criterion.

Current batching differences and the tested precision effects are much smaller
than the cache/re-encoding discrepancy. This argues against them being its main
explanation on the inspected sample. The mismatch persists after subtracting a
known deterministic ray projection from the cached tokens, pointing toward a
semantic/input discrepancy rather than simply forgotten rays. It does not prove
that JPEG compression, weights, calibration or preprocessing is the unique cause.

## Precision and geometry caveats

This PyTorch 2.5.1 MPS implementation disables BF16 autocast. The script detects
that and reports it unavailable, rather than mislabelling FP32 execution as BF16.
Native CPU FP16/BF16 reached a ray-construction dtype conflict (`Float` vs `Half`
or `BFloat16`). Their failure is retained in the native CPU report.

A separate **diagnostic-only FP32-ray control** computes the rays outside autocast,
then runs the encoder/projection under the requested FP16 or BF16 CPU precision.
Both precisions complete in this controlled mode; this does not change production
MiniUVE or establish equivalence to its native low-precision CPU forward. The
same-JPEG FP16/BF16 comparison above is explicitly this control, not a CUDA result.

Ray subtraction is approximate because the stored cache rounded the sum, and
its original semantic and mixed-precision geometry contributions were not saved
separately. Source-level MiniUVE configuration and deterministic ray-projection
seed match the archived cache-building code; historical encoder weight hashes
were not recorded. Current CPU/MPS projection/weight fingerprints match each other,
but cannot establish the historical encoder's identity retrospectively.

Current projection SHA256:
`87a3302a42932e35f235f07f7a2cdcd56a97051f2b1bf80ab97abe2af577232c`.
First/last weight names, shapes and SHA256 plus the full backbone-state fingerprint
are in the JSON. No model weights, camera images or personal machine paths are
committed with this report.

## Decision

Stop after this local diagnosis. The cache mismatch remains unresolved; do not
relax its thresholds or claim a successful 10x16 experiment. Arm E, geometry fusion,
history dropout, paid pilots, archive-path fixes and the 3DGS retry remain deferred.
The original decoded pixels and historical weight fingerprints would help isolate
JPEG vs encoder differences. If those cannot be recovered, a separately approved
JPEG-reencoded 4x8 control may be needed before a clean spatial-grid comparison.

Tests cover source preservation, ray decomposition, fingerprints, camera masking,
nonfinite rejection, controlled precision and disabled-autocast detection.
