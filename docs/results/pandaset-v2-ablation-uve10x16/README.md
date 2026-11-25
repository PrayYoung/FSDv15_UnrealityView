# Step 1: 10x16 UVE ablation stopped at input parity

**No 10x16 training result exists.** The managed run stopped at the frozen
4x8 re-encoding parity gate. General CUDA preflight and verification of all 76
preserved caches passed. No new 10x16 cache, 20-step training preflight, optimizer
updates or new checkpoint was produced. The existing 4x8 best checkpoint and
21.884630dB validation baseline remain unchanged.

Source commit: `3cdd603983f5fa19fcd9a18eddf34e7b8720ef62`.
[Predeclared protocol](../../UVE_GRID_ABLATION.md), [execution summary](execution.json),
[unaltered returned parity measurements](parity.json).

## Measurements

The first training episode was re-encoded from stored quality 90 JPEGs at 4x8,
using the stored camera-slot order, intrinsics, extrinsics and masks. Per-camera
relative token RMS had to be <= 0.20 and cosine similarity >= 0.98. All six cameras
failed both frozen limits:

| Camera storage slot | Relative RMS | Cosine |
|---|---:|---:|
| front_left | 0.28755 | 0.95994 |
| front | 0.25913 | 0.96701 |
| front_right | 0.29737 | 0.95700 |
| rear_left | 0.26052 | 0.96691 |
| rear | 0.22947 | 0.97425 |
| rear_right | 0.27769 | 0.96256 |

The stop prevented an unqualified input change from being interpreted as the
spatial-grid-only experiment. This is a parity/execution failure, not evidence
that unpooled conditioning improves or worsens blur. Thresholds were not relaxed;
reserved test scenes were not evaluated. There are no new comparison images to
publish from this attempt.

## Analysis and next diagnostic

JPEG re-encoding is a known input discrepancy, but this measurement alone does
not identify its contribution. Check encoder weight identity, preprocessing,
batch-size/precision effects, calibrated camera order and geometry separately.
A lightweight one-episode diagnostic should compare batch 1 vs 8 and numerical
precision with the same JPEG inputs, separate semantic and ray contributions,
and report per-camera errors. Prefer local testing before another paid campaign.

If original decoded pixels cannot be recovered and the JPEG discrepancy remains,
a separately trained 4x8 JPEG-reencoded control may be needed to isolate the grid
change. That would be a new protocol/budget decision, not permission to lower the
parity threshold silently or launch two training runs automatically.

The earlier [Step 0 diagnosis](../pandaset-v2-pilot-5000/diagnosis/README.md) remains
valid for its saved checkpoint: train 26.34 dB vs validation 21.88 dB, later-frame
validation decay, and strong history sensitivity. A successful blur fix is still
unverified. No appearance-token, CFG, sharpness-loss or extra training run started.

The rental was reported destroyed. Workload 21 seconds, setup 193 seconds,
provisioning 244 seconds; approximate charge **$0.040**. Only the small parity JSON
was returned; the dataset was not downloaded back to the Mac.


The subsequent [free local Part 1 diagnostic](../pandaset-uve-parity-local/README.md)
quantifies batching, precision and ray-subtracted differences without modifying
the cache or starting any training.
