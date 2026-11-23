# PandaSet pipeline verification

A camera-only CUDA smoke run completed on an RTX 3090 with PyTorch 2.5.1/CUDA 12.4.
This report contains aggregate technical evidence only. The model is a native-front
reconstruction baseline; 30 optimizer steps leave predictions noise-like.
These are version-1 stage-attention results and the older strict decoder timing
path. See the [implementation audit](UPGRADE_REVIEW.md) for the upgraded model.
Full base training and style-adapter quality training have not been completed.
There is no public results GIF yet.

## Verified workload

- CUDA matmul, autocast, Mean-Flow JVP/backward and 24 tests passed at the smoke revision.
- Camera-only selective extraction: 1,608 camera/JSON files, 297.98 MiB. No LiDAR.
- Train sequences 002/003, validation 014, final test 001; 64 native frames each,
  approximately 10 Hz. Checkpoint provenance preserves these whole-scene splits.
- Small DiT: width 128, depth 4, 1,924,512 parameters; planner: 3,280,036 parameters.
- Thirty finite optimizer steps. Last flow loss 1.1658, policy loss 1.4897.
- Checkpoint, metrics and internal rollout artifacts were retrieved and validated.
  The temporary paid GPU instance was destroyed.

## Held-out metrics

Two chunks per split, shared noise across NFE variants. Pixel range is [-1,1].
The reference is the TAESD round trip of the observed front JPEG, which is also a
conditioning input. These scores do not establish prediction or novel-view quality.

| Split | NFE | PSNR (dB) | Pixel L1 | Waypoint XY L1 (m) |
|---|---:|---:|---:|---:|
| Validation 014 | 1 | 8.5564 | 0.60975 | 3.3747 |
| Validation 014 | 2 | 8.5566 | 0.60973 | 3.3747 |
| Validation 014 | 4 | 8.5566 | 0.60973 | 3.3747 |
| Test 001 | 1 | 8.8426 | 0.59075 | 8.8268 |
| Test 001 | 2 | 8.8428 | 0.59074 | 8.8268 |
| Test 001 | 4 | 8.8428 | 0.59073 | 8.8268 |

Similar NFE scores at this stage do not establish one-step quality. Four rollout
chunks have latent L1 0.9975, 0.9975, 0.9972 and 1.0052. Zero drift is not established.

## Small-model compute benchmark

Three repeats after two warmups, batch 1, eight frames. BF16 DiT, FP32 TAESD;
precision-sensitive decoder calls use native convolutions with cuDNN/TF32 disabled
for strict full/tiled equivalence. Times include generation, decoding and planner;
exclude encoder, capture, transport and browser rendering. These are neither full
512×12 model measurements nor end-to-end interactive FPS.

| Variant | Median (ms) | Compute FPS | Peak allocated VRAM (MiB) |
|---|---:|---:|---:|
| 1-NFE full | 44.93 | 178.04 | 224.75 |
| 1-NFE tiled | 122.44 | 65.34 | 132.11 |
| 2-NFE full | 50.16 | 159.48 | 224.75 |
| 2-NFE tiled | 127.65 | 62.67 | 132.11 |

Planner median: 2.11 ms. Full/tiled max absolute error: 1.2159e-5.
Analytical SRAM estimate: 25.166 MiB; physical SRAM use is not measured.
Full GPU intermediates/workspaces remain allocated. Recordings retain native ~10 Hz.

Aggregate JSON: [validation](evidence/pandaset_validation_metrics.json),
[test](evidence/pandaset_test_metrics.json), [training](evidence/pandaset_training_summary.json),
[rollout](evidence/pandaset_test_rollout.json), [compute](evidence/pandaset_benchmark.json).
No synthetic previews or undertrained prediction images are published.
