# Implementation audit before full training

The audit addresses software correctness and training readiness. It does not
establish convergence, a fixed speedup, zero flicker or elevated-view quality.
The prior native-front smoke used the older stage-attention architecture;
new training defaults to per-block cross-attention. Existing weights load under
their original architecture and cannot be silently reused as the upgraded model.

| Finding | Change and verification |
|---|---|
| Sparse windows | Train stride 2; val/test stride 8. An 80-frame sequence yields 18/5/5 windows. The 36-frame label horizon is retained. Explicit stride override is supported. |
| Policy lag | Separate exact-start policy tokens in dataset, both trainers' validation and serving. Tests distinguish them from earlier video-stage context. |
| Slow TAESD | One B×T encode/decode call; normal CUDA cuDNN/autocast. Strict reference backend is opt-in with `precise=True`; strict tile tests retain their tolerance. Fast and precise output errors are measured separately. |
| Weak context conditioning | Every DiT block has gated AdaLN-Zero self/cross/MLP branches. Freshness enters valid camera tokens before attention; missing-camera masks are preserved. JVP tests remain. |
| Early EMA lag | Both trainers use min(target, (1+step)/(10+step)). |
| Incorrect HUD metrics | Comic/cyberpunk use matching teacher PNGs; absent/hybrid references have no PSNR. Full-strength teacher identity is labelled. Video compute FPS excludes policy and one-off merge time; aggregate serial compute is separately reported. |
| Missing server flags | Data/checkpoint/port flags plus existing style paths, with localhost binding, matching-port origin checks, CSP and TrustedHost retained. |
| Hardcoded scene count | Configurable counts or all scenes, native 80-frame default and explicit extraction cap up to 25 GiB. Frozen whole-scene hash assignments stay fixed. |
| Missing restart/best weights | Safe optimizer/scaler/raw/EMA/RNG/sampler state, data fingerprints, checkpoint and checkpoint_best. Interrupted/resumed CPU training matches uninterrupted weights in tests. Same schedule/config/cache required. |
| Style clipping and switching | Comic 72/28 cel/color blend, cyberpunk road-line/taillight heuristics and bloom. Trainer defaults to early rank 8 / later rank 16, start block 0; start block 4 remains available. Full default adapter tensors measure 7.455 MiB. |
| Missing chase teacher | Optional calibrated-SIFT/static-gsplat script with per-scene fitting and held-out PSNR/chase-coverage gates. Geometry/triangulation are unit-tested. Full real-scene teacher quality remains unverified. |

CUDA validation exposed a strict-mode interaction: native convolution can use
GEMM, so disabling cuDNN TF32 alone was insufficient once training enabled matmul
TF32. Precise mode now disables and restores both flags, including on exceptions;
the original tiled tolerance is retained.

No new fast spatial mode with only an eight-latent full-decoder halo is claimed
exact: that halo does not cover TAESD's full receptive field. The module-wise tile
path remains available; normal full decoding is batched.

GPU verification uses one bounded runner-owned lease, public pretrained models,
real PandaSet camera subsets and short optimizer/resume checks. Returned evidence
stays under ignored outputs until summarized without personal runtime metadata.
Random-tensor capacity tests and short training are not public demo material.

## Completed verification

- CUDA suite: **42 passed**, including downloaded pretrained DINOv2/TAESD.
  Local CPU tests and an Apple GPU LoRA autocast/finite-JVP backward check passed.
- Full upgraded 512×12 DiT + policy optimizer capacity probe, batch 16: **9,461.6 MiB**
  peak PyTorch allocation in one random-input training step. This is not a full
  campaign's total GPU reservation or a guarantee on every 16 GB device. The real
  training configuration now uses batch 16 / accumulation 1; reduce batch and use
  `--grad-accum` to retain effective batch size when needed.
- TAESD default FP16 batched decode: **113.91 ms** versus **434.79 ms** for strict
  FP32 framewise decoding, **3.82×** in the same lease. Five repeats, two warmups,
  eight 192×320 frames. Strict full/tiled max error **8.11e-6**; FP16 versus strict
  mean/max absolute error **0.000802 / 0.02379**, in [-1,1] pixel units. A prior BF16
  probe showed a larger max difference, motivating FP16 for the frozen codec.
  Absolute timings varied substantially between machines with the same GPU model;
  these are compute probes, not interactive FPS or a universal 5–8× claim.
- gsplat 1.5.3 compiled successfully and passed a CUDA rasterization/backward check.
  The teacher extra includes setuptools/Ninja; a CUDA development image with nvcc
  and a C++ compiler is required. Full-scene teacher quality remains unverified.
- Real PandaSet: 002/003 train, 014 validation, 001 test; 64 native frames each,
  298 MiB camera/JSON extraction on remote scratch only. Upgraded small base trained
  to step 2, stopped safely, resumed to step 4. Both optical-flow style caches built
  across all splits. Each adapter stopped at step 1 and resumed to step 2; policy
  hashes remained unchanged. Latest/best weights and safe optimizer/EMA/RNG/sampler
  restart artifacts were retrieved and checked locally.
- Small-model merged DiT medians: Reality **11.759 ms**, Comic **11.797 ms**,
  Cyberpunk **11.810 ms**, Hybrid **11.823 ms**. Merge costs were separately
  **38.99 / 6.87 / 8.61 ms** for Comic/Cyberpunk/Hybrid; the first merge includes
  cold initialization and base backups. Adapter file reading/model hashing can add
  latency outside the merge timer. A signature cache hit means no merge is performed.
- Internal contact sheets were inspected: real camera inputs and deterministic
  style teachers are recognizable; learned predictions are still noise-like.
  No quality claim or public demo GIF follows from four base/two adapter steps.
- The managed job returned SUCCESS, local artifact integrity checks passed, and
  the paid instance was destroyed. No personal cloud runtime metadata is published.

Anonymous numerical evidence: [CUDA checks](evidence/upgrade_cuda_checks.json) and
[merged-style benchmark](evidence/upgrade_style_benchmark.json). Checkpoints and
internal media remain in ignored local outputs. Full quality training and held-out
visual evaluation remain necessary before publishing the single real-data GIF.
