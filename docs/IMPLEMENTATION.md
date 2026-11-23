# UnrealityView implementation

An independent research implementation of camera-conditioned Mean-Flow video
synthesis and a separate waypoint planner. Native PandaSet timing and camera
calibration are preserved. Elevated views require externally validated calibrated
teacher exports; the current real-data baseline reconstructs the native front view.

## Base model

- Frozen DINOv2-S/14: six cameras, 32 pooled tokens each, width 384, calibrated
  Plücker rays. Missing-camera tokens are masked from both video and policy attention.
- Four-stage AdaLN-Zero DiT, QK RMS normalization, previous-chunk concatenation,
  context age embeddings and linear unpatchify. Defaults: width 512, depth 12,
  8×24×40 latent chunks, 1/2/4-NFE sampling.
- Improved Mean-Flow: stop-gradient directional derivative on r<t samples,
  50% boundary matching, adaptive loss weighting, previous-context noise.
  Math-SDPA supports exact JVP; finite differences support FP16/MPS.
- Separate 12-waypoint transformer with goal FiLM and training goal dropout.
  Front-camera-centered forward/left/up coordinates are explicit.
- Frozen TAESD, latent scale 1, full or exact halo-tiled decode. CUDA uses native
  reference convolutions for strict equivalence. SRAM estimates are analytical;
  this implementation does not execute physical SRAM INT8 kernels.
- Immutable completed snapshots in a single-writer ring, four stage updates,
  timestamp-derived ages and availability checks. Cached playback is not live capture.

## Style extension

`models/lora.py` implements pure-PyTorch JVP-compatible LoRALinear with Kaiming A,
zero B and alpha/rank scaling. Only A/B train; all base parameters remain frozen.
Defaults rank=16, alpha=16, start_block=4. Attention q/fused-kv/out and MLP linears
are adapted in later blocks, plus output unpatchify. Stage-level cross-attention
is adapted only when its entire block group is after the frozen prefix.

Adapters carry a base-weight SHA-256 and export only adapter tensors in safetensors.
The inference manager validates every proposed merge before changing weights,
merges into ordinary Linear layers, caches signatures, and restores exact originals.
Hybrid is an additive 0.55/0.55 blend. Neither encoder nor policy is patched.
Merge latency is separate from steady inference latency. Identical forward
structure does not imply identical measured wall time or guaranteed geometry.

`data/stylize_cache.py` builds deterministic comic and cyberpunk references from
real target JPEGs. Bidirectional optical-flow confidence gates temporal blending;
disocclusions reset uncertain regions. This reduces some temporal instability but
is not a 3D-consistent renderer. Color/vertical heuristics are not semantic masks.
Separate style files record source-cache hashes; original tokens and poses stay
untouched. Internal sanity grids include real, styled and TAESD round-trip frames.

`train_style_lora.py` retains improved Mean-Flow/JVP, trains only adapters, tracks
adapter EMA, mixes 80% style/20% real previous context and caps noise at 0.25.
Defaults: 1,000 steps, effective batch 16, learning rate 2.5e-4, warmup/cosine decay.
Validation includes steady style, real-to-style switching, contact sheets and
32-frame closed-loop rollouts. `benchmark_styles.py` measures actual merged
forward latency and merge cost; `render_style_demo.py` creates one held-out real
PandaSet comparison GIF for later inspection. Neither command publishes media.

## Cockpit and verification

Localhost-only FastAPI, safe paths, strict request schemas, foreign-origin rejection,
safetensors weights/caches. Style Studio exposes compatible adapters, strength,
closed-loop context, merge cost and adapter size. Style-colored projected chevrons
are available only for elevated targets; front targets disable chase projections
and tap-to-drive. No vehicle control interface exists.

Software tests cover JVP gradients, frozen prefixes, merges/restoration, invalid
adapters, policy isolation, temporal shader determinism, cache integrity, safe HTTP
routes and sequential closed-loop switching. Fixture tests are not quality evidence.
A short real PandaSet CUDA base smoke is recorded in [results](PANDASET_RESULTS.md).
The style extension still needs meaningful base/style training and real-data quality
evaluation before publication. Performance, zero drift and zero flicker are
hypotheses to measure, never guaranteed results.
