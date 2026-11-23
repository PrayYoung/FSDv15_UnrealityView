# UnrealityView implementation

An independent research implementation of camera-conditioned Mean-Flow video
synthesis and a separate waypoint planner. Native PandaSet timing and camera
calibration are preserved. Elevated views require externally validated calibrated
teacher exports; the current real-data baseline reconstructs the native front view.

## Base model

- Frozen DINOv2-S/14: six cameras, 32 pooled tokens each, width 384, calibrated
  Plücker rays. Missing-camera tokens are masked from both video and policy attention.
- Four-stage DiT with per-block AdaLN-Zero 9-way self/cross/MLP modulation, QK RMS
  normalization, previous-chunk concatenation,
  freshness embeddings added to valid camera tokens before attention, and linear unpatchify. Defaults: width 512, depth 12,
  8×24×40 latent chunks, 1/2/4-NFE sampling.
- Improved Mean-Flow: stop-gradient directional derivative on r<t samples,
  50% boundary matching, adaptive loss weighting, previous-context noise.
  Math-SDPA supports exact JVP; finite differences support FP16/MPS.
- Separate 12-waypoint transformer with goal FiLM and training goal dropout.
  Front-camera-centered forward/left/up coordinates are explicit.
- Frozen TAESD, latent scale 1, full or exact halo-tiled decode. Normal CUDA uses batched cuDNN/autocast; opt-in `precise=True` uses native
  FP32 reference convolutions for strict equivalence. SRAM estimates are analytical;
  this implementation does not execute physical SRAM INT8 kernels.
- Immutable completed snapshots in a single-writer ring, four stage updates,
  timestamp-derived ages and availability checks. Cached playback is not live capture.

## Style extension

`models/lora.py` implements pure-PyTorch JVP-compatible LoRALinear with Kaiming A,
zero B and alpha/rank scaling. Only A/B train; all base parameters remain frozen.
The style trainer defaults to start_block=0, early rank=8 (blocks 0–3), later
rank=16 and alpha=16. Each adapted block includes self-attention and cross-attention
q/fused-kv/out plus MLP projections; output unpatchify is adapted too.
`--start-block 4` retains a frozen early prefix. Version-1 stage-attention checkpoints
load explicitly as the legacy architecture, with their original adapter targets.
New per-block models require new training; their adapters cannot be attached to an
old model because weight fingerprints and architecture differ.

Adapters carry a base-weight SHA-256 and export only adapter tensors in safetensors.
The inference manager validates every proposed merge before changing weights,
merges into ordinary Linear layers, caches signatures, and restores exact originals.
Hybrid is an additive 0.55/0.55 blend. Neither encoder nor policy is patched.
Merge latency is separate from steady inference latency. Identical forward
structure does not imply identical measured wall time or guaranteed geometry.

`data/stylize_cache.py` builds deterministic comic and cyberpunk references from
real target JPEGs. Bidirectional optical-flow confidence gates temporal blending;
disocclusions reset uncertain regions. This reduces some temporal instability but
is not a 3D-consistent renderer. Comic teachers blend 72% posterized / 28% saturated color to preserve shading.
Cyberpunk adds smooth bright-white/yellow road-line and red-chroma taillight masks
with bloom. These color/vertical heuristics are not semantic masks.
Separate style files record source-cache hashes; original tokens and poses stay
untouched. Internal sanity grids include real, styled and TAESD round-trip frames.

`train_style_lora.py` retains improved Mean-Flow/JVP, trains only adapters, tracks
adapter EMA, mixes 80% style/20% real previous context and caps noise at 0.25.
Defaults: 1,000 steps, effective batch 16, learning rate 2.5e-4, warmup/cosine decay.
Validation includes steady style, real-to-style switching, contact sheets and
32-frame closed-loop rollouts. `benchmark_styles.py` measures actual merged
forward latency and merge cost; `render_style_demo.py` creates one held-out real
PandaSet comparison GIF for later inspection. Neither command publishes media.

## Training state and data use

Train windows overlap at stride 2; validation/test stride remains 8. Future waypoint
labels still require a 36-frame horizon: the withheld tail is needed for label
completeness and is not discarded accidentally. Eighty frames produce 18 train
windows and five validation/test windows (the conservative exclusive end is kept).
`policy_tokens` comes from frame start, separately from the earlier video-stage
snapshots, and all train/evaluation/serving policy paths follow that alignment.

EMA ramps with min(target, (1+step)/(10+step)). Both trainers save safe restart state
(raw weights, EMA, AdamW, scaler, RNG, sample permutation/cursor) and best validation
1-NFE PSNR weights. Cached data and hyperparameters must match on resume. Latest
inference weights are EMA; resumable raw training weights are stored separately.
Content-addressed safetensors are committed by an atomic JSON pointer with a hash;
no optimizer pickle is accepted. Use the same total-step schedule and output root.

The optional `prepare_chase_teacher.py` uses calibrated SIFT triangulation and static
gsplat optimization per scene, holding out camera views for reconstruction PSNR.
Only after PSNR and chase alpha-coverage gates pass does it write an importer-readable
manifest. This cannot guarantee quality in moving traffic or unseen elevated regions.
Native camera timestamps are preserved, with no interpolation claim.

## Cockpit and verification

Localhost-only FastAPI, safe paths, strict request schemas, foreign-origin rejection,
safetensors weights/caches. Style Studio exposes compatible adapters, strength,
closed-loop context, merge cost and adapter size. Style-colored projected chevrons
are available only for elevated targets; front targets disable chase projections
and tap-to-drive. No vehicle control interface exists. Comic/cyberpunk PSNR uses
matching full-strength teacher PNGs where available; hybrid/missing teachers show
no metric. Video compute FPS excludes independent policy compute and one-off
merge time; serial request compute is reported separately. Configurable ports
remain loopback-only and retain exact-origin checks.

Software tests cover JVP gradients, frozen prefixes, merges/restoration, invalid
adapters, policy isolation, temporal shader determinism, cache integrity, safe HTTP
routes and sequential closed-loop switching. Fixture tests are not quality evidence.
A short real PandaSet CUDA base smoke is recorded in [results](PANDASET_RESULTS.md).
The style extension still needs meaningful base/style training and real-data quality
evaluation before publication. Performance, zero drift and zero flicker are
hypotheses to measure, never guaranteed results.
