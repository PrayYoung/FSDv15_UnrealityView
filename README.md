# Mini-RealityView

A camera-conditioned Mean-Flow video model with a frozen shared DINOv2 encoder,
independent waypoint planner, streaming context and local research cockpit.
**PandaSet is the active training and testing data source.** MetaDrive is retained
as an optional legacy collector and is not used by the default remote workflow.

The first real-data experiment is a **native front-camera reconstruction baseline**.
It verifies the camera pipeline and is not an elevated Reality View. Novel chase
views need separately calibrated teacher exports. This independent research
project is unaffiliated with Tesla and does not control a vehicle.

## Start the local cockpit

Python 3.11 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --locked --extra dev --extra realdata
uv run --no-sync python -m app.server
```

Open `http://127.0.0.1:8088`. With no cache/checkpoint, the cockpit offers a labelled
procedural preview. No inference performance is fabricated. The server binds only
to loopback, validates paths/requests, rejects foreign origins, and loads only
safetensors model/cache files. No pickle or TensorFlow is needed for camera imports.

## Remote PandaSet pipeline check

This is the recommended first run. It downloads **only camera JPGs and calibration,
pose and timestamp JSONs**, using HTTP byte ranges into a public PandaSet mirror.
It does not download the complete 44.5 GB archive, LiDAR, or annotations. Source
revision, ZIP CRCs, byte counts and a frozen sequence split are recorded.

```bash
vast-run run --project-root . --gpu-model 'RTX 3090' \
  --remote-setup-command-file scripts/gpu_setup.sh \
  --command-file scripts/real_gpu_job.sh
```

The repository `.vast/profile.toml` is automatic. Global pricing/reliability
safeguards remain in force. The runner owns provisioning, logs, output retrieval,
and destruction. Workload time is capped at 45 minutes. Do not manually manage SSH
readiness or leave a debug instance running.

The job runs:

1. CUDA matmul, autocast/JVP/backward preflight, and the full test suite.
2. Selective download: train sequences **002, 003**, validation **014**, final test
   **001**, 64 native frames each, extraction limited to 2 GiB.
3. Pretrained DINOv2 + TAESD encoding into safe caches at the recorded ~10 Hz.
4. A small **30-step**, width-128/depth-4 integration training run.
5. Validation, then a separately invoked final test, 1/2/4-NFE metrics and a
   four-chunk autoregressive rollout. This is not a convergence/quality study.
6. Edge timing, decoder equivalence and compact artifact validation.

Raw data and encoded training caches stay under `.cache/` on the rented machine.
Only `outputs/pandaset_smoke/` evidence/checkpoints are retrieved. This temporary
instance is destroyed on completion, so its caches are not durable storage; a
larger campaign should put caches in user-owned remote object storage first.
The terminal runner result gives the exact local `synced_outputs` directory.

[Real-data details](docs/REAL_DATA.md) describe sensor geometry, splits, front-view
limits and external teacher exports. [Results](docs/PANDASET_RESULTS.md) records
the current remote outcome. Previous procedural-only results remain in
[historical local verification](docs/RESULTS.md).

## Use an existing PandaSet camera download

Expected layout: `<root>/<sequence>/camera/<camera_name>/{00.jpg, poses.json,
timestamps.json, intrinsics.json, ...}`. Six native camera names are supported:
`front_left_camera`, `front_camera`, `front_right_camera`, `left_camera`,
`back_camera`, `right_camera`. No LiDAR directory is required.

```bash
# Inventory all sequences and freeze whole-sequence train/val/test assignments.
uv run --no-sync python prepare_real_data.py plan \
  --pandaset-root /absolute/path/to/pandaset --output outputs/panda_splits.json

# Front-camera pipeline baseline. Native timestamps are preserved.
uv run --no-sync python prepare_real_data.py cache \
  --manifest outputs/panda_splits.json --output outputs/real_dataset --target front

# Real-data defaults: 512x12 DiT, effective batch 16, 20,000 steps.
uv run --no-sync python train_cached_meanflow.py
uv run --no-sync python evaluate.py --split val
# Only after choosing the final checkpoint; do not tune on these results.
uv run --no-sync python evaluate.py --split test --output outputs/final_test
uv run --no-sync python -m app.server
```

The normal local cache/checkpoint defaults are `outputs/real_dataset` and
`outputs/real_train/checkpoint`. Alternate paths use:

```bash
REALITY_DATA=/absolute/path/to/cache \
REALITY_CHECKPOINT=/absolute/path/to/checkpoint \
uv run --no-sync python -m app.server
```

A train, validation or test sequence is never split into frame-level windows
across partitions. Import refuses a changed split manifest in an existing cache.
Training audits source IDs across partitions; checkpoints preserve training IDs.
Final testing rejects overlaps and checkpoints without provenance. Mixed-source
sampling is available but PandaSet alone is the active experiment.

## Architecture and current scope

- **Shared vision:** frozen pretrained DINOv2-S/14, 6x32 pooled tokens of width
  384, calibrated Plücker rays in ego forward/left/up coordinates. Missing-camera
  tokens are zero and excluded from DiT cross-attention and policy attention.
- **Video model:** channel-concatenated previous chunk; 960 patches for 8x24x40
  latents; four context stages; AdaLN-Zero, QK RMS normalization, age embeddings;
  native 1/2/4-NFE sampling.
- **Objective:** improved Mean-Flow with stop-gradient directional derivative
  only on `r<t`, 50% boundary matching, adaptive weighting and previous-chunk noise.
  CUDA uses math-SDPA for JVP and normal SDPA selection for the primal pass.
- **Planner:** separate ~3.3M-parameter transformer, 12 ego waypoints and goal FiLM
  with 50% endpoint dropout. PandaSet's reference is explicitly the front-camera
  center with forward/left/up axes, not an assumed undocumented IMU origin.
- **Decoder:** frozen TAESD, latent scale 1, module-wise exact halo tiling.
  CUDA uses the native convolution reference backend for strict tile equivalence;
  reported timing includes this conservative precision choice.
  SRAM figures are analytical fused INT8-weight/FP16-buffer estimates; execution
  is floating point and full intermediates remain in DRAM.
- **Streaming:** immutable completed snapshots in a single-writer CPython ring.
  Cached replay emulates staged arrival and reports actual cache timestamps.
- **Cockpit:** six synchronized camera panels, source/episode/chunk controls,
  playback at native cadence, NFE, teacher-forced or autoregressive context,
  tiled decoder, telemetry. Chase-view tap-to-drive is disabled for front targets.

For Apple Silicon, `configs/apple.json` selects the smaller 384x8 model and FP16
with finite-difference JVP. MPS uses exact mean pooling, linear unpatchify and
FP32 attention reductions to avoid unsupported or unstable PyTorch 2.5 operations.
Checkpoints contain EMA inference weights/configuration; optimizer resume is not
implemented. Native 10 Hz recordings are not relabelled as 36 Hz observations.

The Waymo Perception v2 camera reader is retained as an optional, fixture-tested
adapter. It needs authorized official downloads; **real Waymo data has not been
used or validated**. The current run intentionally uses PandaSet only.

## Verification

```bash
uv sync --locked --extra dev --extra realdata
uv run --no-sync pytest -q
RUN_PRETRAINED=1 uv run --no-sync pytest -q -m pretrained
node --check app/static/app.js
```

Tests cover native camera/pose formats, calibration scaling, source timing,
missing-camera masks, frozen scene splits, train/test leakage, tile equivalence,
Mean-Flow derivatives and gradients, ring freshness, waypoint math, safe caches,
HTTP validation and closed-loop state. CI tests the source in a clean checkout.

## Data provenance and licenses

PandaSet was created by Hesai and Scale AI. Cite [PandaSet: Advanced Sensor Suite
Dataset for Autonomous Driving](https://arxiv.org/abs/2112.12610). The source format
is documented by the [official devkit](https://github.com/scaleapi/pandaset-devkit).
The selective downloader uses the **unofficial** [georghess/PandaSet mirror](https://huggingface.co/datasets/georghess/pandaset),
which declares CC BY 4.0. The retired `pandaset.org` domain is not used for downloads.
Source/revision and original author attribution accompany returned evidence.

Third-party dataset/model licenses apply separately from this repository's MIT
code license. No raw datasets, credentials or model weights are committed to Git.
Pretrained models: [DINOv2-S](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)
and [TAESD](https://huggingface.co/madebyollin/taesd).
