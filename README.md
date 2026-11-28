# UnrealityView

Camera-conditioned video generation with an independent waypoint planner and
switchable visual styles. **PandaSet cameras are the active training and testing
source.** This is an independent research prototype with no vehicle control.

The [stateless PixArt v2 pilot](docs/results/pandaset-p2-pixart-stateless-v2-2000/README.md)
completed 2,000 updates and scored **26.21 dB** across eight validation scenes,
versus the preserved v1 adapter's reproduced **25.12 dB**. Perceptual distance
fell 23.25%; static inspections show less blur, but fine detail remains weak and
temporal consistency did not improve under the measured proxies. V2 also uses
observed target-frame planar motion, so this is a bundled recipe/conditioning
comparison. The report includes one real-data diagnostic GIF and all comparison
panels. It is a native-front result; flying-camera and style quality remain unapproved.

The corrected PandaSet 5,000-step pilot passed its numerical gate: **21.81 dB
one-step validation PSNR**, beating frame copy at 18.75 dB, with an 83.5% pixel-error
reduction from initialization. Images still show blur and ghosting; this is not a
finished demo. See the [results, diagnostic visuals and analysis](docs/results/pandaset-v2-pilot-5000/README.md).
The [earlier failed pilot](docs/results/pandaset-pilot-5000/README.md) remains documented.
The [10x16 ablation attempt](docs/results/pandaset-v2-ablation-uve10x16/README.md)
stopped at JPEG/token parity before training. A [free local parity diagnosis](docs/results/pandaset-uve-parity-local/README.md)
found batching/precision differences too small to explain the discrepancy; its cause remains unresolved. The [one-scene 3DGS build check](docs/results/chase-3dgs-040-build-check/README.md)
passed native CUDA checks but stopped at archive-record writing before reconstruction.
Comic/cyberpunk adapters, real-data stop/resume and Style Studio are implemented
and tested, but satisfactory base visuals and style training remain pending.
No successful styled model result is claimed.

The [completed Arm E pilot](docs/results/pandaset-v3-arme-5000/README.md) increased
camera-conditioning sensitivity but scored 21.59 dB at its best checkpoint, below
v2's 21.88 dB, with worse later-frame decay. The [scene-040 3DGS run](docs/results/chase-3dgs-040-smoke/README.md)
completed 15,000 configured iterations, passed numerical coverage/held-out native
view checks and exported 80 elevated frames. Visual review found warped cars and
floaters, so those frames are **not approved as training targets or for batching**.
The [scene040 Difix3D replacement](docs/results/chase-3dgs-040-difix/README.md)
completed cleanup of all80renders and returned verified artifacts, but warped cars
and floaters remain; it is not approved as a training teacher. The associated
completed rentals are verified destroyed; costs and lifecycle failures remain
documented in the linked reports.

The [new scene040 paired-pose diagnostic](docs/results/chase-3dgs-040-v2-diagnostic/README.md)
passed its26dB native-view gate at **26.43dB**. Lower poseB plus640x384Difix looks
cleaner in the inspected panels;full temporal/human review is still required.
The [zero-training explicit3D proof](docs/results/pandaset-explicit3d-040-proof/README.md)
completed both depth variants and orbit samples,but camera-only poseB has **30.46%**
whole-frame projection holes,including near-ego gaps. Six-camera depth took about
157ms and Difix577ms/image in this configuration,so20–30fps is not established.
Both rentals are verified destroyed;Prompt3 remains on hold.

The approved [next diagnostic and gated dataset/training plan](docs/TWO_TIMESTAMP_HYBRID_PLAN.md)
tests strict live-priority t/t−5 projection and GPU FP16 depth on scene040 before
any12-scene export or Hybrid P2.5 training. Hole reduction,speed and moving-object
quality remain experimental questions;current-covered pixel identity alone is
not a general ghosting guarantee.

The independent [stateless PixArt-LCM pilot](docs/PROMPT_D_PROTOCOL.md) uses six
raw cameras from frame t-1 to predict the observed **native front camera** at t,
with1200tokens, LoRA, calibrated rays and no previous latent input. Its completed v2 GPU result is reported above; the
[v1 baseline](docs/results/pandaset-p2-pixart-stateless-2000/README.md) is preserved. Historical pilot measurements above
use their original chunk/history protocol and are not matched stateless results.
An external calibrated teacher is still required for elevated novel views.

## Setup and cockpit

Python 3.11 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked --extra dev --extra realdata
uv run --no-sync python -m app.server
```

Open `http://127.0.0.1:8088`. Import real recordings first; an empty installation
shows a load-data state. The cockpit offers six camera panels, native-cadence
playback, 1/2/4-NFE generation, closed-loop context and tiled decoding.
Style Studio offers Reality, Comic, Cyberpunk and Hybrid when compatible adapters
are installed, strength 0–1.5, and measured adapter size and merge time.

Defaults are `outputs/real_dataset`, `outputs/real_train/checkpoint`,
`outputs/adapters` and `outputs/styles`. Override with `UNREALITYVIEW_DATA`,
`UNREALITYVIEW_CHECKPOINT`, `UNREALITYVIEW_LORA_DIR`, and
`UNREALITYVIEW_STYLES_DATA`. The server accepts `--data`, `--checkpoint`, `--port`, `--lora-dir` and
`--styles-data-dir`; origin checks follow the configured localhost port. It binds to localhost and uses safetensors for weights/caches.

See the [gated pipeline roadmap](docs/PIPELINE_ROADMAP.md) for the isolated dynamic
3DGS flying-camera adapter, SSH recovery, stage scripts and verification limits.

## PandaSet training

The first quality campaign completed the [5,000-step pilot plan](docs/TRAINING_PLAN.md).
[Its negative result and next diagnostic steps](docs/results/pandaset-pilot-5000/README.md)
are recorded without claiming a successful demo.

Keep large camera files and encoded caches on the remote GPU's durable storage.
No dataset, weights, credentials or personal cloud runtime profiles belong in Git.
The selective downloader can fetch camera JPGs and calibration/pose/timestamp JSONs
without downloading LiDAR or the full archive; see [data workflow](docs/REAL_DATA.md).

```bash
uv run --no-sync python prepare_real_data.py plan \
  --pandaset-root /path/to/pandaset --output outputs/panda_splits.json
uv run --no-sync python prepare_real_data.py cache \
  --manifest outputs/panda_splits.json --output outputs/real_dataset --target front
uv run --no-sync python train_cached_meanflow.py
uv run --no-sync python evaluate.py --split val
```

The full configuration uses a width-512/depth-12 DiT with gated cross-attention
in every block, batch 16 / accumulation 1, and 20,000 optimizer steps. Training windows use stride 2; validation
and test use stride 8. The policy consumes tokens at the waypoint origin frame.
Choose the checkpoint using validation scenes. Final testing is a separate
`evaluate.py --split test --output outputs/final_test` invocation. Whole sequences
are split before caching; provenance checks reject train/test overlap.
[Recorded smoke results](docs/PANDASET_RESULTS.md) establish integration, not quality.

See [v2 fixes and the gated verification protocol](docs/MEANFLOW_V2.md) for the
implemented stability changes. The [Stage 1A one-window overfit check](docs/results/pandaset-v2-overfit/README.md)
passed at 30.76 dB one-step PSNR. The subsequent [full pilot](docs/results/pandaset-v2-pilot-5000/README.md)
passed numerical validation; visual quality remains insufficient for a public demo.

## Scaling and resuming

The selective fetcher accepts `--train-count`, `--val-count`, `--test-count`,
`--frames` (default 80), or `--all` for every available scene. Hash assignments stay
fixed. Set an explicit extraction budget up to `--max-gib 25` for a larger download.
Run it on remote storage; archive/image sizes depend on the selected mirror.

Both trainers ramp EMA decay with `min(target, (1+step)/(10+step))`. They preserve
latest and best validation 1-NFE PSNR weights (`checkpoint/`, `checkpoint_best/`).
Restart state uses safetensors plus an atomic, fingerprinted JSON manifest for raw
weights, optimizer, scaler, EMA, RNG and sampler position; no pickle is loaded.
Keep the same configuration, total-step schedule and cached data when resuming:

```bash
uv run --no-sync python train_cached_meanflow.py \
  --resume outputs/real_train/checkpoint
uv run --no-sync python train_style_lora.py --style comic \
  --output outputs/adapters/comic --resume outputs/adapters/comic/checkpoint
```

`--stop-after N` safely checkpoints at step N without changing the planned schedule.
CUDA training enables TF32 and pinned transfers. Decoder encode/decode batches all
frames and uses CUDA FP16/cuDNN by default (or an explicit caller autocast dtype).
DiT training remains BF16 where supported. `PretrainedTAESD(precise=True)`
selects FP32 reference convolutions for strict tiled-equivalence checks. Tiled decode
keeps its module-wise halos; an 8-latent full-decoder halo is not exact.

## Style adapters

Train a useful base before style adapters. Deterministic comic/cyberpunk teachers
use real PandaSet target JPEGs and bidirectional optical-flow confidence to reduce
flicker. They are a 2D temporal baseline, with no guarantee of 3D consistency or
zero flicker. Style caches stay separate; original vision tokens and poses remain
unchanged. Internal sanity grids compare reference, shader and TAESD round trip.

```bash
uv run --no-sync python prepare_style_cache.py \
  --data outputs/real_dataset --output outputs/styles --style both
uv run --no-sync python train_style_lora.py --style comic \
  --output outputs/adapters/comic
uv run --no-sync python train_style_lora.py --style cyberpunk \
  --output outputs/adapters/cyberpunk
uv run --no-sync python benchmark_styles.py
```

Style training defaults to rank 8 in blocks 0–3 and rank 16 in later self/cross
attention and MLP projections, plus the output projection. Base weights, encoder
and independent planner stay frozen. Use `--start-block 4` to omit early adapters. Training
uses improved Mean-Flow with JVP, adapter EMA, 80% styled/20% real previous context,
noise up to 0.25, effective batch 16 and 1,000 steps by default. Validation includes
real-to-style switching and a 32-frame autoregressive rollout.

Inference merges adapters into ordinary Linear weights and caches the selected
configuration. Reality restores exact base weights; Hybrid combines 0.55 Comic +
0.55 Cyberpunk at unit strength. The forward architecture stays unchanged; merging
has a separately measured cost. Equal wall times and unchanged visual geometry
are evaluation questions, not promises. The planner receives unchanged real tokens.

After quality training and held-out review, render the **single real-data GIF**:

```bash
uv run --no-sync python render_style_demo.py
```

It writes an internal 32-frame comparison under `outputs/demo/`. Publish one GIF
only after inspecting learned results. The README intentionally has no demo media
until that evidence exists.

## Optional elevated-view teacher

`uv sync --extra realdata --extra teacher` installs the optional CUDA
[gsplat renderer](https://docs.gsplat.studio/versions/1.5.3/apis/rasterization.html).
Use a CUDA development image with `nvcc` and a C++ compiler for gsplat kernel
building (the extra includes setuptools and Ninja). `prepare_chase_teacher.py` initializes a static 3DGS scene using calibrated SIFT
triangulation from cameras, fits each sequence independently, and exports the
existing elevated-camera manifest only after held-out reconstruction PSNR and
chase-view alpha coverage pass configurable gates:

```bash
uv run --no-sync python prepare_chase_teacher.py --pandaset-root /path/to/pandaset \
  --sequence 002 --output outputs/chase_targets
```

This is experimental: dynamic objects, occluded areas and unseen elevated surfaces
may fail. Native timestamps are preserved; no 36 Hz observations are invented.
Full real-scene teacher quality is not yet established. See [data details](docs/REAL_DATA.md).

## Implementation and verification

See the [implementation audit](docs/UPGRADE_REVIEW.md) and
[architecture and style design](docs/IMPLEMENTATION.md). The shared frozen
DINOv2-S/14 encoder produces 6×32 tokens with calibrated rays; the separate planner
predicts 12 ego waypoints. Frozen TAESD provides full/halo-tiled decoding. Streaming
uses immutable completed camera snapshots and timestamp-derived context ages.

```bash
uv run --no-sync pytest -q
RUN_PRETRAINED=1 uv run --no-sync pytest -q -m pretrained
node --check app/static/app.js
```

Tests cover import geometry, native timing, scene separation, Mean-Flow gradients,
LoRA JVP/freezing/merge/restore, planner isolation, temporal shaders, cache integrity,
request validation and closed-loop switching. Fixtures are for software tests and
are never presented as real-data model demonstrations.

## Attribution

PandaSet: Hesai Technology and Scale AI, [paper](https://arxiv.org/abs/2112.12610),
[official devkit](https://github.com/scaleapi/pandaset-devkit).
The selective downloader uses the unofficial
[georghess/PandaSet mirror](https://huggingface.co/datasets/georghess/pandaset),
which declares CC BY 4.0. Preserve original attribution when distributing derivatives.
Pretrained models: [DINOv2-S](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m)
and [TAESD](https://huggingface.co/madebyollin/taesd). Their licenses and dataset terms
apply separately from this repository's MIT code license.
