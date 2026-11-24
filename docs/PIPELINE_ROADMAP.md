# Gated front-view → flying-view → style pipeline

Execution is explicit and ordered. Run Stage 1A first and inspect its real outputs.
Do not automatically start the full pilot, a 76-scene batch, or style training.
The user-requested future path is dynamic 3DGS reconstruction, not a relabeled
front-camera image or a claim that the legacy static teacher handles moving traffic.

## Current implementation and evidence

The repository contains the isolated dynamic-teacher adapter and stage scripts.
CPU tests verify coordinate geometry, safe extraction, manifest acceptance by the
core teacher cache importer, and stage gates. They do **not** establish successful
DriveStudio compilation, sharp moving cars, reconstructed distant buildings, or
Difix quality on a GPU. Those remain gated experiments, not completed results.
Stage 1A has been submitted through the corrected detached global runner; its run
identity and recovery paths are private, outside Git. No Stage 2 batch is running.

[Mean-Flow v2 changes](MEANFLOW_V2.md) and the
[failed original pilot](results/pandaset-pilot-5000/README.md) remain documented.
The original 60/8/8 source allocation and reserved test scenes are unchanged.

## Remote execution first

The global `vast-run` owns provisioning, hourly caps, retries, synchronization and
cleanup. [Detached transport integration](../tools/remote_runner/README.md) adds
keepalives, a remote session/lock, durable logs, an atomic exit-code file and bounded
reconnection to the **same worker**. It does not launch duplicate work after a
lost acknowledgement. A temporary SSH exit 255 is not a model-quality failure.
Permanent host loss remains an infrastructure failure, with bounded recovery.

The superseded retry was still provisioning and was terminated through its
runner's signal handler before a replacement was submitted. Cleanup was checked;
no duplicate rental was left behind. The detached runner has a six-hour overall
command bound, while Stage 1A itself has a 30-minute workload timeout. Keep the
existing $0.35/hour ceiling; absence of an acceptable GPU is not permission to
increase it. Controller monitoring stays quiet rather than streaming every step
through the chat. Outputs and remote logs return on terminal completion.

Use a clean staging bundle of tracked files plus explicitly selected data; the
current global snapshot implementation does not reliably honor `.gitignore`.
Never upload `.private`, cloud credentials, personal runtime profiles or unrelated
workspace outputs. Large data and checkpoints are not Git artifacts.

## Stage 1A — prove the base can fit real data

Run `scripts/train_pandaset_v2_overfit_job.sh` using the main setup:

```bash
vast-run run --project-root . --gpu-model 'RTX 3090' --max-price 0.35 \
  --disk-gb 40 --loading-timeout-seconds 300 --max-infra-retries 1 \
  --remote-setup-command-file scripts/gpu_setup.sh \
  --command-file scripts/train_pandaset_v2_overfit_job.sh \
  --sync-path outputs/pandaset_v2_overfit
```

The staging root must contain `recovery_cache/train/...` and `val/...`, complete
reference JPEGs, and the tracked code. Reuse the preserved encoded caches; no
44.52 GB archive is needed here. The two 500-step controls use one fixed real
training window, repeated to batch 16. This is an overfit diagnostic, not held-out
model-quality evidence. The held-out scene keeps its original partition.

Return `gate.json`, both NFE curves, the step-500 contact sheet, codec/copy baselines
and the 32-frame closed-loop GIF. Mean-Flow is graded at 1 NFE (>=22 dB); the
boundary control at 4 NFE, with its unseen-interval limitation clearly recorded.
Allow 0.5 dB step and peak-regression tolerance; derivative median must be in
(0,3], maximum <=10. Visual review must show recognizable crisp geometry.
See the versioned gate policy in [MEANFLOW_V2.md](MEANFLOW_V2.md).

A passing numeric report alone does not launch another stage. A separate review
JSON records `visual_review_passed: true` and `report_sha256` equal to the SHA-256
of the exact report being reviewed. This binds the decision to the inspected
result; it is not a generic approval for different data or thresholds.

## Stage 1B — front-view base pilot

After Stage 1A numeric and visual checks pass, manually invoke
`scripts/train_pandaset_pilot_v2.sh` in a managed job, with `OVERFIT_GATE` and
`OVERFIT_REVIEW` pointing to the included reviewed reports. The staging cache must
contain **all 76** verified native-front scenes; `validate_pilot_cache` rejects the
tiny overfit bundle. No archive download or style work is necessary.

It runs 5,000 optimizer steps using `configs/pandaset_pilot_v2.json`, with the
20,000-step planned schedule preserved, then emits a front-view numeric report.
Retain the original >=15 dB / >=3 dB improvement / >=25% L1-reduction gate and
add the requested requirement to beat previous-chunk-copy PSNR. Review the
closed-loop GIF separately. A desired 15–20+ dB range is not an already measured
result and does not silently replace the original gate.

## Stage 2A — one dynamic 3DGS flying-camera scene

Run only after the Stage 1A result is reviewed. This single-scene experiment does
not authorize Stage 1B or a batch automatically. Use a training scene such as
040, rather than consume a reserved final-test scene for iterative tuning.

The isolated environment is `.drivestudio-venv`, Python 3.10 / Torch 2.5.1 CUDA
12.4. Main application dependencies and its environment are unchanged. Setup
fetches immutable source revisions from
[`upstream_pins.json`](../tools/chase_3dgs/upstream_pins.json). The wrapper invokes:

1. Checksum-verified native archive preparation, then extraction of only the
   selected scene's six cameras, LiDAR, cuboids and metadata. The extractor never
   unpickles; the pinned external PandaSet devkit reads its native tables only
   inside the isolated tools environment.
2. DriveStudio preprocessing with images/lidar/calib/pose/dynamic_masks/objects.
3. Modern pinned SegFormer sky masks in `sky_masks/NNN_camera.png`, class 10,
   without the obsolete Torch 1.8/MMCV path. That pinned model has a tensor `.bin`
   checkpoint, loaded with `weights_only=True` in the isolated environment.
4. OmniRe fitting with Background (LiDAR + near/far random seeds), RigidNodes,
   DeformableNodes and Sky. `load_smpl=false`; SMPLNodes and its eager import are
   removed, so no gated SMPL assets or human-pose reconstruction are requested.
   Camera-pose optimization is disabled to retain the calibrated poses.
5. Native-time flying-camera rendering with exact cockpit calibration, alpha
   auditing, held-out timestep reconstruction metrics, PNG/GIF inspection outputs,
   and conditional core-contract export.

```bash
# Include the passing Stage 1A gate/review files in the staging bundle.
# In its command wrapper set OVERFIT_GATE and OVERFIT_REVIEW to those files.
# Optional: SEQUENCE=040 THREEDGS_ITERS=15000 DIFIX_CLEANUP=0
vast-run run --project-root . --gpu-model 'RTX 4090' --max-price 0.35 \
  --disk-gb 120 --template-hash '' \
  --docker-image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  --remote-setup-command-file scripts/gpu_setup.sh \
  --command-file scripts/chase_3dgs_smoke_1scene.sh \
  --sync-path outputs/chase_teacher
```

The CUDA development image is intentional: the adapter needs `nvcc`, a C++
compiler and GL/EGL headers for its CUDA extensions. It checks the CUDA toolkit
before the large transfer; setup installs the native build headers on an
ephemeral root Linux container. A runtime-only image is insufficient.

An RTX 3090 may be used under the same cap if a suitable 4090 is unavailable;
performance must then be measured. Build time, data transfer, fit time and prices
are not the screenshot's promised 5–10 minutes or <$0.10. The preceding real full
archive transfer took approximately 80 minutes. That scratch archive was deleted
with its instance; encoded camera caches contain **no LiDAR or cuboid tables**.
The one-scene wrapper may therefore need a fresh **remote-only** 44.52 GB download.
It never downloads that archive to the Mac. Transfer and fitting each have explicit
two-hour bounds. No persistent remote archive store is currently configured.

### Calibration correction from upstream inspection

At the pinned DriveStudio revision, PandaSet `extrinsics/NNN_0.txt` already stores
world-from-camera. Its loader uses:

```text
c2w_front = inverse(ego_pose[0]) @ world_camera_pose[index]
```

It does **not** multiply per-frame ego pose by the extrinsics again. The adapter
asserts agreement with the loaded front camera pose, then calls the repository's
existing `chase_calibration`: back 6.8 m, height 3.8 m, pitch 19°, horizontal FOV
75°, 320x192. The projection test checks a point 15 m ahead and other lateral
points against `geometry.project_waypoints` under translated/rotated poses.
Renders use native frame times and the dataset's own normalized dynamic time, not
an interpolated generic novel-view trajectory. The blue inspection markers are
known local ground-plane calibration points, **not** predicted waypoints or a
measurement of lane alignment.

Upstream references: [PandaSet preprocessing](https://github.com/ziyc/drivestudio/blob/e59bda4fa681f829dbb1d65f0de582b0f633c450/datasets/pandaset/pandaset_preprocess.py),
[pose loader](https://github.com/ziyc/drivestudio/blob/e59bda4fa681f829dbb1d65f0de582b0f633c450/datasets/pandaset/pandaset_sourceloader.py),
[installation/dependencies](https://github.com/ziyc/drivestudio/tree/e59bda4fa681f829dbb1d65f0de582b0f633c450).

### Numeric export and visual gate

Hold out every tenth native timestep from all six image-training views. This is
within-scene reconstruction validation; source scene train/val/test identities
for the eventual DiT remain fixed. Export an importer-visible manifest only when
mean held-out camera PSNR >=22 dB and minimum per-frame chase alpha coverage >=0.85.
Coverage means raw Gaussian opacity >=0.5; sky lighting is not counted as geometry.
This strict condition may fail when sky occupies a large area; any change requires
a separately recorded protocol amendment, not a hidden mask adjustment.

Write to `outputs/chase_teacher/pandaset/040/`:

- `inspection.png`: original front dashcam / raw dynamic flying view / optional
  cleanup, with clearly labeled calibration markers.
- `chase_raw.gif`, optional `chase_difix.gif`, `quality.json`, and `calibration.json`.
- `rgb/*.png`, `alpha/*.png`, optional `rgb_clean/*.png`.
- `manifest.json` only on numeric pass: elevated_chase, exact timestamps and source
  ID, camera calibration, frame list and per-frame hashes.

Review moving cars, distant buildings, lane geometry and lower-center ego-region
coverage. A numeric pass is not a claim of crisp geometry. On failure preserve
inspection artifacts and withhold/invalidate the manifest. On pass, the wrapper
also exercises real core `target_frame`/`cache_manifest(... target='teacher')`
ingestion. The legacy `prepare_chase_teacher.py` remains an experimental static
camera-only baseline and is not presented as the dynamic solution.

Optional `DIFIX_CLEANUP=1` adds a pinned official [Difix3D pipeline](https://github.com/nv-tlabs/Difix3D/tree/c76edc595586e16732c91ddee82f3a6d83a8a9cc)
and [`nvidia/difix_ref`](https://huggingface.co/nvidia/difix_ref). It uses raw render
plus same-time real-front reference for one cleanup step, retaining both versions.
The model card limits the model to research/noncommercial use. Cleanup can invent
content and does not establish geometric truth or temporal consistency. Raw
coverage/held-out reconstruction gates remain in force. No cleanup is run by
default, and its GPU compatibility/quality has not yet been validated here.

## Stage 2B/2C — separately reviewed batch, cache and training jobs

Only after front-pilot and one-scene teacher numeric/visual review, use
`scripts/chase_3dgs_batch_and_train.sh render`, then `cache`, then `train` as
**separate managed commands**. Each invocation returns through the runner for
artifact synchronization before the next stage. Require `FRONT_GATE`,
`FRONT_REVIEW`, `TEACHER_GATE`, `TEACHER_REVIEW` and `FRONT_CHECKPOINT` explicitly.
The script never defaults to running all stages. Batch uses the exact frozen
60/8/8 selection; each scene must independently pass numeric export gates.

Render requires the verified archive and isolated setup on remote storage. Batch
extracts/fits one scene at a time and removes only its owned native/processed/fit
scratch after exporting. Completed exports are hash-checked for reuse. Cache then
extracts only camera members for the main importer and consumes the elevated
teacher contract. Train uses Stage 1B best weights through `--init-weights`, with
fresh optimizer/counters, and initially stops at 5,000 steps for review.

For reuse within one lease, use the runner's documented bounded debug campaign and
`reconnect --skip-upload` for each subsequent stage, synchronize after each command,
and explicitly destroy the instance on completion; retain its idle watchdog.
Otherwise provide explicit persistent storage and include returned inputs in the
next clean bundle. A fresh ephemeral lease does not retain the previous archive.
Do not assume six-hour batch-plus-training timing fits the runner's six-hour
command bound: split work and measure throughput first.

## Stage 3 — future generative appearance teachers only

External object swaps or video stylization produce aligned RGB PNGs with base-cache
fingerprints; import via `prepare_style_cache.py import-teacher`, then train adapters
against the useful flying-camera base. This phase is documentation only now. No
OpenCV style caching, LoRA training, object swaps or generative teacher creation is
launched by Stage 1 or Stage 2. Merged adapters avoid extra adapter operations in
the inference graph; merge latency, size, visual quality and throughput still need
measurement. No zero-latency or successful styled-output claim is made.
