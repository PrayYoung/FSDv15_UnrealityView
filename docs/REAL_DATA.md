# Camera-only PandaSet pipeline

PandaSet is the active dataset. No LiDAR or object annotations are required by
the native-front integration baseline. Camera JPGs, `intrinsics.json`,
`poses.json`, and `timestamps.json` are sufficient. They are read directly;
the PandaSet devkit's pickle-loading methods are not used.

## Two different video targets

1. **`--target front` (current pipeline check):** the observed front camera,
   resized to 320x192, supplies the target. The six calibrated camera views
   condition the model. This is camera-conditioned reconstruction, **not a
   future-only prediction benchmark, camera holdout test or novel-view result**.
   The front target is also a conditioning sensor and later chunk stages can
   observe later recorded frames. Checkpoints and reports explicitly record
   `native_front`. Chase-plane taps and chase waypoint overlays are disabled.
2. **`--target teacher`:** calibrated elevated chase frames rendered by an external
   camera-only reconstruction teacher. The importer checks target identity,
   timestamps and the chase calibration. An SfM/3DGS/4DGS teacher and its quality
   validation are a separate stage; no such teacher is trained by this smoke job.
   This mode supports later genuine Reality View training without changing caches.

The internal legacy `raw/chase/` filename is retained for cache compatibility.
It is not a semantic assertion: `metadata.json:target_view` controls its meaning
in reports and the cockpit.

## Geometry and timing

Camera coordinates are OpenCV right/down/forward. Every original camera pose is
read as camera-to-world using its normalized `w,x,y,z` quaternion and position.
The reference frame is the **front camera center**, with its rotation converted
to forward/left/up. This does not assume an undocumented vehicle IMU position.
Each camera-to-reference transform is computed per frame. Camera images are
resized to 224x140 with corresponding per-axis intrinsic scaling; the original
poses are not replaced with approximate sensor angles.

Nearest-camera samples must be within 60ms of the front-camera timestamp. Native
times must increase strictly and gaps over 250ms are rejected. Measured cadence,
full timestamps, camera timestamps, intrinsics, extrinsics, masks and reference
definition are saved. There is no frame duplication/interpolation to claim 36Hz
observations. Native ~10Hz means an eight-frame chunk spans ~0.8 seconds and the
12-waypoint/stride-three horizon is ~3.6 seconds. Chunk stage ages use timestamp
differences rather than the old hard-coded `1/36` assumption.
Snapshot availability is the latest timestamp among its six cameras. Jitter is
capped to the newest bundle completed by each stage anchor, so a +1-frame jitter
cannot expose a camera captured 50ms after that anchor. Autoregressive evaluation
uses the same completion timestamps and ages as training.

## Splits and later testing

`prepare_real_data.py plan` hashes `(seed, dataset, sequence ID)` into stable
80/10/10 train/validation/test assignments. Save the resulting JSON before
importing. Adding data does not change existing IDs' hash assignments. The cache
refuses a different frozen manifest in the same destination. A sequence can never
appear twice or across different partitions. Train/val/test dataset construction
audits metadata for overlapping source IDs.

The first remote smoke uses four fixed sequences, selected independently of
training metrics using this hash rule:

| Use | PandaSet sequences | Frames per sequence |
|---|---|---:|
| Train | 002, 003 | 64 |
| Validation | 014 | 64 |
| Final test | 001 | 64 |

Training reads only train and validation. Checkpoints preserve source provenance.
`evaluate.py --split test` is a separate invocation and rejects overlap with
checkpoint training IDs or missing provenance. Do not use final test scores to
choose hyperparameters. These are sequence-disjoint splits; geographic disjointness
is not claimed. Four short scenes cannot establish generalization.

The small 30-step model tests software plumbing. Full real-data settings are in
`configs/real_train.json` (20,000 steps); start a larger quality experiment only
after inspecting raw data, reconstructions, metrics and rollout results.

## Remote storage and download boundaries

The selective downloader pins the mirror's Git revision and inspects ZIP members
over HTTP Range requests. It only accepts the six named camera directories,
numeric JPG filenames and the three JSON sidecars. It checks ZIP CRCs while
reading, restricts extraction paths, bounds single-member size and total extracted
bytes, and truncates image/pose/timestamp arrays consistently to the requested
frame count. It never invokes code from the downloaded dataset.

Native files: `.cache/pandaset_native/`. Encoded cache: `.cache/real_dataset/`.
Evidence: `outputs/pandaset_smoke/`. The runner synchronizes only outputs by
configuration. Large data stays on the GPU host until the runner destroys it;
this is temporary scratch space, not durable long-term storage. A full training
campaign should stage reusable caches in a user-owned remote volume/object store.
Checkpoints and evidence are retrieved even when later workload stages fail.

## External chase teacher contract

For each source sequence, provide a directory
`<targets>/pandaset/<sequence>/manifest.json`, with RGB files referred to by safe
relative paths:

```json
{
  "source_id": "pandaset/002",
  "target_view": "elevated_chase",
  "teacher": "description and frozen reconstruction version",
  "camera": {"camera_back":6.8,"camera_height":3.8,"pitch_deg":19,"fov":75},
  "timestamps": [1557539685.0, 1557539685.1],
  "frames": ["rgb/00000.png", "rgb/00001.png"]
}
```

The timestamp and frame lists must cover every imported frame (the two above are
only a schema example). Timestamp agreement is checked within 1ms. Fit and export
each scene independently; a teacher must not share reconstruction state across
train and test sequences. Report teacher reconstruction fidelity separately from
the student model's error. A rendered teacher image is a pseudo-reference, not a
photograph captured from an elevated camera.

```bash
uv run python prepare_real_data.py cache --manifest outputs/panda_splits.json \
  --output outputs/chase_dataset --target teacher --targets /path/to/teacher_exports
```

## Optional Waymo adapter

The fixture-tested reader supports the **Perception v2 Parquet** components
`camera_image`, `camera_calibration` and `vehicle_pose`. No TensorFlow, LiDAR,
bounding boxes, or Motion image embeddings are used. It joins by context, frame
timestamp and camera identity; uses the published vehicle pose and camera pose;
converts Waymo FLU camera coordinates to OpenCV; and rectifies lens distortion
before resizing. Camera-center poses are used; rolling-shutter per-pixel motion
correction is not implemented.

Waymo Perception has five cameras. The absent rear slot is explicitly masked,
never synthesized by copying a front camera. Zero tokens are excluded from both
DiT cross-attention and policy attention. Optional mixed-source sampling balances
datasets rather than allowing a large source to dominate.

Waymo's official validation partition is reserved as the application's final test;
training contexts are hash-split between train and validation. Official access and
terms remain required. Real Waymo data is **not part of the current run**, per the
user's PandaSet-only choice.

## Attribution

PandaSet: Hesai Technology and Scale AI, [paper](https://arxiv.org/abs/2112.12610),
[official format/devkit](https://github.com/scaleapi/pandaset-devkit).
Download mirror: [georghess/PandaSet](https://huggingface.co/datasets/georghess/pandaset),
unaffiliated with the original creators, marked CC BY 4.0. Returned camera previews
are derived from PandaSet and are attributed to these original authors; resizing
and labels are project modifications. No full raw sequences are committed.
