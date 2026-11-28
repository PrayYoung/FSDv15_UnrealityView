# Updated Step2:12-scene paired flying-view dataset export

The user explicitly approved IMG0328 after the partial Step1 result. This
supersedes the earlier wait-before-export decision;it does not approve training.
Generate exactly10train/2val scenes from the existing split,never open or list
reserved test scenes,and stop for visual approval before Step3.

## Frozen selection and execution

Selection is frozen before any fit in `configs/flying_export_selection.json`.
Only existing train/val metadata and frame40previews were inspected. Tags describe
visual sampling,not official semantic ground truth. Scene IDs are never selected
or replaced based on later fit scores.

| Chunk | Train | Val | Intended diversity |
| --- | --- | --- | --- |
| 1 | 040,006,067 | 047 | Hills,day intersections,night avenue |
| 2 | 034,011,084,073 | — | Dusk hill,waterfront,multi-lane,night intersection |
| 3 | 027,024,099 | 074 | Hill,dusk/glare,avenue,night validation |

Each chunk has a **$2maximum rental budget**,aggregate **$6**,and **$0.35/hour
maximum**. The global Vast owner performs each canonical foreground job,CUDA/
transfer preflight,sync and exact destruction. A durable CPU campaign driver waits
for that terminal result,verifies the completed caches and destroyed instance,
publishes the chunk's summary/strips,then submits the next authorized chunk.
It never provisions via raw APIs,manages SSH,or polls instance readiness.
No concurrent chunk leases or automatic paid infrastructure retries are used.

Hard scene gates are held-out native PSNR>=22dB and minimum poseB lower-band raw
alpha>=0.85 across68targets. Scores22–24dB are explicitly flagged for visual
review;the stated hard cutoff is not silently raised to24. A failed hard gate,
runtime/integrity failure,unverified cleanup or budget violation stops later
chunks. No replacement scenes or negative-result retries are automatic.

The driver records each canonical run ID immediately. A file lock prevents two
campaign controllers. Resume reuses validated same-recipe completed scenes and
deducts recorded prior costs from both the chunk and aggregate limits;unknown
cost or an existing run not recorded destroyed prevents a duplicate submission.
It does not guarantee recovery of an unfinished fit after host loss. Four-scene
chunk returns and finalizer recovery reduce loss;no promise of zero lost work.

## Teacher and camera-only student pairing

Each rental downloads and SHA256-verifies the44.5GBpinned archive remotely once.
Extract only the current selected scene once using the verified native extractor;
remove native/processed/fitting scratch after its cache passes validation,and
delete the archive at chunk completion. Request120GBscratch to accommodate the
archive plus build environments,fitting and one scene at a time. No native source
archive or fitting checkpoint returns to the laptop.

For each scene,fit the pinned dynamic OmniRe model with all6cameras,800kLiDAR
initialization,near_randoms10000 (far_randoms100000unchanged),20000configured
iterations (the upstream inclusive loop can record20001updates),and the original
native held-out split. Render **poseB only**,back5.5m,height3.0m,pitch17deg,FOV75,
frames12–79 at640x384. Check native PSNR and raw alpha gates before any paired
feature export. Pinned Difix,timestep199,guidance0,seed42,cleans these renders at
640x384 with full native front images directly resized as references.

Student rough features use **only** current/past cameras,FP16 metric Small at
input392,and batched GPU RGB preprocessing. Pre-roll source frames2–11. At each
targett,infer one new six-camera depth batch,then reuse cached t−5and t−10RGB,
depth and calibration. Project all three groups into the same current full6DoF
poseB,and compose with strict priority **t>t−5>t−10**. The ten-frame pre-roll is
extra startup work;steady state has no additional depth inference versus Step1.
No LiDAR scale anchors or previous generated latent enter student features.

Apply one snapshot-based3x3pass:an uncovered pixel with at least4of8originally
valid neighbors receives their average. Do not iterate or use newly filled
neighbors. Preserve all original covered RGB values exactly. Eligible concave
hole boundaries can move by one pixel;large-hole interiors remain zero. This is
a local threshold rule,not a semantic guarantee about true3Docclusions.

Store **M=pinhole_closed_M** and a separate **observed_M** for the raw3timestamp
union. The former includes interpolated pixels;it must not be described as
entirely real source observation. Runtime checks verify exact current-covered
pixel identity. Historical stale objects and current projection seams remain
possible;neither priority nor averaging proves zero moving-object ghosting.

## Calibrated compact cache

Teacher rendering uses the external normalized world frame. Convert its target
pose back through the first native ego world pose and assert agreement with the
student's native-world poseB. Current native6DoF poses,intrinsics,timestamps and
target poses are stored. Each original native JPEG is decoded once per source
frame/camera,then resized directly to640x384for depth and256x160for camera TAESD;
the old224x140recovery-cache images are used only for selection previews,not export.

Save canonical target and post-close rough PNGs remotely at512x320and320x192.
The512x320pair uses a shared calibrated letterbox (512x307content,top6/bottom7),
with intrinsics updated for actual resize and padding. RGB uses Lanczos;masks use
nearest neighbor and stay binary. No target/rough alignment is silently stretched.

The frozen pinned TAESD encoder yields these FP16 cache fields for68frames:

| Field | Per-frame shape |
| --- | --- |
| target/rough latent512 | 4x40x64 |
| target/rough latent320 | 4x24x40 |
| Six native-source camera latents | 6x4x20x32 |
| M and observed_M | Binary512x320and320x192,bit-packed on width |

Compact gzip/safetensors also retains **lossless WebP paired RGB at both sizes**.
This preserves exact pixel/perceptual-loss and raw-rough evaluation targets for
later training instead of substituting codec-decoded target latents. Every RGB
roundtrip is pixel-checked;no lossy quantization is used to force the original
150–300MBestimate. Actual cache size is measured and reported and may be larger.
Only this critical cache,masks,metadata,receipts and inspection strips return.
No raw source JPEGs,native tables,model weights,optimizer or fit checkpoints are
downloaded. Public Git receives only compact receipts/metrics and strips,not the
dataset cache itself or private rental-access/machine metadata.

Each scene has a manifest binding compressed and uncompressed cache checksums,
metadata/quality/inspection hashes,and the frozen recipe. Loaders verify all
latent shapes/finiteness,frame/split identity,RBGpixel checksums,mask containment,
calibrations and rigid rotations. The dataset format never deserializes pickle.
Before archive download/fitting,actual-weight depth FP16 and TAESD shape/numerical
checks plus a lossless RGB codec check run in the paid rental.

## Reporting and stop

After each chunk,verify its artifact manifest,all completed scene caches,and
instance absence before publishing to
`docs/results/pandaset-flying-teacher-12scenes/`. Each strip uses12/35/55/75:
native front,3timestamp+pinhole rough,closed M,poseB Difix teacher. Report per-stage
hole fractions before/after each timestamp and closing,live-priority checks,
teacher PSNR/alpha,flags,actual artifact size and approximate rental costs.

Numerical passes are not human visual approval,real flying-camera ground truth,
or proof of real-time deployment. All12scenes require visual approval before
Step3. No Hybrid P2.5,style training or GCP action is launched by this campaign.

CPU reviewer checks:strict priority and immutable observed pixels;single-pass
neighbor fill with no large-hole flood;correct letterbox calibration;frozen split
composition;codec/latent/manifest corruption rejection;terminal cleanup checks;
per-chunk and aggregate accounting;public report contains no dataset/private data.

## First chunk and infrastructure recovery

Chunk 1 exported all four scenes and passed every numeric gate in 3 h 56 min, with estimated rental cost $0.9196. Native held-out PSNR was 26.438 (040), 24.844 (006), 30.124 (067), and 25.584 dB (047). Minimum raw lower-band alpha was at least 0.993575. Across the four scenes, mean whole-image holes decreased from 30.87% (single timestamp), to 14.60% (two), 8.38% (three), and 2.54% (after pinhole fill). Filled pixels are inferred colors, not additional observations; rough-view warping and seams remain visible. Native-view PSNR does not establish novel-view geometry correctness.

Chunk 2's first rental timed out during provider startup without running a workload. The runner verified its destruction and estimated cost at $0.0492. The campaign incorrectly required the status label `DESTROYED`, although the runner preserves `INFRA_FAILURE` alongside a destruction timestamp and verified absence. The controller now treats classification and cleanup separately, tests both status cases, and still refuses advance without canonical verified absence. One bounded infrastructure retry excludes the failed host and deducts the failed attempt from the original $2 chunk / $6 aggregate caps. The scientific recipe and completed caches are unchanged.
