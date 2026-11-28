# Two-timestamp explicit projection and gated Hybrid P2.5 plan

The user approved the plan in IMG0325–0327. Execute Step1 now; preserve the
specified quality gates before later work. No corporate GCP action is authorized
by the quoted screenshot side question.

## Step1: one scene040 two-timestamp / FP16 diagnostic

Use the stricter **$0.25 headline cap**, one paid attempt, zero automatic retries,
and at most$0.35/hour. The screenshot's body permits a larger cap; it is not used.
Global Vast owns provisioning, transfer verification, budget watchdog and exact
instance destruction. Return only small diagnostic media/calibration/metrics.

Compare frozen metric Small depth with GPU RGB preprocessing and CUDA FP16
autocast at input sizes518and392, using the same pinned source/checkpoint as the
completed explicit3D proof. Preprocessing matches its lower-bound/multiple-of14
size policy:640x384inputs become868x518or658x392. PyTorch bicubic replaces CPU
OpenCV resizing; image normalization remains RGB/ImageNet. No per-camera
`image2tensor` loop, LiDAR scale anchors, Difix model or previous generated latent.

Use only poseB (back5.5m,height3.0m,pitch17deg,FOV75deg), targets12–79. For each
variant, pre-roll actual camera frames7–11,then retain a bounded rolling cache.
At targett, estimate/alignment depth once for its six images;reuse cached actual
camera RGB/depth/calibration fromt−5. Unproject both timestamp groups through their
own world-from-camera6DoF transforms into **the same current chase camera**.
Validate the observed timestamp gap is approximately0.5seconds. No future frames
or generated-history latents are used.

Composite strictly with current coverage priority: current RGB wins every covered
pixel, historical RGB fills current holes, and M is the boolean union. Pixels
outside both masks stay black. Assert exact identity on current-covered pixels.
This is **not** a general zero-ghosting guarantee: stale moving objects can remain
where current coverage is absent, and temporal scale/depth error can create seams.

Report per-frame single/2timestamp hole fractions over the full image and the
original lower-band proxy rows67:192 at320x192 (rows134:384 at640x384). It is an
image band, not a semantic road mask. The advertised8–12%holes and25–40ms depth
are hypotheses, not acceptance facts. Retain all measured outcomes honestly.

Measure direct start/end CUDA-event elapsed and processing wall latency for GPU
preprocessing, current six-camera depth, camera-only overlap alignment, current
projection, past projection and compositing. Report the **measured total** median/
p95 and FPS, not a sum of independently measured medians. GPU-resident timing
excludes JPEG decode/resizing, upload, model load, five-frame pre-roll and diagnostic
export; report I/O-inclusive frame wall latency/FPS separately. The steady-state
cache allows one new six-camera depth pass per target; startup work is recorded.
Model AMP batch/memory preflight runs before downloading the native archive.

Use full native current/past cameras from the SHA256-verified pinned archive on
the remote GPU. Only scene040 is extracted;the archive is then deleted remotely.
The native reader disables LiDAR inputs in this experiment. Existing poseB
OmniRe+Difix teacher samples at12/25/40/55/70 are hash-bound and uploaded as small
comparison images; no teacher refit or additional Difix inference is required.

Publish to `docs/results/pandaset-explicit3d-040-2ts-fast/`: per-variant comparison
panels (front,single rough,two-timestamp rough,single M,two-timestamp M,teacher),
one68-frame `explicit3d_2ts_raw.gif` using the518variant, selected native rough/mask
samples, all measurements and an honest visual review. Retain one additional
small392variant GIF for temporal quality comparison before choosing a depth size.
Preserve every earlier
result report. Stop after reporting;later steps require their quality review.

CPU checks cover exact live priority, union/invalid masks, old-camera world-pose
projection, stale-content limitations, batched RGB normalization/resize dimensions,
scaled road-band metrics and disabling LiDAR. A synthetic real-model GPU preflight
is API/numerical/memory evidence only. Scientific success requires fewer holes
without unacceptable added seams/stale objects,and actual measured runtime;
uncertain visuals withhold the next stage. Different preprocessing,precision and
hardware prevent attributing a historical speed difference solely to FP16.

## Step2: gated12-scene dataset export, aggregate cap$6

**Only after Step1 looks clean on visual review.** Use10diverse scenes from the
existing60-scene training split and2from the existing8-scene validation split.
Never open reserved test scenes. Preserve diversity across hills,intersections,
avenues and night/dusk. No scene list is selected or job submitted yet.

For each scene: fit the verified poseB teacher recipe (near_randoms10000,
800kLiDARinit,20000configurediterations,six cameras), native held-out PSNR>=22dB
(flag<24dB), and lower-band rawalpha>=0.85. Render targets12–79 at640x384,then use
pinned Difix at640x384 with native-front references. Numerical gates do not
replace per-scene visual approval.

Export paired targets and camera-only t/t−5 roughRGB plus binary M at512x320and
320x192, with identical calibrated resizing/padding for every paired field. Save
frozen TAESD target/rough latents and six-camera TAESD latents encoded directly
from full native JPEGs resized to256x160 (32x20latent pixels per camera), not the
old224x140camera cache. Record all poses,intrinsics,timestamps,transforms,hashes and
split identity. No previous generated latent or LiDAR input enters rough features.

Keep four-scene chunks resumable under an **aggregate$6 cap**, not$6per chunk;
at most$0.35/hour. Verify important cache artifacts after each scene/chunk before
deleting native/archive/fitting scratch. Return only compact latents,masks,
metadata and inspection strips (estimated150–300MB,not guaranteed). Publish all
12strips at12/35/55/75 to `docs/results/pandaset-flying-teacher-12scenes/`.
Stop for visual approval of all12before training. Durable sync reduces loss risk;
it cannot guarantee no work is ever lost during a host outage.

## Step3: gated one-attempt Hybrid P2.5 training, cap$1.50

**Only after all12exported scenes pass visual approval.** No training is started
or implementation choice finalized before those gates. Use10train/2val targets,
frames12–79,initial320x192resolution with optional512x320as a separately reviewed
follow-up,never reserved test scenes.

Freeze the PixArt-LCM611M backbone and TAESD. Condition on pixel-aligned
TAESD(roughRGB) and binary latent M, with masked noise mixing or a tested zero-init
adapter, plus six native-source camera tokens (960tokens) and calibrated Plucker
rays. Zero-init role/ray adapters. Target one-NFE,t999,rank32/alpha64LoRA on all
28attn1/ffblocks plus requested output/conditioning parameters. The refiner is
stateless;its explicit front end still uses a fixed real-camera history buffer.
Pixel-aligned initialization is a hypothesis for reducing blur,not a guarantee.

Train4000updates,effectivebatch16,AdamW1.5e−4cosine,150stepwarmup,EMA.992,mild
per-camera±10%jitter and0.10random hole-patch augmentation. Use the clean loss
formula0.5PseudoHuber(c.1)+2Charbonnier(eps.001)+0.8channel-normalized VGG squared
distance;avoid applying the RGBfactor twice. Training RGBclamp[−.25,1.25],VGG
internally[0,1]. No Sobel training term or PatchGAN.

Report PSNR,V2VGG,Sobel evaluation against the generative teacher and rawrough;
fixed/fresh-noise temporal metrics;measured full front-end+refiner+decode latency;
paired panels and one68-frame val GIF,plus a separate crisp ego-car overlay
preview. Do not claim teacher-relative accuracy as real novel-view ground truth,
or assume20+fps before measurement. Publish to
`docs/results/pandaset-hybrid-p25-flying-4000/`,stop after reporting.

## Updated Step2 authorization

After reviewing the partial Step1 result,the user approved IMG0328's
[updated export protocol](FLYING_EXPORT_STEP2.md). It locks392,extends causal
priority tot/t−5/t−10 and adds one-pass neighbor filling. This later request
authorizes the three$2chunks despite the earlier wait-before-export decision.
Step3 still requires all-scene visual approval and is not launched.
