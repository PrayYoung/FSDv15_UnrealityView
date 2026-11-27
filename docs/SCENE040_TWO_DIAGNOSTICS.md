# Authorized scene040 diagnostics: refit and explicit projection

The user approved Prompts1 and2 from IMG0322–0324. Each gets **one paid attempt,
$0.60 total cap, $0.35/hour maximum**, zero automatic infrastructure retries,
the global Vast lifecycle owner, verified artifact retrieval and destruction.
Prompt3 (batch teacher generation / Hybrid P2.5 training) remains on hold.
Existing raw3DGS, Difix and PixArt reports are frozen and must remain unchanged.

Both probes use **68 frames, indices12–79**, and poseA6.8m/3.8m/19degrees versus
poseB5.5m/3.0m/17degrees, horizontal FOV75degrees. The proposed pre-start,
random-splat and canopy explanations are hypotheses, not established root causes.
Pixel edit L1 alone cannot establish that resolution caused poor cleanup.

## Prompt1: one controlled diagnostic refit

Fit only scene040 with the pinned dynamic OmniRe pipeline, six cameras and800k
LiDAR initialization points. Overlay near_randoms10000 (previously100000), leave
far_randoms100000 unchanged, and configure20000iterations (the upstream inclusive
loop can record20001updates). Render both poses from the **same checkpoint** at
640x384; this is the screenshot's permitted lower-cost higher-resolution option,
not the exact1024x576 resolution documented by the [Difix model card](https://huggingface.co/nvidia/difix_ref).

Render the original held-out native split and report mean PSNR and sample count;
the requested numeric gate is **>=26dB**. Save raw alpha coverage as a descriptive
proxy, not geometry truth. Use full native front JPEGs resized directly to640x384
as Difix references, never upscaled320x192 cache images. Apply the same pinned
Difix model/code/timestep199/guidance0 as the completed standalone baseline.

Return a five-column comparison at12/25/40/55/70 (front, Araw, Aclean, Braw,
Bclean), one68-frame poseB-clean GIF, exact calibration, metrics and stage times.
Keep paired native-size examples at25/40, and small320x192/512x320 previews. The
512x320 preview is letterboxed to preserve aspect ratio. Do not export a passing
training-teacher manifest or auto-launch batches even if numeric coverage passes.
Publication target: `docs/results/chase-3dgs-040-v2-diagnostic/`.

Success requires coherent cars/road/buildings, fewer real floaters and acceptable
motion on visual review, plus native PSNR>=26dB. Ambiguity withholds approval.
The changed iterations, density, resolution and frame selection are bundled;
this single run cannot identify which change caused a difference.

## Prompt2: frozen six-camera explicit projection

No training. Use the official [Depth Anything V2 metric Small outdoor model](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)
fine-tuned on Virtual KITTI, pinned source and checkpoint SHA256 in
`configs/depth_metric_pins.json`. Model weights are downloaded only remotely.

Infer all six calibrated cameras at each native target timestamp. Use complete
world-from-camera6DoF poses and axial camera-z metric depth; there is no ground
plane. Unproject to world space and nearest-depth z-buffer into each chase pose
at640x384 and512x320. Exact-depth ties resolve by input order; no hole dilation,
texture blending, flat-ground assumption or invented geometry is added.

Two depth variants use the **same frozen predictions**:

1. Camera-only: approximate robust overlapping-view log-scale constraints with
   the front metric prediction as fixed gauge and a small identity prior for
   disconnected cameras. No LiDAR enters this variant. Occlusion, translation
   and bad depth can bias alignment; report all scales and matched sample counts.
2. LiDAR-anchored comparison: nearest synchronized sparse world-space LiDAR is
   z-buffered into each camera, then median metric/predicted ratios set per-camera
   scale. Fewer than30valid anchors falls back to1 and is reported. This does not
   eliminate all seam or shape errors and is not a guaranteed accuracy upper bound.

Mask convention: **M=1 means an observed projection**, matching the requested
hole fraction `1-M.mean()`. Hole pixels remain black. Difix receives rough RGB
and native-front reference at640x384, not the mask. It is a generative cleanup
probe, not validated masked inpainting. Use only required timestep199; omit the
optional299 sweep to preserve a single fixed comparison and budget.

Return per-frame hole fractions for both resolutions/poses/sources, GPU-event and
wall latency for six-camera depth, alignment, LiDAR projection and final splats,
per-frame Difix timings, raw/clean temporal RGB L1, two five-frame comparison
sheets, one68-frame camera-only poseB GIF, and frame55 orbit views at−25/0/+25deg.
Orbit rotates both camera position and orientation around the front-reference
vertical axis. The HTML control switches three sampled angles; it is not a
continuous renderer. It supports visual geometry sanity checks, not training.
Publication target: `docs/results/pandaset-explicit3d-040-proof/`.

Success is conditional on actual coverage/visual coherence and measured complete
front-end latency. No20–30fps claim, clean geometry claim or PixArt blur solution
is assumed. Camera I/O, preprocessing, depth, projection and cleanup costs must
remain distinguishable; CUDA event elapsed includes host-induced GPU idle.
No calibrated flying-view ground truth exists here, so no novel-view PSNR is used.

## Transfer, stopping and critique

Each independent rental downloads and SHA256-verifies the existing44.5GB pinned
PandaSet archive remotely, extracts only scene040 through the verified extractor,
validates camera/LiDAR/cuboid frames, then deletes the archive scratch file. A120GB
disk request provides room for build environments, selected native data, models
and fitting scratch. No archive/native data/environment/model checkpoint is
downloaded to this laptop. Only bounded diagnostic media, calibration and compact
metrics/receipts return. Do not publish personal machine paths or rental access data.

CPU checks cover identity projection,6DoF roundtrip, six-camera occlusion, invalid
points, holes, scale recovery/fallback, native calibration/timestamps, orbit and
aspect-preserving previews. A random-weight official metric Small CPU batch
forward validates its API only. Actual CUDA and real-data quality require the
paid attempts. A failed engineering run is not a negative research result, but
does not authorize another paid attempt automatically.

Reviewer checklist before accepting either result: distinguish appearance from
geometry; account for bundled changes, unknown novel-view truth and temporal
blur bias; compare both poses on identical fitted weights/depth; report failed
native gates/anchor fallbacks honestly; withhold Prompt3 until human visual review.
No corporate GCP actions are authorized by the quoted screenshot side question.
