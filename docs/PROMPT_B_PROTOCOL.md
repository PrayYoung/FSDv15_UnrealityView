# Prompt B: one raw dynamic3DGS scene, after Prompt A reporting

The user authorized Prompt B to run independently while Prompt A runs.
No result from Prompt A is required for this one-scene reconstruction check. This is one managed scene040 verification, not a76-scene batch or a
Difix experiment. One paid attempt, <=$1, existing <=$0.35/hour ceiling, owner cost
watchdog and automatic cleanup. Preserve the independent-verification mode:
the old v2 numeric front gate is evidence of pipeline readiness, not a forged
visual approval of an Arm E result.

## Local fixes before rent

- `download()` rejects a directory-valued archive-record destination before disk
  checks or network work. Its unit test asserts the downloader is never called.
- Working artifacts remain in `outputs/chase_teacher`. An EXIT trap preserves
  whitelisted diagnostics, even after a quality-gate failure, in the separate
  `outputs/chase_teacher_return/` DIRECTORY.
- Use that DIRECTORY as the first and sole sync root. The runner creates its first
  sync root as a directory; never put an individual JSON filename first.
- Return only archive/stage timing JSON; scene quality/calibration/target manifest,
  inspection PNG, raw GIF, optional already-created Difix GIF,80RGB target PNGs;
  and small ingestion JSON summaries/manifests. Exclude alpha frames, native
  archive/data, compiled environments, ingestion tensors/RGB and fit checkpoints.
  A collection manifest records every returned file's size/hash and original exit code.
- Record build duration, archive download+verify duration, preprocessing, sky masks,
  fitting, rendering/metrics and ingestion. Missing stages remain missing, not
  invented durations. Pinned native submodule/header and CUDA-kernel checks run
  before the large archive download. Full CPU suite must pass before provisioning.

## Single verification and frozen acceptance

Run `scripts/chase_3dgs_smoke_1scene.sh --independent-verification` with scene040,
15,000 fit iterations and Difix off. Retrieve only the return directory. Held-out
native-camera PSNR>=22dB, minimum whole-frame opacity coverage>=0.50, minimum
rows67:192 coverage>=0.85 remain frozen. Ego footprint coverage is reported.
The exact calibrated target manifest is withheld on numeric failure; preserve
quality JSON/inspection/GIF for review anyway. If numeric export succeeds,
verify_ingestion must finish and produce its explicit success receipt.

Publish under `docs/results/chase-3dgs-040-smoke/`. Include exact available timing,
approximate rental cost, all quality values/ingestion status and an honest visual
review of floaters, sky holes, distant geometry and ego-car gap. A build pass is
not reconstruction success, and numeric success is not visual certification.
Do not automatically retry a poor scientific result or spend for a second rental.
Stop after reporting. User review decides whether cleanup/Difix or batching is
warranted; no automatic76-scene batch, generative flying-view or style training.

## RunPod compiler-discovery correction

The first RunPod attempt passed CUDA compute but failed before archive download
because the native-build shell could not find nvcc. Public image metadata lists
a CUDA12.4 development toolkit and compiler paths; the terminated Pod cannot be
inspected retrospectively, so PATH reset is supported rather than proven there.
Source `tools/chase_3dgs/cuda_env.sh` in each build/workload shell to discover a
CUDA12 nvcc plus runtime headers, export CUDA_HOME/CUDA_PATH/CUDACXX and PATH,
and reject runtime-only or incompatible installations. A native object compile
probe runs after host build tools are installed and before native package builds
or dataset download. Local tests emulate a hidden compiler and verify child-shell
inheritance and runtime-only rejection; actual CUDA compilation still requires
the submitted GPU preflight. The user's new resubmission authorization retains
the$1 limit and minimal-return/automatic-cleanup policies.

## Ninja-path retry and optimizer-start handoff

The CUDA-path correction reached successful native package builds, but nvdiffrast
runtime JIT could not find Ninja. Native packages are installed in the isolated
`.drivestudio-venv`; add that interpreter's bin directory to PATH in the setup
shell, workload shell and standalone Python preflight/render entry points. Verify
the executable before native builds. Keep CUDA/toolchain and scientific settings
unchanged.

The user authorized another retry and requested monitoring only until fitting
actually starts. An opt-in progress hook runs immediately after the pinned
`trainer.backward(loss_dict)` call, which includes its optimizer step. It writes
an atomic small JSON receipt after updates1,2,10 and every100 thereafter, only
when UNREALITY_PROGRESS_FILE is set. It does not change losses, optimizer state,
random draws, camera processing or quality thresholds. The owner periodically
retrieves the minimal return directory; confirm at least10 completed updates
with finite losses before handing off unattended. This establishes startup,
not eventual completion or quality. Stop monitoring after that receipt.

Runtime tooling is checked with the selected interpreter's bin directory rather
than resolving its python symlink to the system installation. This ensures Ninja
is found both during package builds and nvdiffrast's lazy runtime compilation.
The progress hook is matched against the pinned upstream source and tested for
idempotence; it records only after optimizer work, with finite scalar losses.
The retry's artifact owner syncs the small receipt periodically. Monitoring may
stop after the receipt confirms ten completed optimizer updates, per the user's
instruction; automatic result retrieval and cleanup continue afterward.

## Native-table extraction retry

The verified mirror stores scene040 LiDAR and cuboids as `.pkl.gz`; the pinned
SDK discovers only `.pkl`. Preserve the archive and decompress selected native
tables into SDK-compatible filenames without deserializing them in the
extractor. Enforce expanded per-file/scene byte limits, disk headroom, and
destination uniqueness. Before preprocessing, require all80 frames for each
of six cameras, LiDAR and cuboids, plus camera/LiDAR pose and timestamp arrays.

Local verification used the real pinned archive directory and one real frame:
the pinned SDK with pandas2.2.3 loaded179370 LiDAR points and106 cuboids after
decompression. Temporary native samples were removed. This validates table
discovery and decoding, not full preprocessing, GPU fitting or image quality.
Synthetic tests also cover missing frames, expansion limits and conflicting
compressed/uncompressed destinations. Scene-ID normalization remains enabled.
