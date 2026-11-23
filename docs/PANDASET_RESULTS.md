# PandaSet pipeline verification

The current experiment is a camera-only, native-front integration baseline, not
an elevated novel-view experiment. No real-data quality scores are claimed until
the corrected bounded GPU job terminates.

Local source checks: **23 passed, 1 pretrained check skipped**.
Native PandaSet and Waymo format tests use synthetic fixtures and do not establish
successful operation on real Waymo data. The live remote experiment uses real
PandaSet cameras only, with train 002/003, validation 014 and final test 001.

The earlier insufficient-credit failures in `docs/RESULTS.md` are historical; the
user has since refilled GPU credit.

## First CUDA attempt

Run `vast-20261122T234907Z-33087-4bc895`: **WORKLOAD_FAILED**, RTX 3090, Argentina,
$0.207/hour; runner-reported cost estimate $0.0389. The first infrastructure host
timed out loading and the runner's second host booted successfully.

CUDA 12.4 / PyTorch 2.5.1 matmul and autocast/JVP/backward passed. The suite then
reported 22 passed and one failed: actual pretrained TAESD full-vs-tiled error
reached **0.005329**. No PandaSet images were downloaded and no training began.
The runner retrieved preflight/test logs and confirmed **instance_destroyed=true**.

The CUDA decoder now explicitly disables cuDNN TF32 for FP32 convolutions, so
different tile shapes cannot silently use lower-precision algorithms while
claiming FP32 equivalence. The strict test tolerance is retained. Native bundle
completion timing was also refined and regression-tested locally after the first
snapshot. A new run of the changed source will verify both corrections.
