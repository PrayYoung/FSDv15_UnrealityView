# PixArt v2 / scene040 Difix execution audit

The first two attempts are finished and their exact instances were verified
absent using the global runner. No replacement attempt has been submitted.
These are engineering outcomes, not model-quality findings.

| Attempt | Evidence | Available result | Approximate rental cost |
| --- | --- | --- | --- |
| PixArt v2 | CUDA passed; last saved progress confirms1901 optimizer updates; remote shell exited0 after5541seconds | Durable log and progress receipt only; requested result directory is empty locally | $0.5826 |
| Scene040 Difix | Setup/import passed; CUDA preflight failed with804 | Setup/failure log only; no cleaned RGB | $0.0230 |

Costs are elapsed rental estimates, not verified provider invoices. PixArt's
exit0 is consistent with completion of the2000-update script and evaluation,
but final metrics, loss curve, adapter and images were not retrieved. We cannot
validate training quality or claim a successful scientific result from exit0.

## PixArt recovery limit and fix

A local artifact-store search found only the preserved v1 EMA adapter and curve.
The v2 checkpoint is not in the local return or source staging. Both instances
were destroyed, so the runner cannot reconnect to retrieve their ephemeral
files. No independent persistent checkpoint backup was configured.

The owner's artifact code treated every nonzero path-probe exit as an optional
missing directory and returned an empty synced-path list. This is an established
failure-handling defect. The precise remote reason for this run's failed probe
is unknown because its status/stderr was not persisted. Normal owner completion
then recorded SUCCESS and destroyed the rental. Artifact success was not checked.

The fix retries existing-job retrieval up to three times with bounded probe and
archive timeouts, rejects empty output, checks the new file/SHA256 manifest,
validates files in temporary staging, and records ARTIFACT_FAILED/error text on
exhaustion. Destruction and cost safeguards remain active. It does not guarantee
recovery if the host/network remains unavailable through the retry window.
New job scripts also print compact metric summaries into the durable worker log.

Recovering the same v2 result is not possible with the available artifacts. A
replacement requires the same fresh2000-update training and matched evaluation;
evaluation alone needs the lost adapter. Starting from v1 would change the
frozen experiment. Existing camera caches, Arm E and v1 weights remain available,
so a replacement needs no full PandaSet archive. A tiny return-transfer/GPU
smoke check should precede any replacement training on the same rental.

## Difix fix and remaining verification

Switch only its Torch build from CUDA12.4 to11.8, remove container CUDA
forward-compatibility shim directories from the library search path, and run
GPU preflight before Difix/model downloads. Model, source, diffusion-step and
input pins remain unchanged. CPU tests verify the library-path filtering and
existing cleanup API; successful CUDA and80-frame inference remain unverified.
All80raw renders and references are still available locally.

## Spending boundary

Each original image instruction permitted one paid attempt with zero automatic
infrastructure retries. Both attempts have been consumed. CPU preparation and
runner repair add no rental. A new paid attempt requires renewed authorization;
the proposed per-attempt caps remain1USD for PixArt and0.60USD for Difix, with
0.35USD/hour maximum host price. No automatic batch or additional model training.

The user subsequently authorized restarting both jobs with those unchanged caps.
One replacement per experiment was submitted with zero automatic retries and
the pre-rental CUDA capability filter. Submission alone does not establish
successful GPU preflight, training, artifact retrieval or visual quality.

## Completed replacement outcomes

Both authorized replacements completed, their critical returns were checksum
verified, and their instances were verified absent after destruction. The
[PixArt v2 report](results/pandaset-p2-pixart-stateless-v2-2000/README.md) records
26.21dB validation PSNR and remaining blur/temporal limitations. The
[scene040 Difix report](results/chase-3dgs-040-difix/README.md) records80cleaned
frames and successful CUDA11.8 execution, but rejects the images as a training
teacher because reconstruction distortions persist. The earlier failure evidence
above is retained; it describes the first attempts, not the replacements.
