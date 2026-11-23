# PandaSet base quality pilot

First run: 60 training, 8 validation and 8 reserved test scenes; 80 native frames
each. Camera-only native-front reconstruction, with no elevated-view or future-only
prediction claim. Frozen hash splits and seed 42. No test evaluation during tuning.

Use `configs/pandaset_pilot.json`: full 512×12 per-block cross-attention DiT,
batch 16 / accumulation 1, a 20,000-step schedule stopped safely at step 5,000.
Validation every 500 steps covers all 40 windows from the eight held-out scenes.
Record step-0 validation with the same noise as later evaluations. Inspect step
2,500 and 5,000 contact sheets and 32-frame closed-loop rollouts before continuing.

Predeclared pilot quality criteria: validation 1-NFE PSNR at least 15 dB and 3 dB
above baseline; pixel L1 at least 25% lower; recognizable road geometry on manual
review. These are pilot decision criteria, not promised results. Check waypoint
error and rollout drift separately. An independent review pass must examine
confounding, overclaiming and any protocol changes before accepting the result.
If quality is still improving and visual review passes, resume toward 15k–20k.
Only a useful selected base qualifies for 1,000-step Comic/Cyberpunk adapters and
later review of one held-out real-data GIF. A poor result is analyzed, not hidden.

The job performs CUDA preflight, selective native-camera fetch, caching and training
under one managed Vast lease. Workload has a three-hour timeout; hourly offer
ceiling is $0.35, with existing global safeguards retained. Checkpoints, restart
state, internal diagnostics and the encoded/resized cache are retrieved at the
end, including on workload failure where the host remains accessible. The native
full-resolution cameras stay on remote scratch. No runtime credentials or personal
cloud metadata are committed. Runtime state belongs in ignored `.private/`.

Checkpoint-best metadata includes target views. No decoder optimization or new
3DGS teacher changes are bundled into this training pilot.

## Data-validation retry

The first lease fetched all 76 scenes but stopped during caching before any
optimizer step: scene 052 contained duplicate rear-camera timestamps. A CPU-only
preflight now checks all 103 scenes before a retry. At the pinned mirror revision,
98 pass: 77 train, 10 validation and 11 test candidates. Invalid scenes 052, 054,
058, 062 and 139 are excluded as whole sequences, without changing clocks or hash
partitions. The first 60/8/8 valid scenes are frozen in
`configs/pandaset_validated_pilot.json`; quality thresholds and model/schedule stay
unchanged. This selection correction precedes all learning, not metric inspection.

Thirty-three complete native-front caches from the failed lease passed revision,
source/split, target configuration, tensor and RGB-completeness checks. The retry
uploads a clean code/cache bundle and reuses them unchanged; only 43 native scenes
need fetching with four workers. Cache provenance is rebased to the new frozen
manifest with the original provenance retained. Caching can validate/reuse recovered
episodes without reopening their native source directories.

The retry uses a four-hour workload timeout, adding the requested buffer, with
the unchanged $0.35/hour ceiling. The managed runner owns retrieval and destruction.
No periodic training-log polling is required; identity and recovery paths are saved
privately. No claim is made that metadata preflight detects every possible image
or hardware failure.

## Bulk archive restart

The four-worker range retry hit HTTP 429 after 18 native scenes, before any
optimizer step. The next lease uses 120 GB disk and downloads the pinned 44.52 GB
ZIP once, with the modern Hub/Xet SDK isolated from the training environment.
The source revision, exact byte count and SHA-256 are frozen in
`configs/pandaset_archive.json`. The SDK resumes incomplete transfers and handles
rate limits; bounded outer retries cover transient HTTP and transport failures.
Checksum, space and permanent-access errors stop the job rather than retry blindly.

Only the selected camera JPEG/JSON members are extracted locally on the GPU host;
LiDAR and pickle files are never opened. All 33 verified encoded caches are reused,
leaving 43 scenes to encode. Full-archive transfer has a two-hour bound within the
four-hour workload bound; training still stops at 5,000 steps with the same frozen
quality criteria. The hourly ceiling remains $0.35. Actual download throughput and
training completion time remain unmeasured for this transport.

The archive and native sources are remote scratch, excluded from artifact return.
They do not survive automatic instance destruction; no persistent object store is
configured. Encoded caches, training checkpoints, metrics and the archive integrity
record are returned by the managed runner. No periodic agent log polling is used.
