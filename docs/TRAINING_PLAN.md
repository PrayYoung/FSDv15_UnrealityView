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
