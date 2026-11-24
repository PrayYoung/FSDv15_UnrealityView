# Step 1: finer MiniUVE conditioning, then stop

Change spatial pooling only: 4x8 (32 tokens/camera) to unpooled 10x16
(160/camera, 960 total). Token count equality with the DiT patch count does not
establish spatial alignment: rays follow each camera's own grid. 5x8 is supported
for future bounded tests but is not an automatic fallback in this campaign.

The immutable 60/8/8 allocation, stage_context timing, architecture/parameter
shapes, seed42, MeanFlow objective, policy_weight0, 20k LR schedule stopped at5k,
40-window validation seed917, EMA and numerical gates stay unchanged. Start from
scratch, not the previous checkpoint. No appearance features, augmentation, CFG,
sharpness losses, extra training steps or 3DGS batch. Test scenes may be re-encoded
as specified in the full76 cache, but are never evaluated or used for selection.

## Input contract and parity gate

All supported grids carry per-episode/manifest provenance; a missing grid means
legacy4x8 only for192 tokens. Reject mixed grids and checkpoint/dataset mismatches.
The importer supports `cache --uve-grid H W`. The dedicated reencoder preserves
all tensors except UVE features and links/copies RGB into a distinct directory.
Camera storage order is front_left/front/front_right/rear_left/rear/rear_right;
these last names are storage aliases for physical left/back/right, not rotations.

Before encoding10x16, repeat4x8 encoding on the first train episode from quality90
cached camera JPEGs, using stored calibration/masks. Per available camera, require
relative token RMS error<=0.20 and cosine>=0.98; record all values and stop on
failure. These predeclared tolerances allow lossy JPEG re-encoding, not a claim
of exact parity or proof of correctness from similarity alone. CPU tests also
check slot-order perturbation rejection, geometry and all non-UVE tensor equality.
Both episode metadata and the manifest disclose JPEG reuse and source hashes.
This creates a small additional input discrepancy versus original decoded pixels;
report it alongside any result, even when parity passes.

## Managed command and safeguards

Run `scripts/train_uve_grid_ablation.sh` with the main CUDA setup and only the
verified recovery_cache as input. No native dataset archive download is needed.
A clean bundle includes all referenced requirements/config files, but no local
venv, credentials, personal runtime files or historical checkpoints.

1. Verify all76 original caches; run4x8 parity, then re-encode into a sibling cache.
2. In a separate process run20 optimizer updates with10x16, batch16/accum1.
   Stress full-strength exact JVP (beyond early warmup) to avoid underestimating
   later training memory. Record peak allocated/reserved VRAM and step/sec;
   reserve additional EMA/policy/decoder memory plus GPU headroom.
3. Only on CUDA OOM or insufficient headroom, retry preflight at batch8/accum2,
   preserving effective batch16. Record this as a protocol/resource difference:
   microbatch random draws are not guaranteed bitwise identical to batch16.
4. Use measured training projection with40% allowance for evaluation/checkpoint IO.
   If it exceeds the existing4h training timeout, stop and report before any5x8
   fallback. Keep the runner's <=$0.35/hour and hard cost/time safeguards.
5. Train5,000 steps under the unchanged20k schedule. Preserve full restart state,
   best weights, all metrics and initial/best/final previews; omit only verified
   duplicate weights. Return no dataset or re-encoded cache.
6. Diagnose the validation-best checkpoint at its recorded step. Reproduce its
   ordered40-window metric within0.05dB, then compare the same40train scenes,
   40validation windows, frame profiles and conditioning controls. No test evaluation.

## Predeclared interpretation and report

Compare with baseline best21.884630dB, final21.808995dB, train26.339204dB;
validation f0/f7=23.42/21.01dB, frozen-stage delta~-0.0048dB,
wrong-scene UVE delta~-1.243dB, zero/swapped history14.290/13.987dB.
Primary outcome: change in validation-best1-NFE PSNR; supporting outcomes are pixel
and latent L1, f0..f7 decay, matched train metrics, conditioning deltas and visuals.
A >=0.5dB best-validation gain plus visibly improved detail would support practical
benefit from richer spatial conditioning. Mixed/smaller gains remain inconclusive;
report every metric rather than promote one favorable control. Thresholds in the
existing training quality gate are unchanged.

A poor10x16 result does not prove that DINO features lack appearance: optimization,
training diversity, token routing or temporal handling could also limit benefit.
Subtracting frame-copy PSNR gaps is descriptive, not an identified causal correction
for motion difficulty. Wrong-scene and zero-history controls remain OOD probes.

Publish the resulting comparison under
`docs/results/pandaset-v2-ablation-uve10x16/`, then STOP for review. Do not launch
appearance features, sharpness losses, CFG, extended training or flying-view
training automatically. GPU runtime, parity and quality are unverified until the
managed job returns; CPU tests alone do not establish any of them.
