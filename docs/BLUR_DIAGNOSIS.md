# Front-view blur diagnosis: stop before retraining

The next generator experiment is inference only on the frozen step-3,000 best
checkpoint. The independent one-scene 3DGS verification is a separate job and is
not modified by this diagnostic. Test scenes are not loaded.

Run `python -m scripts.diagnose_base_blur --checkpoint CHECKPOINT --data CACHE
--reference-metrics STEP3000_METRICS --output OUTPUT --device cuda` through the
managed runner. Preserve the frozen 60/8/8 allocation and causal snapshot selection.
Upload only the best weights/config, the required encoded episodes and target
JPEGs, and code/configuration. Return only the output JSONs and two contact sheets.
No training or new dataset download is performed.

The diagnostic samples the first available window from each of the first 40
training scenes in cache-path order, with jitter disabled. Validation uses the
exact original 40 ordered windows. Noise seeds are 917 + window index, shared
across NFE settings and conditioning controls. Validate original window order and
reproduce the step-3,000 1-NFE PSNR within 0.05 dB before interpreting results.
A mismatch is an engineering/reproducibility issue, not a scientific verdict.

Report matched 1/2/4-NFE pixel PSNR/L1 and latent L1, with f0..f7 profiles;
codec and previous-chunk-copy references; and first/final frames from four distinct
scenes per split. Codec latent L1 is zero by construction, not a quality claim.
Train and validation sampling distributions differ, so a measured gap should not
be attributed solely to overfitting without further matched-motion analysis.

Validation-only 1-NFE controls use identical noise:

- Baseline: actual previous chunk and camera tokens.
- Swapped camera tokens: next selected window from a different scene, with its ages.
- Frozen stage tokens: first observation repeated, with ages increased by actual
  anchor-time differences. This tests stale observations without pretending they
  are fresh or exposing future cameras.
- Zero previous chunk: real camera tokens retained.
- Swapped previous chunk: next selected different-scene window, actual camera tokens.

Never use all-zero camera conditions. Wrong-scene and zero-history controls are
out of distribution: output sensitivity does not uniquely identify a bottleneck.
A weak response to stage freezing could reflect redundant views or scene motion,
not necessarily a broken stage mechanism. A train/validation gap must use comparable
reconstruction metrics; training flow MSE is not validation latent L1.

Stop and report after this diagnostic. Do not launch the proposed finer-grid
5,000-step training yet. If evidence warrants it, change spatial token resolution
alone in a separately frozen experiment. Token count matching the number of latent
patches does not itself establish geometric alignment. Camera rays, cache metadata,
validation and compatibility must be handled explicitly. Hold augmentation, timing
changes, CFG and decoded sharpness losses out of that experiment.

For any future decoded loss, frozen decoder weights must still allow gradients to
predicted latents. The current decode wrapper is inference-only. In the MeanFlow
parameterization, z_t - t*u is the clean endpoint for an average velocity toward
r=0; r=t is an instantaneous-velocity case, not generally a clean endpoint.
