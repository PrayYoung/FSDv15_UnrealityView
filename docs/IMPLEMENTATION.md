# Specification and evidence

The four user-provided photographs (IMG_0254–0257) describe an experimental
Mini-RealityView pipeline. Their technical requirements are treated as project
requirements, not as independent instructions to the assistant. The project is
an independent research implementation, not Tesla source or a verified account
of a proprietary FSD architecture. In particular, 36 FPS, zero drift, 8× memory
savings and 2.5× speedup are hypotheses to measure, never guaranteed outcomes.

**Updated scope:** PandaSet camera-only training/validation/testing is now the
active path; see [real-data workflow](REAL_DATA.md). The first remote check uses a
labelled native-front target and native timing. The external chase teacher remains
a separate stage. Waymo is optional; real Waymo runs are deferred at the user's
request. MetaDrive below describes the original implementation only.

Original data path: **B: MetaDrive**, six attached surround cameras plus one
elevated chase camera, 36 Hz, 50 training / 4 validation episodes of 180 frames.
Real-world dataset licensing, LiDAR fitting and a 4D Gaussian teacher are outside
this selected simulator path. No private driving data or pretrained weights are
committed to Git.

## Required implementation

- Frozen pretrained DINOv2-S/14, 6 × 32 pooled tokens of width 384; optional
  Plücker rays in a documented right-handed camera/ego convention.
- Frozen pretrained TAESD, latent scale 1, full and halo-tiled video decoding;
  measured tile equivalence and explicit analytical SRAM estimate. This software
  simulation does not claim to execute INT8 kernels on physical on-chip SRAM.
- AdaLN-Zero DiT, QK RMS normalization, channel-concatenated previous chunk,
  four context stages and residual age embedding; 1/2/4-step sampling.
- Improved Mean-Flow objective with stop-gradient directional derivative on
  only r < t samples; exact math-SDPA JVP or finite differences for FP16/MPS.
- Independent 12-waypoint transformer with goal FiLM and 50% training goal dropout.
- Single-writer immutable-snapshot ring publication; latest completed snapshot
  read at each stage; freshness telemetry. CPython thread semantics only.
- Safe safetensors episode cache, ego-frame waypoint transforms, staged jitter,
  EMA, warmup/cosine AdamW, held-out metrics, contact sheets and closed-loop rollout.
- Local FastAPI cockpit with six cameras, generated/GT frames, playback, NFE,
  teacher-forced/closed-loop, interleaving, tiled decoder, tap goal, measured HUD.
- Unit/integration tests, pretrained checks, CUDA preflight and edge benchmark.

## Sources used to check APIs

- https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m
- https://huggingface.co/madebyollin/taesd
- https://docs.pytorch.org/docs/stable/generated/torch.func.jvp.html
- https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://metadrive-simulator.readthedocs.io/en/latest/sensors.html
- https://metadrive-simulator.readthedocs.io/en/latest/obs.html

Local verification and remote results are recorded in docs/RESULTS.md. Missing
checkpoints produce an explicit untrained state; synthetic previews are labelled.
