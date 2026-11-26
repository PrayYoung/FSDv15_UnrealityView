# Prompt D v1: original stateless pilot (source de373d0)

Frozen before GPU execution. This is a new bundled architecture/loss experiment,
not a causal ablation of a single component. The synthetic-ablation claims in
the supplied instructions are background supplied by the user, not independently
verified results from this repository.

## Data and architecture

Use only existing recovery_cache:60train/8val scenes,80frames each at native10Hz.
Do not upload, open or evaluate the reserved8-scene test data. Do not download the
44GB archive in this job. Six camera JPEGs at t-1 predict native-front target t,
for t=1..79. Cached target latents and calibration are retained. Camera images
resize224x140 to256x160 and encode with frozen TAESD to4x20x32. No z_prev,
DINOv2, depth model, ground-plane warp or text encoder is present.

[Official PixArt-LCM model](https://huggingface.co/PixArt-alpha/PixArt-LCM-XL-2-1024-MS)
and [TAESD](https://huggingface.co/madebyollin/taesd) revisions are pinned in
[the model pins](../configs/pixart_stateless_pins.json). The frozen transformer
uses28blocks,1152width; rank16/alpha32 LoRA targets attn1 q/k/v/out and both FFN
linear projections. Also train proj_out, adaln_single, scale_shift_table, seven
role embeddings, one4096-channel null caption, and54->256->1152 Fourier-ray MLP.
The final ray linear starts at zero. Full calibrated6DoF camera rays and target
rays use four frequency bands and no flat-ground assumption.

Concatenate240target and6x160camera tokens using the pretrained patch/position
embedding, all28blocks, no attention mask, CUDA Flash SDPA only. Prediction is
one endpoint evaluation:normalized flow t=1 maps to native timestep999;
z_hat=eps-v_pred, using the first4of8output channels. This is the requested
flow-adaptation objective, not a claim that an unmodified LCM scheduler performs
this transformation. Tiny real-Diffusers tests cover token shapes, trainability,
stateless repeatability, gradient checkpointing, and fused-LoRA equivalence.

## Training and budget

2000optimizer updates,effectivebatch16 (8x2),bf16,checkpointed blocks,AdamW
lr1.5e-4,100stepwarmup then cosine,EMA0.992only trainable parameters.
Loss=.50pseudoHuber(c=.1)+1.00Charbonnier(eps=.001)+.80channel-normalizedVGG16.
Pseudo-Huber uses c^2(sqrt(1+(delta/c)^2)-1). VGG distances are mean channel-unit-
normalized feature L1 over relu1_2/relu2_2/relu3_3/relu4_3. RGB losses use[0,1].
Decode differentiably through frozen TAESD on every batch item. No Sobel loss or
discriminator. Gradient norm clipped at1; nonfinite values stop the run.

Cache clean camera latents plus two independent per-camera RGB brightness/
contrast jitter banks(+/-10%) once at startup; sample a jitter bank independently
per camera during training. This bounds repeated encoder work; jitter is sampled
from two cached realizations, not freshly re-encoded every update. Randomly zero
one camera latent with probability.10. Validation uses only the clean bank.

One paid attempt,1USD total intended cap,.35USD/hour hard host ceiling,7200second
workload timeout,owner watchdog,macOS idle-sleep prevention,bounded artifact sync,
automatic destruction. No automatic scientific or paid engineering retry. Exact
completion and observed budget must be checked from owner metadata. Laptop power
loss/lid sleep/provider outages remain lifecycle risks; watchdog configuration
alone is never reported as observed budget compliance.

## Evaluation and stopping

Evaluate every frame1..79in all8validation scenes against raw JPEG GT and measured
TAESD codec ceiling. Report mean per-frame PSNR,VGG distance,Sobel edge L1 for
all6cams,drop_front,swapped_scene_cams,and fresh per-frame noise. Primary inference
uses one fixed eps_0 per validation clip. Each clip has79consecutive frames;
report raw frame-delta L1 and GT-motion-residual L1 for fixed vs fresh noise.
No optical-flow compensation is claimed. Report fused-LoRA model latency and
model+decode latency,with camera encoding excluded and clearly labeled.

Re-evaluate the preserved Arm E best checkpoint on frames8..79 of the same8val
scenes. Arm E receives GT previous8-frame latents,as in its published benchmark;
Pipeline2 has no history input. Report this conditioning advantage. Published
v2=21.8846dB,ArmE=21.5858dB/f0~24,and claimed codec30.25dB are historical references
with different windows; do not represent them as matched measurements.

Return one EMA adapter,JSON receipts/metrics,one8-scene inspection PNG set,and a
fixed-noise diagnostic GIF with GT/codec/ArmE/Pipeline2 labels. No datasets,
preencoded camera banks,backbone/TAESD/VGG weights or optimizer state downloaded.
Publish results under docs/results/pandaset-p2-pixart-stateless-2000/ after the
run. Stop after reporting. No style,flying-view generator,Difix or batch jobs.

## Reviewer pass before accepting a conclusion

Check all8val IDs and causal lag,complete2000updates,finite gradients,codec ceiling,
matched frame ranges,and explicit ArmE GT-history advantage. A conditioning probe
with no quality delta does not demonstrate camera grounding. A lower raw temporal
L1 can reflect blur; assess its PSNR/VGG and GT-motion residual jointly. This
bundle cannot attribute any gain to a single loss/backbone/ray component. An
engineering failure or timeout yields no completed quality conclusion.
