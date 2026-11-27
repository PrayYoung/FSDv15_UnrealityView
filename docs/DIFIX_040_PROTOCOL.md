# Scene040 standalone Difix verification

Authorized by IMG_0311:one paid attempt,0.60USD intended cap (do not raise it to
the separately mentioned1USD hard maximum),0.35USD/hour ceiling,zero automatic
infrastructure retries,owner watchdog/idle-sleep prevention/orphan recovery and
verified destruction. Preserve the completed raw3DGS report untouched. No new
multi-scene batch or generator/style training.

## Exact inputs and runtime

The80raw320x192PNG renders already exist locally. Pair them with the80cached
native-front reference JPEGs at the exact native timestamps from the original
camera recovery cache. References are resized/recompressed cache images,not the
full-resolution native-archive byte stream. Hash every input and calibration.
Upload these small images and code only; do not download/refit the44GB archive.

Invoke the existing cleanup_difix function withDIFIX_CLEANUP=1 semantics:
reference-guided single step,timestep199,guidance0,prompt remove degradation.
Use model nvidia/difix_ref at revision d4830559772a5795c9d136302c5b197d6418d3fb;
code commit c76edc595586e16732c91ddee82f3a6d83a8a9cc,verified pipeline file hash in
configs/difix_runtime_pins.json. The [pinned upstream requirements](https://github.com/nv-tlabs/Difix3D/blob/c76edc595586e16732c91ddee82f3a6d83a8a9cc/requirements.txt)
require the compatible Diffusers0.25.1/Transformers4.38/PEFT0.9/HFHub0.25.1 stack.
Newer Diffusers0.33.1 failed a local custom-UNet import (PositionNet missing).
A CPU random-tiny-UNet reference forward with the actual pinned custom VAE/UNet
code passed before provisioning. This is API evidence,not cleanup-quality proof.
FP32 inference follows the pinned [model/runtime guidance](https://huggingface.co/nvidia/difix_ref).

The first standalone attempt stopped at CUDA preflight with error804 on a
535.161.07 host driver and CUDA12.4 PyTorch. No inference occurred. The prepared
fix keeps Torch2.5.1/torchvision0.20.1 and all model/code pins, uses the official
[CUDA11.8 wheels](https://docs.pytorch.org/get-started/previous-versions/), removes
container CUDA forward-compatibility directories from LD_LIBRARY_PATH, and runs
CUDA preflight immediately after Torch installation, before Difix dependencies
or model downloads. Error804 denotes an unsupported forward-compatibility
configuration ([NVIDIA documentation](https://docs.nvidia.com/deploy/cuda-compatibility/latest/forward-compatibility.html)).
The environment filter is CPU tested; an actual GPU rerun is still required.
The original single paid attempt is consumed; these fixes do not authorize a
second rental automatically.

Use a fixed initial seed42 for the sequence; retain the existing function's
per-frame stochastic behavior instead of silently introducing a different noise
policy. Record model-load,inference,export and total wall times separately.
Download models only on the GPU; return no pretrained weights,venv or cache.

## Outputs and review

Return80candidate cleaned PNGs (no duplicate raw/reference frames),one three-row
inspection at0/39/79,one chase_difix.gif,and compact metrics/receipts. The rows are
native front/raw3DGS/Difix. Add identical blue calibration dots to inspection
copies only. Those dots were intentional overlays in the original inspection;
they are not render floaters,are never sent into Difix,and their being fixed by
construction does not demonstrate faithful generated geometry.

Report the four requested checks explicitly:
1. Actual road/tree color artifacts reduced,distinguishing them from blue overlays.
2. Parked-car roofs and shapes coherent rather than melted or invented.
3. Lane/road/building layout consistent with the raw view and calibration markers,
   with no apparent pose drift. Do not claim overlays themselves prove this.
4. Stability over all80frames; inspect large per-frame edit changes and the GIF.

Color fractions,lower-band gradient correlation,frame-delta L1 and edit-residual
variation are descriptive aids,not geometric ground truth or hard acceptance
thresholds. Difix changes only2D RGB; raw alpha/held-out native PSNR stay context,
not new cleanup metrics. Sharper-looking invented content is not validated3D.
If any check is uncertain,withhold approval. Publish to the new folder
`docs/results/chase-3dgs-040-difix/` and stop. No automatic Difix retraining,
batching,explicit-depth experiment or flying-view/style student training.
