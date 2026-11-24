# Scene 040 dynamic 3DGS: native build verified, output-path failure

**The GLM/native-build fix worked on the GPU; reconstruction has not run.**
Pinned native submodules were fetched, gsplat/PyTorch3D/nvdiffrast built,
DriveStudio imported, and the CUDA forward/backward/rasterization preflight passed.
This establishes the tested build/kernel path, not successful scene reconstruction
or flying-camera quality.

Source commit: `ea585014026d9791adfa748e6aa2cb018ef5afb8`.
[Execution summary](execution.json), [sanitized evidence](evidence.txt).

## Failure and cause

The job then failed in the archive downloader while atomically replacing the
archive record: `IsADirectoryError: archive_record.part -> archive_record.json`.
Our invocation placed the individual `archive_record.json` file first in the
runner's sync-path list. The runner treats its first sync path as the output
root and runs mkdir on it before the workload, creating a directory with that
filename. Our minimal-transfer invocation was incompatible with that behavior.

The downloader reached record writing after its size/SHA256 verification call.
No archive receipt or trustworthy isolated download-duration measurement was
saved, so no new download-throughput figure is claimed. The native archive was
remote scratch and was lost when the rental was destroyed; it was not copied to
the Mac. Scene preprocessing, the 15k fit, quality metrics and elevated rendering
were never reached: **zero reconstruction iterations and zero flying-view images**.

## Next correction

Use a directory as the first sync path. To retain the minimal-transfer policy,
collect only archive/quality/calibration JSON, target manifest, inspection image,
GIF, 80 RGB targets and ingestion manifest into a dedicated return directory.
Do not sync native data, build environments, alpha sequences or fit scratch.
Add a preflight assertion that the archive-record destination is not a directory
before another costly download. No paid retry was launched for this report.

The managed workload failed and the rental was reported destroyed. Workload
1,077 seconds, setup 45 seconds, provisioning 115 seconds; approximate charge
**$0.083**. This is an orchestration failure, not a negative 3DGS quality result.
