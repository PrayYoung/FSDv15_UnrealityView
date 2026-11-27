# Detached transport for the global Vast runner

The global `vast-run` remains the sole lifecycle owner. This helper does not rent
instances, run provider APIs, bypass caps, or create a second job orchestrator.
It is installed into that runner, not run directly by project job scripts.

```bash
python tools/remote_runner/install.py --runner /path/to/vast_run.py
```

The installer backs up the runner and adds a sibling `vast_detached.py`. The
integration is installed on the current development machine. New runs use it;
already-running Python processes retain their old loaded implementation.

Workloads launch under `nohup setsid` with detached stdin/stdout, a remote lock,
a durable log, a PID and an atomically written exit-code file. If SSH loses its
launch acknowledgement, retrying the launch is guarded against duplicate work.
Subsequent SSH connections observe the same worker, without resubmitting its
training. SSH exit 255 is transport loss; only the recorded workload exit code
can report a genuine workload 255. Permanent loss is an infrastructure failure.

SSH keepalives are 30 seconds / 10 failures with TCP keepalive enabled. Reconnection
is bounded to ten minutes of outage. An independent remote timeout and local wait
deadline bound a managed command to six hours; project scripts impose shorter
workload-specific limits. The runner retains artifact synchronization, bounded
infrastructure retries, signal handling and destruction. Permanent host loss or
force-killing the local controller cannot be made failure-free by SSH keepalives.

Verified artifact retrieval is installed with
`python tools/remote_runner/artifact_install.py --runner /path/to/vast_run.py`.
The old owner silently skipped a nonzero SSH path probe, allowing an empty return
to be reported as SUCCESS. Retrieval now makes up to three bounded attempts on
the existing job, rejects missing/empty returns, checks a SHA256 manifest when
present, and records ARTIFACT_FAILED plus the error rather than SUCCESS. Files
are validated in temporary staging before replacing local results. This does
not preserve the remote disk indefinitely: the existing cost deadline and
destruction policy still apply even if all retrieval attempts fail.
A tiny unique receipt roundtrip exercises the same SSH/archive retrieval before
the workload starts, so an initially broken return path fails before training.

The optional pre-rental CUDA capability filter is installed with
`python tools/remote_runner/cuda_install.py --runner /path/to/vast_run.py`.
Use `--min-cuda-version 12.4` for the PixArt CUDA12.4 build and11.8 for the
standalone Difix CUDA11.8 build. The owner filters both its search query and
returned offers before renting; the actual GPU matmul preflight is still required.

The PixArt v2 and standalone Difix scripts seal their small return folders and
also print compact metric summaries into the separately retrieved worker log.
No dataset, pretrained weights or optimizer state is added to result downloads.

Per-step training logs are not streamed into the chat. The runner quietly observes
remote completion, retrieves its log, then retrieves outputs and destroys the
instance. Unit tests simulate disconnects, lost acknowledgements, genuine nonzero
workload exits and disappearance without an exit record. They do not establish
recovery from every provider or network failure.

For storage-constrained jobs, `vast-run run --discard-snapshot-after-upload` opts in
to removing the local compressed upload after a verified transfer. The uncompressed
source/cache bundle remains available and the runner rebuilds the archive for an
infrastructure retry. Default behavior for other projects is unchanged.

Infrastructure recovery can be installed with
`python tools/remote_runner/recovery_install.py --runner /path/to/vast_run.py`.
The owner makes one bounded log/output recovery attempt before destruction,
records confirmed destruction even for infrastructure failures, and retains the
instance reference for finalizer retry if cleanup fails. Normal artifact transfers
remain unchanged; failure recovery transfers use a45-second bound. The opt-in
`--exclude-host-id` flag prevents reselecting a known failed host. No manual SSH,
provider API bypass, workload resubmission or automatic extra paid attempt is used.
