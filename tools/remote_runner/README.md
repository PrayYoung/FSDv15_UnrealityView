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

Per-step training logs are not streamed into the chat. The runner quietly observes
remote completion, retrieves its log, then retrieves outputs and destroys the
instance. Unit tests simulate disconnects, lost acknowledgements, genuine nonzero
workload exits and disappearance without an exit record. They do not establish
recovery from every provider or network failure.
