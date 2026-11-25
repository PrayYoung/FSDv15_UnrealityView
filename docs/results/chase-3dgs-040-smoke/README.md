# Prompt B: scene 040 infrastructure failure

**No usable reconstruction results were retrieved.** The single managed attempt
reached its running phase, then lost its connection beyond the bounded recovery
window. Runner classification: `INFRASTRUCTURE_FAILED`. No workload exit code,
remote log, timing receipts, quality JSON, target images or ingestion receipt was
returned. Reconstruction progress and quality are therefore **unverified**; this
is not evidence of a negative 3DGS result or proof that no reconstruction ran.

Source: `a5f8a92a98ec694859bad7a4b18d2cd65791ec03`.
[Protocol](../../PROMPT_B_PROTOCOL.md), [execution summary](execution.json).

The local fixes were applied before submission: early record-directory rejection,
directory-first minimal artifact collection on EXIT, stage timings and an explicit
ingestion receipt. The full CPU suite passed 124 tests / 1 skipped, with 14 targeted checks
passing afterward. Remote connection loss prevented verification of those stages
in this attempt. No automatic Difix or batch was launched.

## Cleanup and cost reporting

The runner's terminal summary reported approximately **$0.208** at failure, but
omitted a destruction timestamp. Its infrastructure-error branch invokes the
owner's destroy command without recording successful cleanup in metadata.
During this status review, the owner destroy command was reissued successfully
and closure was recorded. The missing timestamp alone does not establish that
the rental continued billing. Do not extrapolate its saved hourly rate to the
review time as an actual provider bill; the final bill has not been independently
verified. The separate $1 owner watchdog was configured as a further safeguard.

No paid retry was started: this was the sole authorized Prompt B attempt. Before
another attempt, improve cleanup/receipt durability and select a reliable host;
local tests or the prior CUDA-build success cannot establish this run's result.
Any new rental requires continuation of the experiment budget/scope.
