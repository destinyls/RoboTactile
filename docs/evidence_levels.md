# Evidence Levels

| Level | Establishes | Excludes |
|---|---|---|
| Software contract tests | contracts, runner, artifacts, matrix plumbing | learned model and simulator |
| `live_preflight_no_simulator_execution_v1` | request, source, artifact, GPU, and Isaac-Python readiness | simulator allocation and task execution |
| `unqualified_live_univtac_execution_v1` | verified runtime trace | Isaac acceptance and publishable task result |
| `in_process_snapshot_replay_equivalence_v1` | exact post-reset witness equality for conditions reloaded inside one live process | cross-process determinism, Isaac acceptance, and task success |
| Legacy qualification v1/v2 | prior task/action evidence, with at most one global source binding | paper promotion and exact task-local runtime identity |
| Qualification v3 | eight task-local source/observation-parity bindings, including task-specific normalizer and serve bundle | policy outcome and paper sample size |
| Source-bound attempt v3 | one attempt exactly binds qualification v3, the N0 rank-0 attestation, and the Isaac-child attestation | statistical sufficiency and real-robot behavior |
| Dual-attested paper bundle | a complete `paper_v1` summary and all attempts pass exact-match publication verification | hardware fault calibration and real-robot validity |

Evidence levels are monotonic only through explicit new receipts. Running an
unqualified command on a GPU does not change its evidence level. Only the last
row can remove `simulator_not_qualified`; legacy v1/v2 cannot be promoted.

Official comparison uses `initial_state_policy=official_reproduction`: reset
terminal signals remain diagnostics rather than an independent rejection gate.
`replace_initial_terminal_v1` is a separate robustness diagnostic and is not
directly comparable with the public 84.5% reference. A hard watchdog covers the
full Clean lifecycle but is infrastructure-only for newly generated official N0
requests. Only the frozen action/observation horizon can produce a scoreable
timeout; classified `exception_replaced` attempts are outside the model
denominator.
