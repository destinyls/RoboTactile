# RoboTactile Benchmark Card

## Purpose

RoboTactile evaluates optical-tactile robot policies under 14 registered
Availability, Fidelity, Temporal, and Context operators. Each eligible system
is compared under matched `clean`, `faulted`, `no_touch`, and `restored`
conditions using the same trial identity and action contract.

## Model integrations

ACT and N0-TWAM are first-class integrations behind one `PolicyAdapter`. ACT
has a registered matched vision-only profile. Frozen N0-TWAM has no matched
no-touch artifact and cannot represent structural tactile absence; those cells
remain explicitly unsupported.

## Evidence levels

- Software contracts: deterministic contracts and protocol plumbing.
- Live preflight: frozen deployment resources and host readiness, without allocation.
- Unqualified live: a hash-verified runtime trace without simulator acceptance.
- Qualification v3: eight ordered task-local source/observation-parity bindings,
  including a task-specific normalizer and serve bundle.
- Qualified task result: requires source-bound attempt v3 evidence from both the
  N0 rank-0 server and the Isaac child.
- Real robot: outside the current UniVTAC release.

No software test, synthetic operator image, or installation receipt is treated
as hardware fault calibration or task-performance evidence.

## Reproducibility

Operators, severity paths, trial manifests, source maps, traces, outcomes, and
reports are content addressed. External repositories and model artifacts must
match immutable commits and SHA-256 manifests.

Clean baselines use an independent preregistered campaign manifest. It records
the master seed and deterministic seed roles, canonical request hashes, exact
trial/run identities, and all planned artifact paths. `paper_v1` requires all
eight frozen UniVTAC tasks with at least 100 complete-horizon trials per task;
`diagnostic_v1` and `pilot_v1` cannot support paper claims.

N0 Clean execution is source-bound to
`robotactile_n0_training_60hz_ee_v1`: every absolute EE8 endpoint advances two
120 Hz physics ticks and one render, matching the 60 Hz training-row cadence.
The upstream variable-waypoint EE executor is retained only as an explicit
reference diagnostic.

Clean aggregation is intention-to-treat at the planned-inventory boundary. A
live artifact is accepted only with one matching successful command-attempt
receipt and log hash. Missing, invalid, ineligible, and protocol-invalid trials
remain explicit. Per-task success uses Wilson intervals; the equal-task macro
uses a seeded task-stratified bootstrap. Partial summaries are available only
through an explicit diagnostic flag and remain non-claimable.

The hard watchdog covers startup, reset, policy execution, artifact export, and
teardown. Official N0 Clean requests treat it as infrastructure-only; elapsed
wall time never becomes a scoreable failure. Ordinary model outcomes, including
an official action-horizon timeout, remain in the denominator. Only a classified
execution exception with matching lifecycle evidence is `exception_replaced`
and excluded before a preregistered reserve candidate is used.

## Qualification and publication gate

`qualify_univtac_all_tasks.sh` checks import, reset, and the declared action
contract once for every frozen task. Qualification v3 augments those receipts
with eight task-local observation-parity/source bindings; normalizer and serve
bundle identities may be task-specific. The N0-only all-task runner then creates
and propagates the N0 rank-0 attestation, Isaac child attestation, and attempt v3
without user-supplied child flags. `clean-campaign-publish` removes
`simulator_not_qualified` only for a complete `paper_v1` bundle with exact
qualification/attestation matches. Legacy v1/v2 qualifications remain loadable
but cannot authorize promotion. Diagnostic one-episode-per-task campaigns are
non-claimable by construction.

## Limitations

Synthetic delivery faults approximate model-visible consequences rather than
complete sensor mechanics. Current public code does not bundle an
Isaac-qualified full result matrix or real-robot evaluation.

Individual Clean artifacts remain unqualified evidence. Paper promotion is a
separate dual-attested bundle gate and does not establish real-robot validity.
The official-comparison default is `official_reproduction`, which records but
does not reject reset-time terminal signals. The opt-in
`replace_initial_terminal_v1` behavior is a separate robustness diagnostic and
is not directly comparable with the public 84.5% reference. This repository
does not claim a newly executed GPU result merely because these gates exist.
