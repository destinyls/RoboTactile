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
- Isaac-qualified: reserved for a separately validated environment receipt.
- Task result: requires qualified trials and source-bound result artifacts.
- Real robot: outside the current UniVTAC release.

No software test, synthetic operator image, or installation receipt is treated
as hardware fault calibration or task-performance evidence.

## Reproducibility

Operators, severity paths, trial manifests, source maps, traces, outcomes, and
reports are content addressed. External repositories and model artifacts must
match immutable commits and SHA-256 manifests.

## Limitations

Synthetic delivery faults approximate model-visible consequences rather than
complete sensor mechanics. Current public code does not bundle an
Isaac-qualified full result matrix or real-robot evaluation.
