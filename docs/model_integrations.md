# Model Integrations

RoboTactile registers exactly two model integrations: `act` and `n0_twam`.
Both implement `PolicyAdapter.reset`, `infer`, `commit`, `abort`, and `close`,
and both feed the same closed-loop runner, artifacts, matrix, and reporting.

## ACT

The main package owns preprocessing, typed lifecycle, artifact verification,
qualification, and the reference UniVTAC binding. Model runtime and weights
remain external. ACT's separately trained vision-only profile provides the
matched no-touch control.

## N0-TWAM

The main package owns qpos8 adaptation, request binding, and the typed
reset/infer/prepare-commit/finalize-commit client. The stateful gateway, model,
normalizer, and weights remain external under CC-BY-NC-SA-4.0. A1, A2, and
no-touch are unsupported until corresponding trained artifacts exist.

## UniVTAC

UniVTAC is the shared simulator integration rather than a third model. Raw
observations become canonical evaluation records before any condition is
delivered to either model.
