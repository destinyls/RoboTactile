# Contributing to RoboTactile

Open an issue before changing operator semantics, schemas, evidence levels, or
the `PolicyAdapter` lifecycle. Small fixes may proceed directly with tests.

Development uses Python 3.9+, `venv`, and the hash-locked pip environment:

```bash
bash scripts/bootstrap_pip.sh
source .venv/bin/activate
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m pytest -q
```

New model integrations must use the static registry, declare their external
license and commit, provide a strict artifact manifest, and fail closed for
unsupported benchmark conditions. Do not commit weights, datasets, generated
outputs, credentials, internal paths, or third-party source trees.
