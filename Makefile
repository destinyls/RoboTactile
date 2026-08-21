VENV ?= .venv
PYTHON ?= $(VENV)/bin/python
RUN = $(PYTHON) -m

.PHONY: bootstrap build check format-check lint matrix pip-check registry smoke test typecheck

bootstrap:
	bash scripts/bootstrap_pip.sh "$(VENV)"

test:
	$(RUN) pytest -q

registry:
	$(RUN) robotactile_benchmark.cli validate-registry

smoke:
	$(RUN) robotactile_benchmark.cli smoke-replay --output outputs/smoke

matrix:
	$(RUN) robotactile_benchmark.cli smoke-matrix --output outputs/smoke-matrix

format-check:
	$(RUN) ruff format --check .

lint:
	$(RUN) ruff check .

typecheck:
	$(RUN) mypy src

pip-check:
	$(RUN) pip check

check: format-check lint typecheck test pip-check registry

build:
	$(RUN) hatchling build
