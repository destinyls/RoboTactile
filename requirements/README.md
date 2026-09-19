# Reproducible pip environments

RoboTactile can be installed, tested, and built without `uv`.

- `core.lock.txt` pins the dependency-light runtime.
- `dev.lock.txt` pins the runtime, test, lint, type-check, build, and optional
  import dependencies needed to check the complete source tree.
- `visualization.lock.txt` pins the optional Pillow renderer dependency.
- all three files include distribution hashes and are consumed with
  `pip --require-hashes`.

From a source checkout, the supported development installation is:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements/dev.lock.txt
python -m pip install --no-deps --no-build-isolation .
```

For a non-development visualization environment, install the optional
visualization dependency separately. Binary-only mode prevents an unrecorded
local Pillow build when no hash-locked wheel matches the current platform:

```bash
python -m pip install --only-binary=:all: --require-hashes \
  -r requirements/visualization.lock.txt
python -m pip install --no-deps --no-build-isolation .
```

On Windows PowerShell, use `.\.venv\Scripts\Activate.ps1` instead of the
`source` command; the remaining commands are unchanged.

The second command intentionally separates third-party resolution from the
local package build. `hatchling` is already pinned in the development lock,
and `--no-deps --no-build-isolation` prevents an unrecorded resolver pass.

For release-wheel consumers, install the runtime lock and the wheel without
dependency resolution:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements/core.lock.txt
python -m pip install --no-deps \
  dist/robotactile_benchmark-0.6.0-py3-none-any.whl
```

`uv.lock` remains an optional maintainer artifact. It is not required by the
pip workflow, CI, Makefile, model integrations, or Isaac runtime. When project
dependencies change, maintainers update `uv.lock`, export the affected pip lock
files, and verify the pip-only CI job before merging. Rerun the local package
install after changing source files; the supported bootstrap deliberately
avoids depending on the pip version required by PEP 660 editable installs.
