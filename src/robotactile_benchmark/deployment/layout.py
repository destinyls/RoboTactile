"""Resolve, initialize, and verify one repo-contained deployment layout."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from robotactile_benchmark.closed_loop.artifact_contracts import (
    ArtifactValidationError,
)
from robotactile_benchmark.closed_loop.artifact_io import (
    canonical_json_bytes,
    strict_json_bytes,
)
from robotactile_benchmark.contracts import canonical_hash
from robotactile_benchmark.deployment.contracts import (
    DeploymentLayoutError,
    DeploymentLayoutReceipt,
)

DEPLOYMENT_ROOT_ENV = "ROBOTACTILE_DEPLOY_ROOT"
LAYOUT_RECEIPT_RELATIVE_PATH = Path("artifacts/deployment/layout_receipt.json")
_MAX_RECEIPT_BYTES = 1024 * 1024
_BROAD_ROOTS = frozenset(
    {
        Path("/"),
        Path("/data"),
        Path("/data1"),
        Path("/data2"),
        Path("/mnt"),
        Path("/mnt/data"),
    }
)
_DIRECTORIES = (
    "sources",
    "runtime",
    "runtime/cache",
    "runtime/cache/cuda",
    "runtime/cache/pycache",
    "runtime/cache/torch",
    "runtime/home",
    "runtime/locks",
    "runtime/omni-cache",
    "runtime/omni-cache/kit",
    "runtime/omni-cache/ov",
    "runtime/omni-cache/user",
    "runtime/pip-cache",
    "runtime/tmp",
    "artifacts",
    "artifacts/models",
    "artifacts/models/act",
    "artifacts/models/n0_twam",
    "artifacts/deployment",
    "artifacts/preflight",
    "artifacts/rest-references",
    "artifacts/live-univtac",
    "requests",
    "requests/calibration",
    "requests/four-condition",
    "requests/primary-matrix",
    "outputs",
    "outputs/matrices",
    "outputs/reports",
    "logs",
)


def _absolute_path(value: Path, name: str) -> Path:
    if not isinstance(value, Path):
        raise TypeError(f"{name} must be a pathlib.Path")
    selected = value.expanduser()
    if not selected.is_absolute():
        raise DeploymentLayoutError(f"{name} must be absolute")
    return selected.absolute()


def _validate_root(root: Path) -> Path:
    selected = _absolute_path(root, "deployment root")
    if selected in _BROAD_ROOTS:
        raise DeploymentLayoutError("deployment root is too broad")
    if selected.is_symlink():
        raise DeploymentLayoutError("deployment root cannot be a symlink")
    ancestor = selected
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if ancestor.is_symlink():
        raise DeploymentLayoutError("deployment root ancestor cannot be a symlink")
    if ancestor.exists() and not ancestor.is_dir():
        raise DeploymentLayoutError("deployment root ancestor must be a directory")
    if selected.exists() and not selected.is_dir():
        raise DeploymentLayoutError("deployment root must be a directory")
    return selected


def discover_repository_root(start: Optional[Path] = None) -> Path:
    """Find the source checkout that owns the repo-contained default."""

    selected = Path.cwd() if start is None else Path(start)
    selected = selected.expanduser().absolute()
    if selected.is_file():
        selected = selected.parent
    for candidate in (selected, *selected.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "src/robotactile_benchmark").is_dir()
            and (candidate / "integrations/integrations.lock.json").is_file()
        ):
            return candidate
    raise DeploymentLayoutError(
        "cannot discover RoboTactile source root; pass --root or set "
        f"{DEPLOYMENT_ROOT_ENV}"
    )


def resolve_deployment_root(
    explicit: Optional[Path] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    repository_start: Optional[Path] = None,
) -> Path:
    """Resolve CLI, environment, then repo-contained default precedence."""

    if explicit is not None:
        return _validate_root(explicit)
    environment = os.environ if environ is None else environ
    raw = environment.get(DEPLOYMENT_ROOT_ENV)
    if raw is not None:
        if not raw or raw.strip() != raw:
            raise DeploymentLayoutError(
                f"{DEPLOYMENT_ROOT_ENV} must be a non-empty path"
            )
        return _validate_root(Path(raw))
    repository = discover_repository_root(repository_start)
    return _validate_root(repository / "deployment")


@dataclass(frozen=True)
class DeploymentLayout:
    """All owned paths beneath one deployment root."""

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", _validate_root(self.root))

    @property
    def sources(self) -> Path:
        return self.root / "sources"

    @property
    def runtime(self) -> Path:
        return self.root / "runtime"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def model_artifacts(self) -> Path:
        return self.artifacts / "models"

    @property
    def requests(self) -> Path:
        return self.root / "requests"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def receipt_path(self) -> Path:
        return self.root / LAYOUT_RECEIPT_RELATIVE_PATH

    def path(self, relative: str) -> Path:
        if relative not in _DIRECTORIES:
            raise KeyError(f"unknown deployment directory: {relative}")
        return self.root / relative

    def directory_paths(self) -> tuple[Path, ...]:
        return tuple(self.root / relative for relative in _DIRECTORIES)

    def to_dict(self) -> dict[str, object]:
        return {
            "directories": list(_DIRECTORIES),
            "layout_id": "robotactile_repo_contained_v1",
            "root": str(self.root),
            "root_path_sha256": canonical_hash(str(self.root)),
        }


def _check_directory(path: Path) -> None:
    if path.is_symlink():
        raise DeploymentLayoutError(f"deployment path cannot be a symlink: {path}")
    if path.exists() and not path.is_dir():
        raise DeploymentLayoutError(f"deployment path must be a directory: {path}")


def initialize_deployment_layout(layout: DeploymentLayout) -> DeploymentLayoutReceipt:
    """Create only known directories and publish an idempotent receipt."""

    if type(layout) is not DeploymentLayout:
        raise TypeError("layout must be an exact DeploymentLayout")
    _check_directory(layout.root)
    for path in layout.directory_paths():
        _check_directory(path)
    layout.root.mkdir(parents=True, exist_ok=True)
    for path in layout.directory_paths():
        path.mkdir(exist_ok=True)
    receipt = DeploymentLayoutReceipt(
        root_path_sha256=canonical_hash(str(layout.root)),
        directories=_DIRECTORIES,
    )
    write_deployment_layout_receipt(layout.receipt_path, receipt)
    return receipt


def deployment_layout_receipt_bytes(receipt: DeploymentLayoutReceipt) -> bytes:
    if type(receipt) is not DeploymentLayoutReceipt:
        raise TypeError("receipt must be an exact DeploymentLayoutReceipt")
    return canonical_json_bytes(receipt.to_dict())


def write_deployment_layout_receipt(
    path: Path, receipt: DeploymentLayoutReceipt
) -> bool:
    """Publish once, accepting only an identical existing receipt."""

    target = _absolute_path(Path(path), "deployment receipt path")
    payload = deployment_layout_receipt_bytes(receipt)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise DeploymentLayoutError("deployment receipt cannot be a symlink")
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError("refusing to replace a different layout receipt")
        return False
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if not target.is_file() or target.read_bytes() != payload:
                raise FileExistsError(
                    "refusing to replace a concurrently written layout receipt"
                ) from None
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def load_deployment_layout_receipt(path: Path) -> DeploymentLayoutReceipt:
    target = _absolute_path(Path(path), "deployment receipt path")
    if target.is_symlink() or not target.is_file():
        raise DeploymentLayoutError("deployment receipt must be a regular file")
    raw = target.read_bytes()
    if not 1 <= len(raw) <= _MAX_RECEIPT_BYTES:
        raise DeploymentLayoutError("deployment receipt size is invalid")
    try:
        value = strict_json_bytes(raw, "deployment layout receipt")
    except ArtifactValidationError as error:
        raise DeploymentLayoutError(str(error)) from error
    if canonical_json_bytes(value) != raw:
        raise DeploymentLayoutError("deployment receipt is not canonical JSON")
    return DeploymentLayoutReceipt.from_dict(value)


__all__ = [
    "DEPLOYMENT_ROOT_ENV",
    "LAYOUT_RECEIPT_RELATIVE_PATH",
    "DeploymentLayout",
    "discover_repository_root",
    "deployment_layout_receipt_bytes",
    "initialize_deployment_layout",
    "load_deployment_layout_receipt",
    "resolve_deployment_root",
    "write_deployment_layout_receipt",
]
