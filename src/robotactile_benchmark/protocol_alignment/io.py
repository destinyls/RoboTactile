"""Strict loading and no-clobber persistence for protocol alignment reports."""

from __future__ import annotations

import hashlib
import os
import tempfile
from importlib import resources
from pathlib import Path
from typing import Optional

from robotactile_benchmark.clean_baseline.io import (
    read_canonical_json_file,
    write_canonical_no_clobber,
)
from robotactile_benchmark.protocol_alignment.contracts import N0PaperReference

_RESOURCE = "configs/protocols/n0_twam_univtac_paper_v1.json"


def _resource_path() -> Path:
    packaged = resources.files("robotactile_benchmark").joinpath(_RESOURCE)
    if packaged.is_file():
        return Path(str(packaged))
    return Path(__file__).resolve().parents[3] / _RESOURCE


def load_n0_paper_reference(path: Optional[Path] = None) -> N0PaperReference:
    """Load the packaged reference or one explicit strict canonical document."""

    selected = _resource_path() if path is None else Path(path)
    document, _ = read_canonical_json_file(selected, "N0 paper reference")
    return N0PaperReference.from_dict(document)


def write_protocol_alignment_report(path: Path, report: dict[str, object]) -> bool:
    """Publish one canonical report without replacing different evidence."""

    return write_canonical_no_clobber(path, report)


def write_text_no_clobber(path: Path, content: str) -> bool:
    """Atomically publish UTF-8 text; an identical file is an idempotent no-op."""

    target = Path(path).absolute()
    payload = content.encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("protocol alignment text output cannot be a symlink")
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise FileExistsError("refusing to replace a different alignment report")
        return False
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            if target.is_symlink() or target.read_bytes() != payload:
                raise FileExistsError(
                    "refusing to replace a different alignment report"
                ) from error
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)


def render_protocol_alignment_markdown(report: dict[str, object]) -> str:
    """Render the gate matrix as a bounded public-result diagnostic."""

    gates = report.get("gates")
    trials = report.get("trials")
    if not isinstance(gates, list) or not isinstance(trials, list):
        raise TypeError("alignment report lacks gate or trial rows")
    lines = [
        "# N0-TWAM / UniVTAC Protocol Alignment",
        "",
        f"- Campaign: `{report['campaign_id']}`",
        f"- Protocol: `{report['protocol_id']}`",
        f"- Evaluation semantics: `{report['evaluation_semantics']}`",
        f"- Completed tasks: `{report['completed_task_count']}/8`",
        f"- Valid artifacts: `{report['loaded_artifact_count']}/{report['planned_trial_count']}`",
        f"- Observed success: `{report['success_count']}/{report['eligible_trial_count']}`",
        f"- Public-result comparison gate: `{'GO' if report['go_for_paper_comparison'] else 'NO-GO'}`",
        "",
        "## Gates",
        "",
        "| Gate | Status | Comparison blocker | Code | Detail |",
        "|---|---:|---:|---|---|",
    ]
    for item in gates:
        if not isinstance(item, dict):
            raise TypeError("gate rows must be objects")
        detail = str(item["detail"]).replace("|", "\\|")
        lines.append(
            f"| `{item['gate_id']}` | **{str(item['status']).upper()}** | "
            f"{'yes' if item['paper_blocking'] else 'no'} | "
            f"`{item['code']}` | {detail} |"
        )
    lines.extend(
        (
            "",
            "## Trial diagnostics",
            "",
            "| Task | Terminal | Obs | Cycles | Class | Reset | Action path |",
            "|---|---:|---:|---:|---|---|---|",
        )
    )
    for item in trials:
        if not isinstance(item, dict):
            raise TypeError("trial rows must be objects")
        lines.append(
            f"| `{item['task_id']}` | `{item['terminal_status']}` | "
            f"{item['observation_count']} | {item['control_cycle_count']} | "
            f"`{item['classification']}` | `{item['reset_mode']}` | "
            f"`{item['action_execution_mode']}` |"
        )
    lines.extend(
        ("", f"Report content SHA256: `{report['report_content_sha256']}`", "")
    )
    return "\n".join(lines)


def sha256_file(path: Path) -> str:
    """Return a local output hash for CLI receipts."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


__all__ = [
    "load_n0_paper_reference",
    "render_protocol_alignment_markdown",
    "sha256_file",
    "write_protocol_alignment_report",
    "write_text_no_clobber",
]
