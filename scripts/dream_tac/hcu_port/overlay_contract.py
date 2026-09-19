"""Immutable Dream-Tac HCU overlay source and wheel identities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

OVERLAY_PROTOCOL: Final[str] = "dream_tac_hcu_source_overlay_v5"
CLAIM_BOUNDARY: Final[str] = "source_compatibility_overlay_only_not_training_success"
PINNED_COMMIT: Final[str] = "14bab51d6862fd07124745c55cd395ea5caa9fd3"
TARGET_RELATIVE_PATH: Final[Path] = Path(
    "cosmos_policy/_src/predict2/networks/minimal_v4_dit.py"
)
ORIGINAL_FILE_SHA256: Final[str] = (
    "e78735f850e9d9c3db12175f52a7f509a0912107432181fc22c5e7c02275181a"
)
PATCHED_FILE_SHA256: Final[str] = (
    "12d2e00006b1b47e40a4cf7e6c5451415154d4dc2219282caf474f74d1449009"
)
EXPERIMENT_CONFIG_RELATIVE_PATH: Final[Path] = Path(
    "cosmos_policy/config/experiment/cosmos_policy_experiment_configs.py"
)
ORIGINAL_EXPERIMENT_CONFIG_SHA256: Final[str] = (
    "f0d64cbb87e568fc57f3e036b3b6fa77b4cc93844af5c8c7689702410e667e92"
)
PATCHED_EXPERIMENT_CONFIG_SHA256: Final[str] = (
    "646ff5873eb36c5c30cddaf3e718831545e4576ee269eaeaadb7f74bfb7eaecc"
)
IMAGINAIRE_CONFIG_RELATIVE_PATH: Final[Path] = Path(
    "cosmos_policy/_src/imaginaire/config.py"
)
ORIGINAL_IMAGINAIRE_CONFIG_SHA256: Final[str] = (
    "b3db09f93b1c3c8b86c7df7426771ea3412ae00f2593a82c104440b422da849c"
)
PATCHED_IMAGINAIRE_CONFIG_SHA256: Final[str] = (
    "7061e6260195214501d9178cff3c18cc2c4f5bf2c87768de5b12b69f1cd2acb5"
)
DISTRIBUTED_RELATIVE_PATH: Final[Path] = Path(
    "cosmos_policy/_src/imaginaire/utils/distributed.py"
)
ORIGINAL_DISTRIBUTED_SHA256: Final[str] = (
    "6f3871571c7ebe233e9cbaf65f34899dcfa3181e1d1762c6d4c7b1d6b65f275c"
)
PATCHED_DISTRIBUTED_SHA256: Final[str] = (
    "fde3d36477bb204d8653d253064e82487d33e358c2101e4bd74e0fe3b8e75ab5"
)
ATTENTION_RELATIVE_PATH: Final[Path] = Path(
    "cosmos_policy/_src/predict2/networks/attention.py"
)
ORIGINAL_ATTENTION_SHA256: Final[str] = (
    "9879fc59c62b4b5a958ebaf8588da1402ee48ddfe7df1979ad986d8a56a34d91"
)
PATCHED_ATTENTION_SHA256: Final[str] = (
    "44f94a6446751a4e27a40bdcb8bf9211e8438f30bf6ab0e2dd25cb2d5468ec97"
)
ITER_SPEED_RELATIVE_PATH: Final[Path] = Path(
    "cosmos_policy/_src/predict2/callbacks/iter_speed.py"
)
ORIGINAL_ITER_SPEED_SHA256: Final[str] = (
    "448ea488041813e164848a852080a7b7e47d7e4b8e35a884443d28e9f5989bf2"
)
PATCHED_ITER_SPEED_SHA256: Final[str] = (
    "03c5649ac34453ac48d0013c5ed921a340e0a100e01da0c3787ba5cd4b3387b4"
)


@dataclass(frozen=True)
class SourcePatchSpec:
    """One exact preimage-to-output transformation in the HCU overlay."""

    relative_path: Path
    preimage_sha256: str
    patched_sha256: str
    semantic_change: str

    def as_manifest_entry(self) -> dict[str, str]:
        return {
            "relative_path": self.relative_path.as_posix(),
            "preimage_sha256": self.preimage_sha256,
            "patched_sha256": self.patched_sha256,
            "semantic_change": self.semantic_change,
        }


SOURCE_PATCH_SPECS: Final[tuple[SourcePatchSpec, ...]] = (
    SourcePatchSpec(
        relative_path=TARGET_RELATIVE_PATH,
        preimage_sha256=ORIGINAL_FILE_SHA256,
        patched_sha256=PATCHED_FILE_SHA256,
        semantic_change=(
            "prefer transformer_engine.pytorch.attention RoPE import and fall back "
            "to transformer_engine.pytorch.attention.rope on ImportError; on HIP "
            "only, replace four broken TransformerEngine RMSNorm instances with "
            "the source-native FP32-accumulation RMSNorm implementation"
        ),
    ),
    SourcePatchSpec(
        relative_path=EXPERIMENT_CONFIG_RELATIVE_PATH,
        preimage_sha256=ORIGINAL_EXPERIMENT_CONFIG_SHA256,
        patched_sha256=PATCHED_EXPERIMENT_CONFIG_SHA256,
        semantic_change=(
            "allow None only for ROBOTACTILE_HCU_CONFIG_PROBE_ONLY=1; otherwise "
            "require DREAM_TAC_BASE_CHECKPOINT and validate it with get_checkpoint_path"
        ),
    ),
    SourcePatchSpec(
        relative_path=IMAGINAIRE_CONFIG_RELATIVE_PATH,
        preimage_sha256=ORIGINAL_IMAGINAIRE_CONFIG_SHA256,
        patched_sha256=PATCHED_IMAGINAIRE_CONFIG_SHA256,
        semantic_change=(
            "admit str | None for the checkpoint load path used only by the explicit "
            "HCU config-probe None path"
        ),
    ),
    SourcePatchSpec(
        relative_path=DISTRIBUTED_RELATIVE_PATH,
        preimage_sha256=ORIGINAL_DISTRIBUTED_SHA256,
        patched_sha256=PATCHED_DISTRIBUTED_SHA256,
        semantic_change=(
            "when ROBOTACTILE_HCU_TRAINING=1 only, skip NVIDIA NVML affinity "
            "and libcudart L2 tuning while preserving torch.cuda and NCCL/RCCL "
            "distributed initialization"
        ),
    ),
    SourcePatchSpec(
        relative_path=ATTENTION_RELATIVE_PATH,
        preimage_sha256=ORIGINAL_ATTENTION_SHA256,
        patched_sha256=PATCHED_ATTENTION_SHA256,
        semantic_change=(
            "when the legacy PyTorch SDPA API requires one fallback backend, "
            "select the probe-validated Flash backend for HIP instead of the "
            "unavailable efficient-attention backend"
        ),
    ),
    SourcePatchSpec(
        relative_path=ITER_SPEED_RELATIVE_PATH,
        preimage_sha256=ORIGINAL_ITER_SPEED_SHA256,
        patched_sha256=PATCHED_ITER_SPEED_SHA256,
        semantic_change=(
            "omit intentional non-finite sentinel metrics for inapplicable sample "
            "subsets from console and Weights & Biases logging without changing "
            "model outputs, aggregate loss, backward, or optimizer behavior"
        ),
    ),
)


@dataclass(frozen=True)
class RuntimeWheelPin:
    """One pure-Python wheel admitted to the isolated HCU overlay runtime."""

    package: str
    version: str
    filename: str
    sha256: str

    def as_manifest_entry(self) -> dict[str, str]:
        return {
            "package": self.package,
            "version": self.version,
            "filename": self.filename,
            "sha256": self.sha256,
        }


PURE_PYTHON_RUNTIME_PINS: Final[tuple[RuntimeWheelPin, ...]] = (
    RuntimeWheelPin(
        package="ftfy",
        version="6.3.1",
        filename="ftfy-6.3.1-py3-none-any.whl",
        sha256="7c70eb532015cd2f9adb53f101fb6c7945988d023a085d127d1573dc49dd0083",
    ),
    RuntimeWheelPin(
        package="wcwidth",
        version="0.2.13",
        filename="wcwidth-0.2.13-py2.py3-none-any.whl",
        sha256="3da69048e4540d84af32131829ff948f1e022c1c6bdb8d6102117aac784f6859",
    ),
    RuntimeWheelPin(
        package="webdataset",
        version="0.2.111",
        filename="webdataset-0.2.111-py3-none-any.whl",
        sha256="57a70eb5d7029303ce2262d900ee3f16443bb5e9cf25f634775ce972859bcee4",
    ),
    RuntimeWheelPin(
        package="braceexpand",
        version="0.1.7",
        filename="braceexpand-0.1.7-py2.py3-none-any.whl",
        sha256="91332d53de7828103dcae5773fb43bc34950b0c8160e35e0f44c4427a3b85014",
    ),
)


__all__ = [
    "ATTENTION_RELATIVE_PATH",
    "CLAIM_BOUNDARY",
    "DISTRIBUTED_RELATIVE_PATH",
    "EXPERIMENT_CONFIG_RELATIVE_PATH",
    "IMAGINAIRE_CONFIG_RELATIVE_PATH",
    "ITER_SPEED_RELATIVE_PATH",
    "ORIGINAL_EXPERIMENT_CONFIG_SHA256",
    "ORIGINAL_FILE_SHA256",
    "ORIGINAL_IMAGINAIRE_CONFIG_SHA256",
    "ORIGINAL_DISTRIBUTED_SHA256",
    "ORIGINAL_ATTENTION_SHA256",
    "ORIGINAL_ITER_SPEED_SHA256",
    "OVERLAY_PROTOCOL",
    "PATCHED_EXPERIMENT_CONFIG_SHA256",
    "PATCHED_FILE_SHA256",
    "PATCHED_IMAGINAIRE_CONFIG_SHA256",
    "PATCHED_DISTRIBUTED_SHA256",
    "PATCHED_ATTENTION_SHA256",
    "PATCHED_ITER_SPEED_SHA256",
    "PINNED_COMMIT",
    "PURE_PYTHON_RUNTIME_PINS",
    "RuntimeWheelPin",
    "SOURCE_PATCH_SPECS",
    "SourcePatchSpec",
    "TARGET_RELATIVE_PATH",
]
