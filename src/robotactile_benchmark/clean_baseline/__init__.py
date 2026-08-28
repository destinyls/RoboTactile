"""Frozen clean-only campaign construction and strict aggregation."""

from robotactile_benchmark.clean_baseline.aggregation import (
    CleanArtifactInventory,
    IncompleteCleanCampaignError,
    VerifiedCleanArtifact,
    aggregate_clean_campaign,
    build_clean_baseline_summary,
    load_clean_artifact_inventory,
)
from robotactile_benchmark.clean_baseline.contracts import (
    CLEAN_BASELINE_EVIDENCE_LEVEL,
    CLEAN_BASELINE_SUMMARY_SEMANTIC_VERSION,
    CLEAN_CAMPAIGN_SAMPLING_SEMANTIC_VERSION,
    CLEAN_CAMPAIGN_SEMANTIC_VERSION,
    CLEAN_POLICY_SEED_ROLE,
    CLEAN_SEED_DERIVATION,
    CLEAN_SIMULATOR_SEED_ROLE,
    CLEAN_UNIVTAC_SEED_DERIVATION,
    CleanCampaignError,
    CleanCampaignManifest,
    CleanCampaignProtocol,
    CleanCampaignSamplingSpec,
    CleanCampaignTrialSpec,
)
from robotactile_benchmark.clean_baseline.generation import (
    build_clean_campaign_manifest,
    discover_clean_request_paths,
)
from robotactile_benchmark.clean_baseline.io import (
    load_clean_baseline_summary,
    load_clean_campaign_manifest,
    write_clean_baseline_summary,
    write_clean_campaign_manifest,
)
from robotactile_benchmark.clean_baseline.paper_bundle import (
    PAPER_RESULT_EVIDENCE_LEVEL,
    PAPER_RESULT_V3_EVIDENCE_LEVEL,
    PaperResultBundle,
    build_paper_result_bundle,
)
from robotactile_benchmark.clean_baseline.qualification import (
    VerifiedAllTaskQualification,
    verify_all_task_qualification,
)
from robotactile_benchmark.clean_baseline.seeds import (
    derive_clean_campaign_seed,
    derive_univtac_task_seed,
    expected_clean_seed_pairs,
    expected_univtac_task_seed_pairs,
    univtac_task_seed_start,
)
from robotactile_benchmark.clean_baseline.summary import (
    CleanBaselineSummary,
    CleanTaskSummary,
    ProtocolInvalidTrial,
)

__all__ = [
    "CLEAN_BASELINE_EVIDENCE_LEVEL",
    "CLEAN_BASELINE_SUMMARY_SEMANTIC_VERSION",
    "CLEAN_CAMPAIGN_SEMANTIC_VERSION",
    "CLEAN_CAMPAIGN_SAMPLING_SEMANTIC_VERSION",
    "CLEAN_POLICY_SEED_ROLE",
    "CLEAN_SEED_DERIVATION",
    "CLEAN_SIMULATOR_SEED_ROLE",
    "CLEAN_UNIVTAC_SEED_DERIVATION",
    "PAPER_RESULT_EVIDENCE_LEVEL",
    "PAPER_RESULT_V3_EVIDENCE_LEVEL",
    "CleanArtifactInventory",
    "CleanBaselineSummary",
    "CleanCampaignError",
    "CleanCampaignManifest",
    "CleanCampaignProtocol",
    "CleanCampaignSamplingSpec",
    "CleanCampaignTrialSpec",
    "CleanTaskSummary",
    "IncompleteCleanCampaignError",
    "ProtocolInvalidTrial",
    "PaperResultBundle",
    "VerifiedAllTaskQualification",
    "VerifiedCleanArtifact",
    "aggregate_clean_campaign",
    "build_clean_baseline_summary",
    "build_clean_campaign_manifest",
    "build_paper_result_bundle",
    "discover_clean_request_paths",
    "derive_clean_campaign_seed",
    "derive_univtac_task_seed",
    "expected_clean_seed_pairs",
    "expected_univtac_task_seed_pairs",
    "load_clean_artifact_inventory",
    "load_clean_baseline_summary",
    "load_clean_campaign_manifest",
    "write_clean_baseline_summary",
    "write_clean_campaign_manifest",
    "univtac_task_seed_start",
    "verify_all_task_qualification",
]
