"""N0-TWAM Clean/Faulted campaign generation and strict contracts."""

from robotactile_benchmark.n0_fault_campaign.aggregation import (
    aggregate_n0_fault_campaign,
)
from robotactile_benchmark.n0_fault_campaign.contracts import (
    N0_FAULT_CAMPAIGN_GENERATION_SEMANTIC_VERSION,
    N0_FAULT_CAMPAIGN_SEMANTIC_VERSION,
    N0_FAULT_TEMPLATE_SEED_DERIVATION,
    N0_SUPPORTED_OPERATOR_IDS,
    N0_UNSUPPORTED_CONTRACT_SEMANTIC_VERSION,
    N0_UNSUPPORTED_OPERATOR_IDS,
    N0FaultCampaignCellSpec,
    N0FaultCampaignError,
    N0FaultCampaignManifest,
    N0FaultCellDisposition,
    N0UnsupportedContractSpec,
)
from robotactile_benchmark.n0_fault_campaign.generation import (
    N0FaultCampaignGenerationSpec,
    derive_operator_template_seed,
    generate_n0_fault_campaign_bundle,
)
from robotactile_benchmark.n0_fault_campaign.io import (
    LoadedN0FaultCampaign,
    N0FaultCampaignGenerationReceipt,
    load_n0_fault_campaign_bundle,
)
from robotactile_benchmark.n0_fault_campaign.report_adapter import (
    LoadedN0CampaignOutcomes,
    load_n0_campaign_outcomes,
    reporting_spec_for_campaign,
)
from robotactile_benchmark.n0_fault_campaign.report_bundle import (
    N0_FAULT_REPORT_EVIDENCE_LEVEL,
    N0_FAULT_REPORT_SEMANTIC_VERSION,
    N0FaultReportMember,
    N0FaultReportReceipt,
    N0FaultReportWriteResult,
    write_n0_fault_report_bundle,
)
from robotactile_benchmark.n0_fault_campaign.reporting import (
    N0_FAULT_SUMMARY_SEMANTIC_VERSION,
    N0AxisSummary,
    N0FaultCampaignSummary,
    N0FaultReportingSpec,
    N0OperatorCellSummary,
    N0SeverityCurve,
    N0SeverityPoint,
    N0TaskSummary,
    OutcomeBreakdown,
)
from robotactile_benchmark.n0_fault_campaign.runner import (
    N0FaultPairRunReceipt,
    run_n0_fault_pair,
)

__all__ = [
    "LoadedN0FaultCampaign",
    "LoadedN0CampaignOutcomes",
    "N0_FAULT_CAMPAIGN_GENERATION_SEMANTIC_VERSION",
    "N0_FAULT_CAMPAIGN_SEMANTIC_VERSION",
    "N0_FAULT_REPORT_EVIDENCE_LEVEL",
    "N0_FAULT_REPORT_SEMANTIC_VERSION",
    "N0_FAULT_SUMMARY_SEMANTIC_VERSION",
    "N0_FAULT_TEMPLATE_SEED_DERIVATION",
    "N0_SUPPORTED_OPERATOR_IDS",
    "N0_UNSUPPORTED_CONTRACT_SEMANTIC_VERSION",
    "N0_UNSUPPORTED_OPERATOR_IDS",
    "N0AxisSummary",
    "N0FaultCampaignCellSpec",
    "N0FaultCampaignError",
    "N0FaultCampaignGenerationReceipt",
    "N0FaultCampaignGenerationSpec",
    "N0FaultCampaignManifest",
    "N0FaultCampaignSummary",
    "N0FaultCellDisposition",
    "N0FaultPairRunReceipt",
    "N0FaultReportMember",
    "N0FaultReportReceipt",
    "N0FaultReportWriteResult",
    "N0FaultReportingSpec",
    "N0OperatorCellSummary",
    "N0SeverityCurve",
    "N0SeverityPoint",
    "N0TaskSummary",
    "N0UnsupportedContractSpec",
    "OutcomeBreakdown",
    "aggregate_n0_fault_campaign",
    "derive_operator_template_seed",
    "generate_n0_fault_campaign_bundle",
    "load_n0_fault_campaign_bundle",
    "load_n0_campaign_outcomes",
    "reporting_spec_for_campaign",
    "run_n0_fault_pair",
    "write_n0_fault_report_bundle",
]
