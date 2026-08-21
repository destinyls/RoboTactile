"""Canonical ACT adapter factory."""

from robotactile_benchmark.policies.univtac_official_act_loading import (
    load_official_univtac_act_policy,
)

load_act_adapter = load_official_univtac_act_policy

__all__ = ["load_act_adapter"]
