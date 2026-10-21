"""Label-dependent kinds are masked as structured fields, not only as labelled text."""

from __future__ import annotations

import json

import pytest

from infrastructure.safety.masking.policy import ALL_KINDS, MaskingPolicy
from infrastructure.safety.masking.rules import MaskingRules

_LABELS = (
    ("namespace", "kube_namespace", "privacy-test"),
    ("namespace", "namespace", "privacy-test"),
    ("namespace", "ns", "privacy-test"),
    ("cluster", "kube_cluster", "prod-east-1"),
    ("cluster", "eks_cluster", "prod-east-1"),
    ("cluster", "cluster", "prod-east-1"),
    ("cluster", "cluster_name", "prod-east-1"),
    ("service_name", "service", "checkout-api"),
    ("service_name", "service_name", "checkout-api"),
    ("service_name", "app", "checkout-api"),
    ("service_name", "deployment", "checkout-api"),
)


def _rules() -> MaskingRules:
    return MaskingRules(MaskingPolicy.model_validate({"enabled": True, "kinds": ALL_KINDS}))


def _placeholder_kind(kind: str) -> str:
    return f"<{kind.upper()}_0>"


@pytest.mark.parametrize(("kind", "label", "value"), _LABELS)
def test_json_key_is_masked(kind: str, label: str, value: str) -> None:
    text = json.dumps({label: value, "status": "Running"})

    masked = _rules().mask(text)

    assert value not in masked, f"{label!r} JSON field kept its value: {masked}"
    assert f'"{label}": "{_placeholder_kind(kind)}"' in masked


@pytest.mark.parametrize(("kind", "label", "value"), _LABELS)
def test_escaped_json_key_in_a_nested_string_is_masked(kind: str, label: str, value: str) -> None:
    text = json.dumps({"result": json.dumps({label: value})})

    masked = _rules().mask(text)

    assert value not in masked, f"escaped {label!r} field kept its value: {masked}"
    assert _placeholder_kind(kind) in masked


@pytest.mark.parametrize(("kind", "label", "value"), _LABELS)
def test_python_repr_key_is_masked(kind: str, label: str, value: str) -> None:
    text = "Tool call: lookup " + repr({label: value})

    masked = _rules().mask(text)

    assert value not in masked, f"repr {label!r} key kept its value: {masked}"
    assert f"'{label}': '{_placeholder_kind(kind)}'" in masked


@pytest.mark.parametrize(("kind", "label", "value"), _LABELS)
def test_dict_keyed_value_is_masked_by_the_value_walk(kind: str, label: str, value: str) -> None:
    rules = _rules()

    masked = rules.mask_value({"query": {label: value}, "limit": 5})

    assert masked == {"query": {label: _placeholder_kind(kind)}, "limit": 5}
    assert rules.unmask_value(masked) == {"query": {label: value}, "limit": 5}


def test_free_text_under_a_labelled_key_is_masked_by_the_text_rules_only() -> None:
    rules = _rules()

    masked = rules.mask_value({"service": "the checkout path is slow"})

    assert masked == {"service": "the checkout path is slow"}


def test_labelled_text_form_is_unchanged() -> None:
    masked = _rules().mask("kube_namespace:payments-prod saw errors")

    assert masked == "kube_namespace:<NAMESPACE_0> saw errors"
