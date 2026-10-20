"""Tests for MaskingPolicy — validation and env-var loading."""

from __future__ import annotations

import pytest

from infrastructure.safety.masking.policy import ALL_KINDS, IdentifierKind, MaskingPolicy


def test_identifier_kind_strenum_membership() -> None:
    assert "pod" in ALL_KINDS
    assert IdentifierKind.POD == "pod"
    assert "not_a_real_kind" not in ALL_KINDS


def test_default_policy_is_disabled() -> None:
    policy = MaskingPolicy()
    assert policy.enabled is False
    assert policy.kinds == ALL_KINDS
    assert policy.extra_patterns == {}


def test_from_env_enabled_true() -> None:
    policy = MaskingPolicy.from_env({"OPENSRE_MASK_ENABLED": "true"})
    assert policy.enabled is True
    assert policy.kinds == ALL_KINDS


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("TRUE", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("", False),
        ("garbage", False),
    ],
)
def test_from_env_bool_parsing(raw: str, expected: bool) -> None:
    policy = MaskingPolicy.from_env({"OPENSRE_MASK_ENABLED": raw})
    assert policy.enabled is expected


def test_from_env_kinds_subset() -> None:
    policy = MaskingPolicy.from_env(
        {"OPENSRE_MASK_ENABLED": "true", "OPENSRE_MASK_KINDS": "pod,namespace"}
    )
    assert policy.kinds == ("pod", "namespace")


def test_from_env_kind_names_match_without_regard_to_case() -> None:
    policy = MaskingPolicy.from_env(
        {"OPENSRE_MASK_ENABLED": "true", "OPENSRE_MASK_KINDS": " POD, Namespace ,SERVICE_NAME"}
    )
    assert policy.kinds == ("pod", "namespace", "service_name")


@pytest.mark.parametrize("kinds", ["pod,namspace", "nope,also_nope"])
def test_from_env_kind_naming_no_kind_stops_while_enabled(kinds: str) -> None:
    with pytest.raises(ValueError, match="OPENSRE_MASK_KINDS"):
        MaskingPolicy.from_env({"OPENSRE_MASK_ENABLED": "true", "OPENSRE_MASK_KINDS": kinds})


def test_from_env_extra_regex_parsed() -> None:
    policy = MaskingPolicy.from_env(
        {
            "OPENSRE_MASK_ENABLED": "true",
            "OPENSRE_MASK_EXTRA_REGEX": '{"jira_key": "\\\\b[A-Z]+-\\\\d+\\\\b"}',
        }
    )
    assert "jira_key" in policy.extra_patterns


@pytest.mark.parametrize(
    "raw",
    ["not valid json", '["payments"]', '{"ns_name": "(payments-prod"}'],
)
def test_from_env_malformed_extra_regex_stops_while_enabled(raw: str) -> None:
    with pytest.raises(ValueError, match="OPENSRE_MASK_EXTRA_REGEX") as raised:
        MaskingPolicy.from_env({"OPENSRE_MASK_ENABLED": "true", "OPENSRE_MASK_EXTRA_REGEX": raw})
    assert "payments" not in str(raised.value), "the error must not quote the operator's pattern"


def test_from_env_disabled_never_reads_the_other_settings() -> None:
    policy = MaskingPolicy.from_env(
        {
            "OPENSRE_MASK_ENABLED": "false",
            "OPENSRE_MASK_KINDS": "pod,namspace",
            "OPENSRE_MASK_EXTRA_REGEX": "not valid json",
        }
    )
    assert policy.enabled is False
    assert policy.kinds == ALL_KINDS
    assert policy.extra_patterns == {}


def test_invalid_regex_in_extra_patterns_raises() -> None:
    with pytest.raises(ValueError, match="not a valid regex"):
        MaskingPolicy.model_validate(
            {"enabled": True, "kinds": ALL_KINDS, "extra_patterns": {"bad": "["}}
        )


def test_is_kind_enabled_respects_enabled_flag() -> None:
    policy = MaskingPolicy.model_validate({"enabled": False, "kinds": ("pod",)})
    assert policy.is_kind_enabled(IdentifierKind.POD) is False


def test_is_kind_enabled_filters_by_selected_kinds() -> None:
    policy = MaskingPolicy.model_validate({"enabled": True, "kinds": ("pod", "namespace")})
    assert policy.is_kind_enabled(IdentifierKind.POD) is True
    assert policy.is_kind_enabled(IdentifierKind.EMAIL) is False
