"""Masking policy — configurable via environment variables.

A ``MaskingPolicy`` decides which kinds of sensitive infrastructure
identifiers are masked and what extra regex patterns apply. It is built
fresh per run, so env-var changes between runs are
picked up. No module-level singleton.

While masking is enabled a setting that cannot be built as configured raises
:class:`MaskingPolicyError`; a disabled switch never reads the other settings.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from enum import StrEnum
from typing import ClassVar

from pydantic import Field, field_validator

from config.strict_config import StrictConfigModel

logger = logging.getLogger(__name__)


class MaskingPolicyError(ValueError):
    """A masking setting cannot be built as configured; the message names no identifier."""


class IdentifierKind(StrEnum):
    POD = "pod"
    NAMESPACE = "namespace"
    CLUSTER = "cluster"
    HOSTNAME = "hostname"
    ACCOUNT_ID = "account_id"
    IP_ADDRESS = "ip_address"
    EMAIL = "email"
    SERVICE_NAME = "service_name"


ALL_KINDS: tuple[IdentifierKind, ...] = tuple(IdentifierKind)
_KINDS_BY_NAME: dict[str, IdentifierKind] = {kind.value: kind for kind in IdentifierKind}


class MaskingPolicy(StrictConfigModel):
    """Configuration that drives what gets masked before LLM calls."""

    enabled: bool = False
    kinds: tuple[IdentifierKind, ...] = ALL_KINDS
    extra_patterns: dict[str, str] = Field(default_factory=dict)

    _ENV_ENABLED: ClassVar[str] = "OPENSRE_MASK_ENABLED"
    _ENV_KINDS: ClassVar[str] = "OPENSRE_MASK_KINDS"
    _ENV_EXTRA_REGEX: ClassVar[str] = "OPENSRE_MASK_EXTRA_REGEX"

    @field_validator("kinds", mode="before")
    @classmethod
    def _coerce_kinds(cls, value: object) -> tuple[IdentifierKind, ...]:
        if value is None or value == "":
            return ALL_KINDS
        if isinstance(value, str):
            return cls._parse_kinds(value.split(","))
        if isinstance(value, list | tuple):
            return cls._parse_kinds(str(p) for p in value)
        raise ValueError(f"kinds must be a string or list, got {type(value).__name__}")

    @classmethod
    def _parse_kinds(cls, entries: Iterable[str]) -> tuple[IdentifierKind, ...]:
        """Kinds named by ``entries``, matched case-insensitively; no entry means every kind."""
        kinds: list[IdentifierKind] = []
        for entry in (raw.strip() for raw in entries):
            if not entry:
                continue
            kind = _KINDS_BY_NAME.get(entry.lower())
            if kind is None:
                raise MaskingPolicyError(
                    f"{cls._ENV_KINDS} entry {entry!r} names no masking kind; "
                    f"known kinds: {', '.join(_KINDS_BY_NAME)}"
                )
            if kind not in kinds:
                kinds.append(kind)
        return tuple(kinds) if kinds else ALL_KINDS

    @field_validator("extra_patterns")
    @classmethod
    def _validate_extra_patterns(cls, value: dict[str, str]) -> dict[str, str]:
        for label, pattern in value.items():
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"extra_patterns[{label!r}] is not a valid regex: {exc}") from exc
        return value

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> MaskingPolicy:
        """Build a policy from the current environment (or an injected dict).

        Raises :class:`MaskingPolicyError` when masking is enabled and a setting
        cannot be built as configured.
        """
        source = env if env is not None else os.environ
        if not _parse_bool(source.get(cls._ENV_ENABLED, "")):
            return cls()
        kinds = cls._parse_kinds((source.get(cls._ENV_KINDS, "") or "").split(","))
        extra_patterns = cls._parse_extra_patterns(source.get(cls._ENV_EXTRA_REGEX, "") or "")
        return cls.model_validate(
            {"enabled": True, "kinds": kinds, "extra_patterns": extra_patterns}
        )

    @classmethod
    def _parse_extra_patterns(cls, raw: str) -> dict[str, str]:
        """Label → regex from the JSON setting; every failure names the setting, never a pattern."""
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            raise MaskingPolicyError(f"{cls._ENV_EXTRA_REGEX} is not valid JSON") from None
        if not isinstance(parsed, dict):
            raise MaskingPolicyError(
                f"{cls._ENV_EXTRA_REGEX} must be a JSON object of label to regular expression"
            )
        patterns = {str(label): str(pattern) for label, pattern in parsed.items()}
        for label, pattern in patterns.items():
            try:
                re.compile(pattern)
            except re.error:
                raise MaskingPolicyError(
                    f"{cls._ENV_EXTRA_REGEX} entry {label!r} is not a valid regular expression"
                ) from None
        return patterns

    def is_kind_enabled(self, kind: IdentifierKind) -> bool:
        return self.enabled and kind in self.kinds


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def compile_extra_patterns(policy: MaskingPolicy) -> dict[str, re.Pattern[str]]:
    """Compile a policy's extra regex patterns into a label→Pattern dict.

    Public helper so callers (e.g. MaskingRules) can compile once per
    run rather than on every mask call.
    """
    compiled: dict[str, re.Pattern[str]] = {}
    for label, pattern in policy.extra_patterns.items():
        try:
            compiled[label] = re.compile(pattern)
        except re.error as exc:
            logger.warning("[masking] skipping extra pattern %r (invalid regex): %s", label, exc)
    return compiled


__all__ = [
    "ALL_KINDS",
    "IdentifierKind",
    "MaskingPolicy",
    "MaskingPolicyError",
    "compile_extra_patterns",
]
