"""Reversible masking of sensitive infrastructure identifiers.

Replaces pod names, cluster names, hostnames, account ids, service names,
IP addresses, and emails with stable placeholders (``<POD_0>``, ``<CLUSTER_1>``)
in the text OpenSRE sends to external LLMs, and restores the originals in what
comes back before it is executed, shown or stored. One mapping serves an agent
turn (see :mod:`infrastructure.safety.masking.scope`). Complementary to
``infrastructure/safety/guardrails`` which performs one-way redaction for
hard-block rules.

Activated per operator by ``OPENSRE_MASK_ENABLED=true``. Off by default.
"""

from __future__ import annotations

from infrastructure.safety.masking.detectors import DetectedIdentifier, find_identifiers
from infrastructure.safety.masking.policy import (
    ALL_KINDS,
    IdentifierKind,
    MaskingPolicy,
    MaskingPolicyError,
)
from infrastructure.safety.masking.rules import MaskingError, MaskingRules
from infrastructure.safety.masking.scope import (
    MaskingScope,
    active_masking_scope,
    authored_texts_in_scope,
    declared_authored_texts,
    masking_turn_scope,
    request_masking_rules,
    restore_turn_text,
    turn_masking_rules,
)

__all__ = [
    "ALL_KINDS",
    "DetectedIdentifier",
    "IdentifierKind",
    "MaskingError",
    "MaskingPolicy",
    "MaskingPolicyError",
    "MaskingRules",
    "MaskingScope",
    "active_masking_scope",
    "authored_texts_in_scope",
    "declared_authored_texts",
    "find_identifiers",
    "masking_turn_scope",
    "request_masking_rules",
    "restore_turn_text",
    "turn_masking_rules",
]
