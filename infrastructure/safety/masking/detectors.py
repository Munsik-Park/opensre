"""Regex detectors for sensitive infrastructure identifiers.

Each detector contributes zero or more ``DetectedIdentifier`` matches when
``find_identifiers(text, policy)`` is called. Contextual detectors (namespace,
cluster, service_name) only match when preceded by a recognized label so
that generic words like ``frontend`` are not mistakenly masked. The label may
be written as text (``namespace: x``) or as a quoted key (``"namespace": "x"``,
``\\"namespace\\": \\"x\\"``, ``'namespace': 'x'``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from infrastructure.safety.masking.policy import (
    IdentifierKind,
    MaskingPolicy,
    compile_extra_patterns,
)


@dataclass(frozen=True)
class DetectedIdentifier:
    """A single identifier found in text."""

    kind: str
    start: int
    end: int
    value: str
    #: Captured after a label followed only by whitespace (``namespace payments``).
    bare_label: bool = False


# Built-in detectors. Each entry maps an ``IdentifierKind`` to compiled
# regexes. Contextual detectors capture the VALUE in the ``value`` group so we
# only mask the identifier itself (not the preceding label like "kube_namespace:").

# A masked cut runs every detector over a whole over-budget tool result, so each
# pattern must stay linear on any input. The pod, hostname and email patterns end in
# an alternative that captures nothing and consumes the run of identifier characters
# where the first alternative failed: no later start inside that run can match, so a
# run is scanned a bounded number of times rather than once from every word start.
_POD_RE = re.compile(
    r"\b(?=[a-z0-9])(?:([a-z0-9](?:[-a-z0-9]*[a-z0-9])?-[a-f0-9]{5,10}(?:-[a-z0-9]{3,10})?)\b"
    r"|[-a-z0-9]+)"
)

_NAMESPACE_LABELS = r"kube_namespace|namespace|ns"
_CLUSTER_LABELS = r"kube_cluster|eks_cluster|cluster(?:_name)?"
_SERVICE_NAME_LABELS = r"service|service_name|app|deployment"
_NAMESPACE_VALUE = r"[a-z0-9][-a-z0-9]*"
_NAME_VALUE = r"[a-zA-Z0-9][-a-zA-Z0-9_]{1,}"


def _labelled_text(labels: str, value: str) -> re.Pattern[str]:
    return re.compile(rf"\b(?:{labels})(?P<sep>[=:\s]+)(?P<value>{value})\b", re.IGNORECASE)


def _quoted_key(labels: str, value: str) -> re.Pattern[str]:
    # One quote style per pair: JSON, the escaped JSON a nested string carries, or repr.
    return re.compile(
        rf"(?P<quote>\\?\"|')(?:{labels})(?P=quote)\s*:\s*(?P=quote)(?P<value>{value})(?P=quote)",
        re.IGNORECASE,
    )


_NAMESPACE_RES = (
    _labelled_text(_NAMESPACE_LABELS, _NAMESPACE_VALUE),
    _quoted_key(_NAMESPACE_LABELS, _NAMESPACE_VALUE),
)
_CLUSTER_RES = (
    _labelled_text(_CLUSTER_LABELS, _NAME_VALUE),
    _quoted_key(_CLUSTER_LABELS, _NAME_VALUE),
)
_SERVICE_NAME_RES = (
    _labelled_text(_SERVICE_NAME_LABELS, _NAME_VALUE),
    _quoted_key(_SERVICE_NAME_LABELS, _NAME_VALUE),
)

#: A dict key naming a contextual kind, and the value shape that kind captures.
_STRUCTURED_KEYS: dict[str, tuple[IdentifierKind, re.Pattern[str]]] = {
    **{
        label: (IdentifierKind.NAMESPACE, re.compile(_NAMESPACE_VALUE, re.IGNORECASE))
        for label in ("kube_namespace", "namespace", "ns")
    },
    **{
        label: (IdentifierKind.CLUSTER, re.compile(_NAME_VALUE))
        for label in ("kube_cluster", "eks_cluster", "cluster", "cluster_name")
    },
    **{
        label: (IdentifierKind.SERVICE_NAME, re.compile(_NAME_VALUE))
        for label in ("service", "service_name", "app", "deployment")
    },
}
_KIND_HOSTNAME = r"kind-[a-z0-9][-a-z0-9]*"  # local kind clusters
_HOSTNAME_RE = re.compile(
    rf"\b(?=[a-z0-9])(?:({_KIND_HOSTNAME}"
    # ec2-style internal hostnames: ip-10-0-1-23.ec2.internal
    # Note: the inner character class excludes '.' so the outer (?:\.LABEL)*
    # has no ambiguity and cannot backtrack exponentially (CodeQL ReDoS).
    r"|ip-\d+-\d+-\d+-\d+(?:\.[a-z0-9][a-z0-9-]*)*"
    # Generic DNS-style: label(.label)+.(tld)
    r"|[a-z0-9][-a-z0-9]*(?:\.[a-z0-9][-a-z0-9]*)+\.(?:com|net|org|io|internal|local|cloud)"
    r")\b"
    # A DNS-style name that fails at one start fails at every later start in the same
    # label chain, so the skip consumes the chain. It stops where an ec2-style name
    # starts; a kind name it passes over is found by _KIND_HOSTNAME_RE.
    r"|(?:(?!\bip-\d+-\d+-\d+-\d+(?!\w))[-a-z0-9]|\.(?=[a-z0-9]))+)",
    re.IGNORECASE,
)
_KIND_HOSTNAME_RE = re.compile(rf"\b({_KIND_HOSTNAME})\b", re.IGNORECASE)
_ACCOUNT_RE = re.compile(r"\b(\d{12})\b")
_IP_RE = re.compile(
    r"\b((?:25[0-5]|2[0-4]\d|[01]?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|[01]?\d?\d)){3})\b"
)
_EMAIL_RE = re.compile(r"\b(?:([\w.+-]+@[\w-]+\.[\w.-]+)\b|[\w.+-]+)")


_BUILTIN_DETECTORS: dict[IdentifierKind, tuple[re.Pattern[str], ...]] = {
    IdentifierKind.POD: (_POD_RE,),
    IdentifierKind.NAMESPACE: _NAMESPACE_RES,
    IdentifierKind.CLUSTER: _CLUSTER_RES,
    IdentifierKind.SERVICE_NAME: _SERVICE_NAME_RES,
    IdentifierKind.HOSTNAME: (_HOSTNAME_RE, _KIND_HOSTNAME_RE),
    IdentifierKind.ACCOUNT_ID: (_ACCOUNT_RE,),
    IdentifierKind.IP_ADDRESS: (_IP_RE,),
    IdentifierKind.EMAIL: (_EMAIL_RE,),
}


def structured_kind(key: str, value: str, policy: MaskingPolicy) -> IdentifierKind | None:
    """The enabled kind ``key`` labels when ``value`` has that kind's shape, else ``None``."""
    entry = _STRUCTURED_KEYS.get(key.lower())
    if entry is None:
        return None
    kind, shape = entry
    if not policy.is_kind_enabled(kind) or shape.fullmatch(value) is None:
        return None
    return kind


def find_identifiers(
    text: str,
    policy: MaskingPolicy,
    compiled_extras: dict[str, re.Pattern[str]] | None = None,
) -> list[DetectedIdentifier]:
    """Return all identifiers found in ``text`` under ``policy``.

    Matches are returned sorted by start position. When two detectors match
    overlapping regions (fully contained *or partially overlapping*), the
    longer earlier match wins so we never corrupt the output in
    ``_apply_replacements``.

    ``compiled_extras`` may be passed by callers that want to compile the
    policy's extra regex patterns once per run instead of on
    every call.
    """
    if not policy.enabled or not text:
        return []

    found: list[DetectedIdentifier] = []

    for kind, patterns in _BUILTIN_DETECTORS.items():
        if not policy.is_kind_enabled(kind):
            continue
        for pattern in patterns:
            _append_matches(pattern, text, kind, found)

    extras = compiled_extras if compiled_extras is not None else compile_extra_patterns(policy)
    for label, extra in extras.items():
        _append_matches(extra, text, label, found)

    return _resolve_overlaps(found)


def _append_matches(
    pattern: re.Pattern[str],
    text: str,
    kind: str,
    out: list[DetectedIdentifier],
) -> None:
    named_value = "value" in pattern.groupindex
    labelled = "sep" in pattern.groupindex
    for match in pattern.finditer(text):
        # Prefer the value group (contextual detectors), then group(1), else the full match.
        if named_value:
            start, end = match.span("value")
            value = match.group("value")
        elif match.groups():
            start, end = match.span(1)
            value = match.group(1)
        else:
            start, end = match.span()
            value = match.group()
        if value:
            bare = labelled and not any(char in "=:" for char in match.group("sep"))
            out.append(
                DetectedIdentifier(kind=kind, start=start, end=end, value=value, bare_label=bare)
            )


def _resolve_overlaps(matches: list[DetectedIdentifier]) -> list[DetectedIdentifier]:
    """Drop matches that overlap (fully or partially) a longer earlier match.

    Sort by start ASC, then by length DESC so the longest match at each
    start position is considered first. Kept intervals are non-overlapping and
    start-ordered, so a candidate only needs to compare against the previous
    kept end — O(N) after the sort, not O(N²) against the full kept list.
    """
    if not matches:
        return []
    by_start = sorted(matches, key=lambda m: (m.start, -(m.end - m.start)))
    result: list[DetectedIdentifier] = []
    last_end = -1
    for m in by_start:
        # Overlap with previous kept: last_end > m.start (half-open spans).
        if m.start < last_end:
            continue
        result.append(m)
        last_end = m.end
    return result


__all__ = [
    "DetectedIdentifier",
    "find_identifiers",
    "structured_kind",
]
