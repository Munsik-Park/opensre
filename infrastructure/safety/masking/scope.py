"""The masking scope of an agent turn, and the rules a request outside it uses.

One ``MaskingRules`` serves every provider request of an agent turn. It lives
in a ``ContextVar`` bound for the turn's block, never on a shared client or in
a store, so two sessions never share it: worker threads and executors do not
inherit ContextVars, and a turn of another session opened inside it binds its
own scope until it returns.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from infrastructure.safety.masking.policy import MaskingPolicy
from infrastructure.safety.masking.rules import MaskingRules


class MaskingScope:
    """The mapping of one agent turn; the policy is read on first use."""

    def __init__(self, session_key: str | None) -> None:
        self.session_key = session_key
        self._rules: MaskingRules | None = None

    @property
    def rules(self) -> MaskingRules:
        """The turn's rules; raises ``MaskingPolicyError`` when an enabled setting is invalid."""
        if self._rules is None:
            self._rules = MaskingRules(MaskingPolicy.from_env())
        return self._rules

    def restore(self, value: Any) -> Any:
        """Restore this turn's placeholders in ``value``; unchanged when nothing was masked."""
        rules = self._rules
        if rules is None:
            return value
        return rules.unmask_value(value)


_CURRENT_SCOPE: ContextVar[MaskingScope | None] = ContextVar("opensre_masking_scope", default=None)
_AUTHORED_TEXTS: ContextVar[tuple[str, ...]] = ContextVar(
    "opensre_masking_authored_texts", default=()
)


@contextmanager
def masking_turn_scope(session_key: str | None) -> Iterator[MaskingScope]:
    """Join the active scope of the same session, or bind a new scope for the block."""
    active = _CURRENT_SCOPE.get()
    if active is not None and session_key is not None and active.session_key == session_key:
        yield active
        return
    scope = MaskingScope(session_key)
    token = _CURRENT_SCOPE.set(scope)
    try:
        yield scope
    finally:
        _CURRENT_SCOPE.reset(token)


@contextmanager
def active_masking_scope() -> Iterator[MaskingScope]:
    """The active scope whatever its session, or an unkeyed scope bound for the block."""
    active = _CURRENT_SCOPE.get()
    if active is not None:
        yield active
        return
    with masking_turn_scope(None) as scope:
        yield scope


def restore_turn_text(text: str) -> str:
    """``text`` with the active turn's placeholders restored."""
    scope = _CURRENT_SCOPE.get()
    return scope.restore(text) if scope is not None else text


def turn_masking_rules() -> MaskingRules:
    """The active turn's rules, or rules local to the caller when no turn is active."""
    scope = _CURRENT_SCOPE.get()
    return scope.rules if scope is not None else MaskingRules(MaskingPolicy.from_env())


def request_masking_rules() -> MaskingRules:
    """Fresh rules confined to one request made outside any agent turn."""
    return MaskingRules(MaskingPolicy.from_env())


@contextmanager
def declared_authored_texts(*texts: str) -> Iterator[None]:
    """Declare OpenSRE-authored texts a caller composes into the turn message."""
    token = _AUTHORED_TEXTS.set((*_AUTHORED_TEXTS.get(), *(text for text in texts if text)))
    try:
        yield
    finally:
        _AUTHORED_TEXTS.reset(token)


def authored_texts_in_scope() -> tuple[str, ...]:
    """The authored texts declared around the current call."""
    return _AUTHORED_TEXTS.get()


__all__ = [
    "MaskingScope",
    "active_masking_scope",
    "authored_texts_in_scope",
    "declared_authored_texts",
    "masking_turn_scope",
    "request_masking_rules",
    "restore_turn_text",
    "turn_masking_rules",
]
