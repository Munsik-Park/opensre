"""Per-run masking rules.

``MaskingRules`` holds a ``MaskingPolicy`` and a stable placeholder map
for the lifetime of a single run (an agent turn, or one request outside a
turn). Mask and unmask operations run over strings, lists, and dicts. The map
lives in memory only and dies with its run.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from pydantic import BaseModel

from infrastructure.safety.masking.detectors import (
    DetectedIdentifier,
    find_identifiers,
    structured_kind,
)
from infrastructure.safety.masking.policy import MaskingPolicy, compile_extra_patterns

# Placeholders are always ``<KIND_N>`` (no nested ``<>``). One scan finds every
# token; dict lookup restores known ones and leaves unknown angle-brackets alone.
_PLACEHOLDER_TOKEN_RE = re.compile(r"<[^<>]+>")

# A token with a placeholder's shape. One a source text already holds is not this
# run's: a masked cut hands it back as written, never as the mapping's original.
_PLACEHOLDER_SHAPE_RE = re.compile(r"<[A-Z0-9_]+_[0-9]+>")
_PRIVATE_USE_AREA = range(0xE000, 0xF900)

# Characters that continue an identifier token, so a known value is replaced
# only where it stands alone: ``privacy-test`` in ``privacy-test/pod`` but not
# inside ``privacy-test-7b5b4cbd55-abcde``.
_TOKEN_CHARS = r"A-Za-z0-9_\-"
_TOKEN_CHAR_RE = re.compile(rf"[{_TOKEN_CHARS}]")

# A value captured only after a whitespace label propagates when its shape sets
# it apart from a word: at least one letter with a digit or one of these.
_SHAPE_MARKS = frozenset("-_.@:")

#: One string's maskable segments (start, end) with the detections inside each.
type _Plan = list[tuple[int, int, list[DetectedIdentifier]]]


class MaskingError(RuntimeError):
    """Masking a request failed; the message names the failure's type, never the text."""


class MaskingRules:
    """Stable masking state for one run."""

    def __init__(
        self,
        policy: MaskingPolicy,
        placeholder_map: dict[str, str] | None = None,
    ) -> None:
        self.policy = policy
        # placeholder -> original value
        self._placeholder_map: dict[str, str] = dict(placeholder_map or {})
        # original value -> placeholder (reverse for reuse/stability)
        self._reverse_map: dict[str, str] = {
            original: placeholder for placeholder, original in self._placeholder_map.items()
        }
        # running counter per kind so placeholder numbers stay stable within a run
        self._counters: dict[str, int] = self._derive_counters()
        # Compile extra regex patterns once per context to avoid per-call work
        self._compiled_extras: dict[str, re.Pattern[str]] = compile_extra_patterns(policy)
        # Originals also replaced where no detector sees them (the set only grows).
        self._propagating: set[str] = set()
        self._propagation_re: re.Pattern[str] | None = None
        self._propagation_size = 0

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> MaskingRules:
        """Reconstruct a context from a persisted state dict.

        Policy is re-read from the environment so env changes are honoured.
        ``placeholder_map`` carries the mappings accumulated by earlier nodes
        in the same run.
        """
        policy = MaskingPolicy.from_env()
        existing = state.get("masking_map") or {}
        if not isinstance(existing, dict):
            existing = {}
        return cls(policy=policy, placeholder_map=dict(existing))

    @property
    def placeholder_map(self) -> dict[str, str]:
        return dict(self._placeholder_map)

    def _derive_counters(self) -> dict[str, int]:
        # Accumulate the maximum index per kind across the whole map first,
        # then add 1 once at the end. Doing "+1" inside the loop would
        # over-count when the map is iterated out of ascending order
        # (e.g. <NS_2>, <NS_0> would yield 4 instead of 3).
        max_index: dict[str, int] = {}
        for placeholder in self._placeholder_map:
            kind, _, index = placeholder.strip("<>").rpartition("_")
            if not kind or not index.isdigit():
                continue
            key = kind.lower()
            max_index[key] = max(max_index.get(key, -1), int(index))
        return {key: value + 1 for key, value in max_index.items()}

    @staticmethod
    def _canonical_label(kind: str) -> str:
        """Token label for ``kind``, upholding the single-bracket invariant.

        Labels come from user extra_patterns config; the one-pass unmask
        matches ``<[^<>]+>``, so brackets and spaces must not survive into
        the token.
        """
        return re.sub(r"[^A-Za-z0-9_]", "_", kind.upper()).strip("_") or "EXTRA"

    def _new_placeholder(self, kind: str) -> str:
        # Counters key on the canonical label, matching what
        # ``_derive_counters`` reads back out of restored placeholders —
        # a raw-kind key would reset to index 0 after a state round trip
        # and overwrite the earlier secret.
        label = self._canonical_label(kind)
        key = label.lower()
        index = self._counters.get(key, 0)
        self._counters[key] = index + 1
        return f"<{label}_{index}>"

    def _ensure_placeholder(self, kind: str, value: str) -> str:
        if value in self._reverse_map:
            return self._reverse_map[value]
        placeholder = self._new_placeholder(kind)
        self._placeholder_map[placeholder] = value
        self._reverse_map[value] = placeholder
        return placeholder

    def _detect(self, text: str) -> list[DetectedIdentifier]:
        """Find identifiers in ``text``, giving each a placeholder and its propagation class."""
        matches = find_identifiers(text, self.policy, self._compiled_extras)
        for match in matches:
            self._ensure_placeholder(match.kind, match.value)
            if not match.bare_label or _distinct_shape(match.value):
                self._propagating.add(match.value)
        return matches

    def mask(self, text: str) -> str:
        """Return ``text`` with sensitive identifiers replaced by placeholders.

        Pass-through (identity) when the policy is disabled.
        """
        if not self.policy.enabled or not text:
            return text
        matches = self._detect(text)
        if not matches:
            return text
        return self._apply_replacements(text, matches)

    def _apply_replacements(self, text: str, matches: list[DetectedIdentifier]) -> str:
        # One forward pass + join: O(L + N_m). Prefer start order (find_identifiers
        # already returns it); sort defensively if a caller passes unsorted spans.
        parts: list[str] = []
        cursor = 0
        for m in sorted(matches, key=lambda x: x.start):
            if m.start < cursor:
                continue
            parts.append(text[cursor : m.start])
            parts.append(self._ensure_placeholder(m.kind, m.value))
            cursor = m.end
        parts.append(text[cursor:])
        return "".join(parts)

    def unmask(self, text: str) -> str:
        """Restore known placeholders in ``text`` (single left-to-right scan).

        Token boundaries are ``<…>``, so ``<NAMESPACE_10>`` is never partially
        rewritten by a shorter key like ``<NAMESPACE_1>``. Replacement text is
        not re-scanned — originals that happen to contain angle-brackets stay
        literal (identifiers from detectors do not look like placeholders).
        """
        if not text or not self._placeholder_map:
            return text
        if "<" not in text:
            return text
        mapping = self._placeholder_map
        return _PLACEHOLDER_TOKEN_RE.sub(
            lambda match: mapping.get(match.group(0), match.group(0)),
            text,
        )

    def mask_value(self, value: Any) -> Any:
        """Recursively mask strings inside dicts/lists/tuples.

        A string under a dict key that labels namespace, cluster or service_name
        is masked as that kind when it has the kind's shape.
        """
        if isinstance(value, str):
            return self.mask(value)
        if isinstance(value, dict):
            return {k: self._mask_item(k, v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.mask_value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.mask_value(v) for v in value)
        return value

    def _mask_item(self, key: Any, value: Any) -> Any:
        kind = self._structured_kind(key, value)
        if kind is None:
            return self.mask_value(value)
        self._propagating.add(value)
        return self._ensure_placeholder(kind, value)

    def _structured_kind(self, key: Any, value: Any) -> str | None:
        if not self.policy.enabled or not isinstance(key, str) or not isinstance(value, str):
            return None
        return structured_kind(key, value, self.policy)

    def unmask_value(self, value: Any, *, keys: bool = False) -> Any:
        """Recursively unmask strings inside dicts/lists/tuples and pydantic models.

        ``keys`` also restores string dict keys.
        """
        return self._unmask_walk(value, self.unmask, keys=keys)

    def unmask_resent(self, value: Any) -> Any:
        """``unmask_value`` for content that later requests send again.

        A placeholder is restored only where those requests mask its original again:
        inside a detection of the restored text, or standing alone when the original
        propagates. Any other placeholder-shaped token stays as written, since its
        original would be sent in the clear.
        """
        return self._unmask_walk(value, self._unmask_resent_text, keys=False)

    def _unmask_walk(self, value: Any, restore: Callable[[str], str], *, keys: bool) -> Any:
        if not self._placeholder_map:
            return value
        if isinstance(value, str):
            return restore(value)
        if isinstance(value, BaseModel):
            return type(value).model_validate(
                self._unmask_walk(value.model_dump(), restore, keys=keys)
            )
        if isinstance(value, dict):
            return {
                (restore(k) if keys and isinstance(k, str) else k): self._unmask_walk(
                    v, restore, keys=keys
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [self._unmask_walk(v, restore, keys=keys) for v in value]
        if isinstance(value, tuple):
            return tuple(self._unmask_walk(v, restore, keys=keys) for v in value)
        return value

    def _unmask_resent_text(self, text: str) -> str:
        if "<" not in text:
            return text
        mapping = self._placeholder_map
        tokens = [
            match for match in _PLACEHOLDER_TOKEN_RE.finditer(text) if match.group(0) in mapping
        ]
        if not tokens:
            return text
        pieces: list[str] = []
        spans: list[tuple[int, int]] = []
        cursor = length = 0
        for token in tokens:
            original = mapping[token.group(0)]
            length += token.start() - cursor
            spans.append((length, length + len(original)))
            pieces += (text[cursor : token.start()], original)
            length += len(original)
            cursor = token.end()
        pieces.append(text[cursor:])
        restored = "".join(pieces)
        detections = find_identifiers(restored, self.policy, self._compiled_extras)
        kept = _masked_again(restored, spans, detections, self._propagating)
        if all(kept):
            return restored
        parts: list[str] = []
        cursor = 0
        for token, keep in zip(tokens, kept, strict=True):
            parts += (
                text[cursor : token.start()],
                mapping[token.group(0)] if keep else token.group(0),
            )
            cursor = token.end()
        parts.append(text[cursor:])
        return "".join(parts)

    def unmask_tracked(self, value: Any) -> tuple[Any, frozenset[str]]:
        """``unmask_value`` that also returns the originals it restored."""
        restored: set[str] = set()
        mapping = self._placeholder_map

        def _restore(match: re.Match[str]) -> str:
            original = mapping.get(match.group(0))
            if original is None:
                return match.group(0)
            restored.add(original)
            return original

        def _walk(item: Any) -> Any:
            if isinstance(item, str):
                return _PLACEHOLDER_TOKEN_RE.sub(_restore, item) if "<" in item else item
            if isinstance(item, dict):
                return {k: _walk(v) for k, v in item.items()}
            if isinstance(item, list):
                return [_walk(v) for v in item]
            if isinstance(item, tuple):
                return tuple(_walk(v) for v in item)
            return item

        if not mapping:
            return value, frozenset()
        return _walk(value), frozenset(restored)

    def promote(self, originals: Iterable[str]) -> None:
        """Replace these known originals in every later request, wherever they appear."""
        self._propagating.update(o for o in originals if o in self._reverse_map)

    def remask_originals(self, text: str, originals: Iterable[str]) -> str:
        """Put the placeholders back for ``originals`` where they stand alone in ``text``."""
        known = [o for o in originals if o in self._reverse_map]
        if not known or not text:
            return text
        pattern = _token_pattern(known)
        return pattern.sub(lambda match: self._reverse_map[match.group(0)], text)

    def cut_masked(
        self, text: str, cut: Callable[[str], str], *, context: Sequence[str] = ()
    ) -> str:
        """``cut`` made on ``text`` masked, with the kept text restored.

        ``text`` comes back whole when ``cut`` keeps it or its masked form whole,
        so a cut can split a placeholder but never an identifier. ``context`` holds
        texts of the same request, masked together with ``text`` so a value they
        label is masked in ``text`` too. Only placeholders this masking wrote are
        restored: a placeholder-shaped token ``text`` already held comes back as
        written. ``cut(text)`` when disabled.
        """
        if not self.policy.enabled:
            return cut(text)
        if cut(text) == text:
            return text
        literals = frozenset(_PLACEHOLDER_SHAPE_RE.findall(text))
        stand_in = _free_character(text) if literals else None
        source = text if stand_in is None else _hide_literals(text, stand_in)
        *_, masked = self.mask_batch([*context, source])
        kept = cut(masked)
        if kept == masked:
            return text
        if stand_in is not None:
            return self.unmask(kept).replace(stand_in, "<")
        if literals:
            # No free stand-in, so a literal and a placeholder written here look alike;
            # neither is restored.
            mapping = {
                token: original
                for token, original in self._placeholder_map.items()
                if token not in literals
            }
            return _PLACEHOLDER_TOKEN_RE.sub(
                lambda match: mapping.get(match.group(0), match.group(0)), kept
            )
        return self.unmask(kept)

    def mask_batch(
        self,
        values: Sequence[Any],
        *,
        protected: Sequence[str] = (),
        protect: Sequence[bool] = (),
    ) -> list[Any]:
        """Mask the texts and values of one request together.

        Detection runs over every value first, in order, so placeholders number
        in that order and a value whose class rises anywhere in the request
        propagates into all of it; then each value is rendered with its
        detections replaced and every propagating original replaced where it
        stands alone. ``protect[i]`` keeps each occurrence of a ``protected``
        text inside ``values[i]`` as written. Identity when disabled.
        """
        if not self.policy.enabled:
            return list(values)
        authored = tuple(text for text in protected if text)
        flags = [bool(protect[i]) if i < len(protect) else False for i in range(len(values))]
        plans: dict[tuple[str, bool], _Plan] = {}
        try:
            for value, flag in zip(values, flags, strict=True):
                self._walk_detect(value, authored if flag else (), flag, plans)
            render = self._renderer()
            return [
                self._walk_render(value, flag, plans, render)
                for value, flag in zip(values, flags, strict=True)
            ]
        except Exception as exc:
            # The failure's own message may quote the text being masked.
            raise MaskingError(f"masking failed: {type(exc).__name__}") from exc

    def _walk_detect(
        self,
        value: Any,
        authored: tuple[str, ...],
        flag: bool,
        plans: dict[tuple[str, bool], _Plan],
    ) -> None:
        if isinstance(value, str):
            key = (value, flag)
            if key not in plans:
                plans[key] = [
                    (start, end, self._detect(value[start:end]))
                    for start, end in _maskable_segments(value, authored)
                ]
        elif isinstance(value, dict):
            for item_key, item in value.items():
                kind = self._structured_kind(item_key, item)
                if kind is not None:
                    self._ensure_placeholder(kind, item)
                    self._propagating.add(item)
                else:
                    self._walk_detect(item, authored, flag, plans)
        elif isinstance(value, list | tuple):
            for item in value:
                self._walk_detect(item, authored, flag, plans)

    def _walk_render(
        self,
        value: Any,
        flag: bool,
        plans: dict[tuple[str, bool], _Plan],
        render: Callable[[str], str],
    ) -> Any:
        if isinstance(value, str):
            return self._render_text(value, plans[(value, flag)], render)
        if isinstance(value, dict):
            return {
                item_key: (
                    self._reverse_map[item]
                    if self._structured_kind(item_key, item) is not None
                    else self._walk_render(item, flag, plans, render)
                )
                for item_key, item in value.items()
            }
        if isinstance(value, list):
            return [self._walk_render(item, flag, plans, render) for item in value]
        if isinstance(value, tuple):
            return tuple(self._walk_render(item, flag, plans, render) for item in value)
        return value

    def _render_text(self, text: str, plan: _Plan, render: Callable[[str], str]) -> str:
        parts: list[str] = []
        cursor = 0
        for start, end, matches in plan:
            parts.append(text[cursor:start])
            segment = text[start:end]
            position = 0
            for match in matches:
                parts.append(render(segment[position : match.start]))
                parts.append(self._reverse_map[match.value])
                position = match.end
            parts.append(render(segment[position:]))
            cursor = end
        parts.append(text[cursor:])
        return "".join(parts)

    def _renderer(self) -> Callable[[str], str]:
        """Replace every propagating original in a text no detection covered."""
        if len(self._propagating) != self._propagation_size:
            self._propagation_re = _token_pattern(self._propagating) if self._propagating else None
            self._propagation_size = len(self._propagating)
        pattern = self._propagation_re
        if pattern is None:
            return _identity
        reverse = self._reverse_map

        def _render(text: str) -> str:
            return pattern.sub(lambda match: reverse[match.group(0)], text) if text else text

        return _render

    def to_state(self) -> dict[str, str]:
        """Return the placeholder map in a form suitable for state storage."""
        return dict(self._placeholder_map)


def _identity(text: str) -> str:
    return text


def _free_character(text: str) -> str | None:
    """A private-use character ``text`` lacks, to stand in for a literal placeholder's ``<``."""
    used = set(text)
    return next((chr(code) for code in _PRIVATE_USE_AREA if chr(code) not in used), None)


def _hide_literals(text: str, stand_in: str) -> str:
    """``text`` with each placeholder-shaped token's ``<`` written as ``stand_in``."""
    return _PLACEHOLDER_SHAPE_RE.sub(lambda match: stand_in + match.group(0)[1:], text)


def _masked_again(
    text: str,
    spans: Sequence[tuple[int, int]],
    detections: Sequence[DetectedIdentifier],
    propagating: set[str],
) -> list[bool]:
    """Whether masking ``text`` replaces each span, by a detection or by propagation.

    ``spans`` and ``detections`` are in start order; a span a detection only partly
    covers is not replaced whole.
    """
    replaced: list[bool] = []
    index = 0
    for start, end in spans:
        while index < len(detections) and detections[index].end <= start:
            index += 1
        if index < len(detections) and detections[index].start < end:
            match = detections[index]
            replaced.append(match.start <= start and end <= match.end)
            continue
        alone = not (start and _TOKEN_CHAR_RE.match(text, start - 1)) and not (
            _TOKEN_CHAR_RE.match(text, end)
        )
        replaced.append(alone and text[start:end] in propagating)
    return replaced


def _distinct_shape(value: str) -> bool:
    return any(char.isalpha() for char in value) and any(
        char.isdigit() or char in _SHAPE_MARKS for char in value
    )


def _token_pattern(originals: Iterable[str]) -> re.Pattern[str]:
    """Match any of ``originals`` where it stands alone, longest first."""
    alternatives = "|".join(re.escape(o) for o in sorted(originals, key=len, reverse=True))
    return re.compile(rf"(?<![{_TOKEN_CHARS}])(?:{alternatives})(?![{_TOKEN_CHARS}])")


def _maskable_segments(text: str, authored: tuple[str, ...]) -> list[tuple[int, int]]:
    """The spans of ``text`` outside every occurrence of an authored text."""
    protected: list[tuple[int, int]] = []
    for item in authored:
        start = text.find(item)
        while start != -1:
            protected.append((start, start + len(item)))
            start = text.find(item, start + len(item))
    if not protected:
        return [(0, len(text))]
    segments: list[tuple[int, int]] = []
    cursor = 0
    for start, end in sorted(protected):
        if start > cursor:
            segments.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < len(text):
        segments.append((cursor, len(text)))
    return segments


__all__ = ["MaskingError", "MaskingRules"]
