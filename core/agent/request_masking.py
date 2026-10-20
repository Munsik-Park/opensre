"""Masking of one ReAct run's provider requests and restoration of its responses.

A run keeps two forms of its exchange. Observers (events, tools, hooks, history,
evidence) read restored content, tool calls and tool results. Each request is
masked again from those originals under the turn's mapping, except an assistant
payload this run built from a provider response, which goes back exactly as the
provider wrote it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from functools import cached_property
from typing import Any

from core.llm.types import AgentLLMResponse, ToolCall
from core.messages import (
    AppRuntimeMessage,
    AssistantRuntimeMessage,
    MessageMapper,
    ProviderMessage,
    RuntimeContent,
    RuntimeMessage,
    ToolResultRuntimeMessage,
    UserRuntimeMessage,
)
from core.tool.execution import ToolExecutionResult
from infrastructure.safety.masking import MaskingRules, MaskingScope

_MODEL_ONLY_INDICES_KEY = "model_only_result_indices"

type _Build[T] = Callable[[Iterator[Any]], T]


class _Slots:
    """The values of one request in masking order, each with its protection flag."""

    def __init__(self) -> None:
        self.values: list[Any] = []
        self.protect: list[bool] = []

    def add(self, value: Any, *, protect: bool = False) -> None:
        self.values.append(value)
        self.protect.append(protect)


class RunMasking:
    """The masking step of one ReAct run, over the active turn scope's mapping."""

    def __init__(
        self,
        scope: MaskingScope,
        mapper: MessageMapper,
        authored_texts: Sequence[str],
    ) -> None:
        self._scope = scope
        self._mapper = mapper
        self._authored_texts = tuple(authored_texts)
        # Messages this run built; kept alive so their ids stay unique.
        self._own: dict[int, RuntimeMessage] = {}
        # Per tool-call id, the originals restored into its arguments.
        self._echoes: dict[str, frozenset[str]] = {}
        self._payload: ProviderMessage | None = None

    @cached_property
    def active(self) -> bool:
        """Whether masking is on; raises ``MaskingPolicyError`` when an enabled setting is invalid."""
        return self._scope.rules.policy.enabled

    @property
    def _rules(self) -> MaskingRules:
        return self._scope.rules

    def mask_request(
        self, system: str, messages: Sequence[RuntimeMessage]
    ) -> tuple[str, list[RuntimeMessage]]:
        """The system text and messages of one request, masked from their originals."""
        slots = _Slots()
        slots.add(system, protect=True)
        builds = [self._plan(message, slots) for message in messages]
        masked = iter(
            self._rules.mask_batch(
                slots.values, protected=self._authored_texts, protect=slots.protect
            )
        )
        masked_system = next(masked)
        return masked_system, [build(masked) for build in builds]

    def restore_response(self, response: AgentLLMResponse) -> AgentLLMResponse:
        """The response with its content and tool-call arguments restored.

        The assistant payload is built first, from the response as received, for
        :meth:`assistant_message`.
        """
        self._payload = self._mapper.to_assistant_provider_message(response)
        calls: list[ToolCall] = []
        for call in response.tool_calls:
            restored, originals = self._rules.unmask_tracked(call.input)
            self._echoes[call.id] = originals
            self._rules.promote(originals)
            calls.append(ToolCall(id=call.id, name=call.name, input=restored))
        return replace(response, content=self._rules.unmask(response.content), tool_calls=calls)

    def assistant_message(self, response: AgentLLMResponse) -> AssistantRuntimeMessage:
        """The in-turn assistant message: restored content and calls, the payload as received."""
        message = AssistantRuntimeMessage(
            content=response.content or "",
            tool_calls=tuple(response.tool_calls),
            provider_payload=self._payload,
        )
        self._payload = None
        self.adopt(message)
        return message

    def adopt(self, message: RuntimeMessage) -> None:
        """Mark a message this run built from restored data."""
        self._own[id(message)] = message

    def restore_result(self, result: ToolExecutionResult) -> ToolExecutionResult:
        """A tool result with each placeholder restored where its string alone is masked again.

        A model-only result is kept as produced: later requests do not mask it again.
        """
        if result.model_only:
            return result
        return replace(
            result,
            content=self._rules.unmask_resent(result.content),
            details=self._rules.unmask_resent(result.details),
        )

    def _plan(self, message: RuntimeMessage, slots: _Slots) -> _Build[RuntimeMessage]:
        if isinstance(message, UserRuntimeMessage):
            content = _plan_content(message.content, slots, protect=True)
            return lambda masked: replace(message, content=content(masked))
        if isinstance(message, AssistantRuntimeMessage):
            return self._plan_assistant(message, slots)
        if isinstance(message, ToolResultRuntimeMessage):
            return self._plan_tool_results(message, slots)
        if isinstance(message, AppRuntimeMessage) and message.include_in_context:
            app_content = _plan_content(message.content, slots, protect=False)
            return lambda masked: replace(message, content=app_content(masked))
        return lambda _masked: message

    def _plan_assistant(
        self, message: AssistantRuntimeMessage, slots: _Slots
    ) -> _Build[RuntimeMessage]:
        if id(message) in self._own:
            return lambda _masked: message
        payload = message.provider_payload
        if payload is not None:
            slots.add(payload)
            return lambda masked: replace(message, provider_payload=next(masked))
        content = _plan_content(message.content, slots, protect=False)
        for call in message.tool_calls:
            slots.add(call.input)

        def _build(masked: Iterator[Any]) -> RuntimeMessage:
            masked_content = content(masked)
            calls = tuple(
                ToolCall(id=call.id, name=call.name, input=next(masked))
                for call in message.tool_calls
            )
            return replace(message, content=masked_content, tool_calls=calls)

        return _build

    def _plan_tool_results(
        self, message: ToolResultRuntimeMessage, slots: _Slots
    ) -> _Build[RuntimeMessage]:
        payloads = message.provider_payloads
        if payloads and id(message) not in self._own:
            for payload in payloads:
                slots.add(payload)
            return lambda masked: replace(
                message, provider_payloads=tuple(next(masked) for _ in payloads)
            )
        model_only = set(message.metadata.get(_MODEL_ONLY_INDICES_KEY, ()))
        for call in message.tool_calls:
            slots.add(call.input)
        plans = [
            None if index in model_only else _plan_content(result, slots, protect=False)
            for index, result in enumerate(message.results)
        ]

        def _build(masked: Iterator[Any]) -> RuntimeMessage:
            calls = tuple(
                ToolCall(id=call.id, name=call.name, input=next(masked))
                for call in message.tool_calls
            )
            results = tuple(
                self._remask_echo(message.tool_calls, index, result)
                if plan is None
                else plan(masked)
                for index, (result, plan) in enumerate(zip(message.results, plans, strict=True))
            )
            # Drop the stored payloads: they hold originals.
            return replace(message, tool_calls=calls, results=results, provider_payloads=())

        return _build

    def _remask_echo(self, calls: Sequence[ToolCall], index: int, result: Any) -> Any:
        """A model-only result as produced, with the call's restored arguments masked again."""
        originals = self._echoes.get(calls[index].id, frozenset()) if index < len(calls) else ()
        if not originals:
            return result
        if isinstance(result, str):
            return self._rules.remask_originals(result, originals)
        if isinstance(result, list):
            return [
                {**block, "text": self._rules.remask_originals(block["text"], originals)}
                if isinstance(block, dict) and isinstance(block.get("text"), str)
                else block
                for block in result
            ]
        return result


def _plan_content(content: RuntimeContent | Any, slots: _Slots, *, protect: bool) -> _Build[Any]:
    """Add the text of ``content`` to ``slots``; non-text blocks pass unchanged."""
    if isinstance(content, str):
        slots.add(content, protect=protect)
        return lambda masked: next(masked)
    if isinstance(content, list):
        builds = [_plan_block(block, slots, protect=protect) for block in content]
        return lambda masked: [build(masked) for build in builds]
    if content is None:
        return lambda _masked: None
    slots.add(content)
    return lambda masked: next(masked)


def _plan_block(block: Any, slots: _Slots, *, protect: bool) -> _Build[Any]:
    if isinstance(block, str):
        slots.add(block, protect=protect)
        return lambda masked: next(masked)
    if (
        isinstance(block, dict)
        and isinstance(block.get("text"), str)
        and block.get("type", "text") == "text"
    ):
        slots.add(block["text"], protect=protect)
        return lambda masked: {**block, "text": next(masked)}
    return lambda _masked: block


__all__ = ["RunMasking"]
