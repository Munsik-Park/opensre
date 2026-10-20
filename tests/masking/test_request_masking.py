"""The agent loop masks provider requests and restores what the model returns."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from core.agent import Agent
from core.llm.types import AgentLLMResponse, ToolCall
from core.tool.contracts import AgentTool, AgentToolContext

POD = "privacy-test-7b5b4cbd55-abcde"
IP = "192.0.2.42"
EMAIL = "privacy-check@example.invalid"
PROMPT = f"Investigate pod {POD} in namespace privacy-test. Example IP: {IP}. Contact: {EMAIL}"
_POD_TOKEN_RE = re.compile(r"pod (<POD_\d+>)")

Step = Callable[[list[dict[str, Any]]], AgentLLMResponse]


class _ScriptedLLM:
    """A generic agent client that answers from a script and records every request."""

    model_id = None

    def __init__(self, steps: Sequence[Step]) -> None:
        self._steps = list(steps)
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def tool_schemas(self, tools: list[Any]) -> list[dict[str, Any]]:
        return [{"name": tool.name, "description": tool.description} for tool in tools]

    def invoke(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentLLMResponse:
        request = json.loads(
            json.dumps({"system": system, "messages": messages, "tools": tools}, default=str)
        )
        with self._lock:
            self.requests.append(request)
            step = self._steps.pop(0) if self._steps else _final("Done.")
        return step(request["messages"])

    def build_assistant_message(self, content: str, tool_calls: list[ToolCall]) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [{"id": tc.id, "name": tc.name, "input": tc.input} for tc in tool_calls],
        }

    def build_tool_result_message(
        self, tool_calls: list[ToolCall], results: list[Any]
    ) -> dict[str, Any]:
        return {
            "role": "tool",
            "results": [{"id": tc.id, "output": out} for tc, out in zip(tool_calls, results)],
        }


def _final(text: str) -> Step:
    def _step(_messages: list[dict[str, Any]]) -> AgentLLMResponse:
        return AgentLLMResponse(content=text)

    return _step


def _pod_token(messages: list[dict[str, Any]]) -> str:
    """The placeholder the request gave the pod; the predicted token when it gave none."""
    match = _POD_TOKEN_RE.search(json.dumps(messages))
    return match.group(1) if match else "<POD_0>"


def _call_lookup(messages: list[dict[str, Any]]) -> AgentLLMResponse:
    token = _pod_token(messages)
    return AgentLLMResponse(
        content="",
        tool_calls=[
            ToolCall(
                id="call_1", name="lookup", input={"query": f"kubectl get pod {token} -o json"}
            )
        ],
    )


def _reply_with_pod(messages: list[dict[str, Any]]) -> AgentLLMResponse:
    return AgentLLMResponse(content=f"Pod {_pod_token(messages)} is healthy.")


def _lookup_tool(calls: list[dict[str, Any]], result: Any) -> AgentTool:
    def _execute(args: dict[str, Any], _context: AgentToolContext) -> Any:
        calls.append(dict(args))
        return result

    return AgentTool(
        name="lookup",
        description="Look up a pod.",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        execute=_execute,
    )


def _run(llm: _ScriptedLLM, tool: AgentTool, prompt: str = PROMPT) -> Any:
    agent = Agent(llm=llm, system="You are an SRE agent.", tools=[tool], max_iterations=4)
    return agent.run([{"role": "user", "content": prompt}])


@pytest.fixture
def masking_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSRE_MASK_ENABLED", "true")
    monkeypatch.delenv("OPENSRE_MASK_KINDS", raising=False)
    monkeypatch.delenv("OPENSRE_MASK_EXTRA_REGEX", raising=False)


@pytest.mark.usefixtures("masking_on")
def test_every_request_carries_placeholders_instead_of_identifiers() -> None:
    calls: list[dict[str, Any]] = []
    tool = _lookup_tool(calls, {"namespace": "privacy-test", "pod": POD, "node_ip": IP})
    llm = _ScriptedLLM([_call_lookup, _reply_with_pod])

    _run(llm, tool)

    assert len(llm.requests) == 2
    for request in llm.requests:
        sent = json.dumps(request)
        for original in (POD, "privacy-test", IP, EMAIL):
            assert original not in sent, f"{original!r} reached the provider: {sent}"
    assert "<POD_0>" in json.dumps(llm.requests[1]["messages"][-1])


@pytest.mark.usefixtures("masking_on")
def test_tool_runs_with_restored_arguments_and_the_next_request_keeps_placeholders() -> None:
    calls: list[dict[str, Any]] = []
    tool = _lookup_tool(calls, "ok")
    llm = _ScriptedLLM([_call_lookup, _reply_with_pod])

    result = _run(llm, tool)

    assert calls == [{"query": f"kubectl get pod {POD} -o json"}]
    replayed = json.dumps(llm.requests[1]["messages"])
    assert "kubectl get pod <POD_0> -o json" in replayed
    assert result.final_text == f"Pod {POD} is healthy."
    for message in result.messages:
        assert "<POD_0>" not in str(getattr(message, "content", "")), message


def test_masking_off_sends_the_request_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENSRE_MASK_ENABLED", raising=False)
    calls: list[dict[str, Any]] = []
    llm = _ScriptedLLM([_call_lookup, _final("ok")])

    _run(llm, _lookup_tool(calls, "ok"))

    assert PROMPT in json.dumps(llm.requests[0]["messages"])
    assert calls == [{"query": "kubectl get pod <POD_0> -o json"}]


@pytest.mark.usefixtures("masking_on")
def test_invalid_setting_sends_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENSRE_MASK_KINDS", "pod,namspace")
    llm = _ScriptedLLM([_final("never")])

    with pytest.raises(ValueError, match="OPENSRE_MASK_KINDS"):
        _run(llm, _lookup_tool([], "ok"))

    assert llm.requests == []


@pytest.mark.usefixtures("masking_on")
def test_concurrent_sessions_on_one_client_never_share_a_mapping() -> None:
    pods = {"a": "alpha-api-7b5b4cbd55-aaaaa", "b": "bravo-db-6c4d8f9e21-bbbbb"}
    barrier = threading.Barrier(2, timeout=10)
    calls: dict[str, list[dict[str, Any]]] = {"a": [], "b": []}

    def _call_after_barrier(messages: list[dict[str, Any]]) -> AgentLLMResponse:
        barrier.wait()
        return _call_lookup(messages)

    llm = _ScriptedLLM([_call_after_barrier, _call_after_barrier])

    def _session(name: str) -> None:
        _run(llm, _lookup_tool(calls[name], "ok"), prompt=f"Investigate pod {pods[name]} now")

    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in [pool.submit(_session, name) for name in pods]:
            future.result(timeout=30)

    assert calls["a"] == [{"query": f"kubectl get pod {pods['a']} -o json"}]
    assert calls["b"] == [{"query": f"kubectl get pod {pods['b']} -o json"}]
    for request in llm.requests:
        sent = json.dumps(request)
        assert pods["a"] not in sent and pods["b"] not in sent, sent
