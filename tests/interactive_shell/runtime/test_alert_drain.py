"""Alert drain and watcher: an alert stays queued until recorded and is retried after a delay."""

from __future__ import annotations

import asyncio
import contextlib
import io
import json
import sys
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from http import HTTPStatus
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from rich.console import Console

from core.agent_harness.session.persistence.contracts import SessionPersistenceSource
from core.agent_harness.session.persistence.memory import InMemorySessionStore
from core.domain.alerts import inbox as alert_inbox
from core.domain.alerts.inbox import AlertInbox, IncomingAlert, set_current_inbox
from infrastructure.alert_intake import build_alert_intake_app
from surfaces.interactive_shell.runtime.background import workers
from surfaces.interactive_shell.runtime.background.alert_drain import drain_and_render_incoming
from surfaces.interactive_shell.runtime.background.workers import BackgroundTaskPool
from surfaces.interactive_shell.runtime.core.state import ReplState, SpinnerState
from surfaces.interactive_shell.session import Session

_ALWAYS = sys.maxsize
_WATCHER_WAIT_SECONDS = 10.0


class _FlakyStore(InMemorySessionStore):
    def __init__(self, failures: dict[str, int] | None = None) -> None:
        super().__init__()
        self.failures = dict(failures or {})
        self.requests: list[str] = []

    def append_turn(self, session: SessionPersistenceSource, kind: str, text: str) -> None:
        self.requests.append(text)
        remaining = self.failures.get(text, 0)
        if remaining > 0:
            self.failures[text] = remaining - 1
            raise OSError("session store unavailable")
        super().append_turn(session, kind, text)


class _RaisingConsole(Console):
    """Console whose ``print`` raises OSError, as a closed terminal stream does."""

    def __init__(self, fail_calls: set[int] | None = None) -> None:
        self.output = io.StringIO()
        super().__init__(file=self.output, width=100)
        self.fail_calls = fail_calls
        self.print_calls = 0

    def print(self, *objects: Any, **kwargs: Any) -> None:
        self.print_calls += 1
        if self.fail_calls is None or self.print_calls in self.fail_calls:
            raise OSError("terminal stream closed")
        super().print(*objects, **kwargs)


class _InboxClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _WatcherDrain:
    def __init__(self) -> None:
        self.calls = 0
        self.fail_next = False

    def __call__(self, session: Session, console: Console, inbox: AlertInbox) -> int:
        self.calls += 1
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("injected drain failure")
        return drain_and_render_incoming(session, console, inbox)


@pytest.fixture
def inbox_clock(monkeypatch: pytest.MonkeyPatch) -> _InboxClock:
    clock = _InboxClock()
    monkeypatch.setattr(alert_inbox, "monotonic", clock, raising=False)
    return clock


@pytest.fixture
def watcher_drain(monkeypatch: pytest.MonkeyPatch) -> _WatcherDrain:
    drain = _WatcherDrain()
    monkeypatch.setattr(workers, "drain_and_render_incoming", drain)
    return drain


@pytest.fixture
def intake_inbox(monkeypatch: pytest.MonkeyPatch) -> Iterator[AlertInbox]:
    monkeypatch.delenv("OPENSRE_ALERT_LISTENER_TOKEN", raising=False)
    inbox = AlertInbox(maxsize=2)
    set_current_inbox(inbox)
    yield inbox
    set_current_inbox(None)


def _retry_delay() -> float:
    return alert_inbox._RETRY_DELAY_SECONDS


def _quiet_console() -> Console:
    return Console(file=io.StringIO())


def _same_objects(actual: Sequence[IncomingAlert], expected: Sequence[IncomingAlert]) -> bool:
    return len(actual) == len(expected) and all(a is e for a, e in zip(actual, expected))


def _pool(session: Session, inbox: AlertInbox) -> BackgroundTaskPool:
    return BackgroundTaskPool(session, ReplState(), SpinnerState(), inbox, lambda: None)


async def _wait_for(condition: Callable[[], bool]) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + _WATCHER_WAIT_SECONDS
    while not condition():
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.01)
    return True


@contextlib.asynccontextmanager
async def _running_watcher(
    session: Session, inbox: AlertInbox
) -> AsyncIterator[asyncio.Task[None]]:
    state = ReplState()
    pool = BackgroundTaskPool(session, state, SpinnerState(), inbox, lambda: None)
    task = asyncio.create_task(pool._alert_watcher())
    try:
        yield task
    finally:
        state.exit_requested = True
        inbox.pending_event.set()
        await asyncio.wait({task}, timeout=_WATCHER_WAIT_SECONDS)


def test_alert_whose_record_fails_stays_queued_until_a_retry_records_it(
    inbox_clock: _InboxClock,
) -> None:
    store = _FlakyStore({"disk full": 3})
    session = Session(store=store)
    inbox = AlertInbox()
    alert = IncomingAlert(text="disk full")
    inbox.put(alert)

    for attempt in (1, 2, 3):
        assert drain_and_render_incoming(session, _quiet_console(), inbox) == 0
        assert _same_objects(inbox.peek_last(5), [alert]), (
            f"alert left inbox after attempt {attempt}"
        )
        assert session.alerts.entries == []
        inbox_clock.now += _retry_delay() + 0.01

    assert drain_and_render_incoming(session, _quiet_console(), inbox) == 1
    assert store.requests == ["disk full"] * 4
    assert _same_objects(session.alerts.entries, [alert])
    assert inbox.qsize == 0


def test_alert_waiting_for_retry_counts_against_the_inbox_bound(intake_inbox: AlertInbox) -> None:
    client = TestClient(
        build_alert_intake_app(), client=("127.0.0.1", 40000), base_url="http://127.0.0.1"
    )
    store = _FlakyStore({"oldest": _ALWAYS})
    pool = _pool(Session(store=store), intake_inbox)

    assert client.post("/alerts", json={"text": "oldest"}).status_code == HTTPStatus.ACCEPTED
    pool.drain_turn_start_output(_quiet_console())
    assert store.requests == ["oldest"]

    second = client.post("/alerts", json={"text": "second"})
    assert second.json() == {"queued": True, "queue_depth": 2}, "retained alert not counted"
    third = client.post("/alerts", json={"text": "third"})
    assert third.status_code == HTTPStatus.ACCEPTED
    assert third.json() == {
        "queued": True,
        "queue_depth": 2,
        "dropped": 1,
        "warning": "inbox full, oldest alert dropped",
    }
    assert [alert.text for alert in intake_inbox.peek_last(5)] == ["second", "third"]


def test_failed_alert_is_not_retried_by_either_drain_before_the_delay(
    inbox_clock: _InboxClock,
) -> None:
    store = _FlakyStore({"cpu": _ALWAYS})
    session = Session(store=store)
    inbox = AlertInbox()
    inbox.put(IncomingAlert(text="cpu"))
    pool = _pool(session, inbox)

    def drain_function() -> None:
        drain_and_render_incoming(session, _quiet_console(), inbox)

    def turn_start_drain() -> None:
        pool.drain_turn_start_output(_quiet_console())

    turn_start_drain()
    assert store.requests == ["cpu"]
    assert inbox.qsize == 1, "the failed alert must wait in the inbox for its retry"

    for retrying_drain in (drain_function, turn_start_drain):
        attempts = len(store.requests)
        failed_at = inbox_clock.now
        for now in (failed_at, failed_at + _retry_delay() - 0.01):
            inbox_clock.now = now
            turn_start_drain()
            drain_function()
            assert len(store.requests) == attempts, f"retried {now - failed_at:.2f}s after failing"
        inbox_clock.now = failed_at + _retry_delay() + 0.01
        retrying_drain()
        assert len(store.requests) == attempts + 1, f"{retrying_drain.__name__} did not retry"


def test_alert_is_recorded_once_however_many_attempts_it_takes(inbox_clock: _InboxClock) -> None:
    store = _FlakyStore({"oom": 2})
    session = Session(store=store)
    inbox = AlertInbox()
    inbox.put(IncomingAlert(text="oom"))

    def history_alerts_requests() -> tuple[int, int, int]:
        history = sum(1 for entry in session.history if entry.get("type") == "incoming_alert")
        return history, len(session.alerts.entries), len(store.requests)

    expected_after_each_drain = [(0, 0, 1), (0, 0, 2), (1, 1, 3), (1, 1, 3), (1, 1, 3)]
    for drain_number, expected in enumerate(expected_after_each_drain, start=1):
        if drain_number > 1:
            inbox_clock.now += _retry_delay() + 0.01
        drain_and_render_incoming(session, _quiet_console(), inbox)
        assert history_alerts_requests() == expected, f"after drain {drain_number}"


def test_alerts_with_identical_payloads_are_tracked_separately(inbox_clock: _InboxClock) -> None:
    payload = {"text": "same body", "received_at": "2026-10-21T05:00:00+00:00"}
    first = IncomingAlert.model_validate(payload)
    second = IncomingAlert.model_validate(payload)
    assert first == second
    store = _FlakyStore({"same body": 1})
    session = Session(store=store)
    inbox = AlertInbox()
    inbox.put(first)
    inbox.put(second)

    drain_and_render_incoming(session, _quiet_console(), inbox)
    assert _same_objects(session.alerts.entries, [second])
    assert _same_objects(inbox.peek_last(5), [first])

    failed_at = inbox_clock.now
    inbox_clock.now = failed_at + _retry_delay() - 0.01
    drain_and_render_incoming(session, _quiet_console(), inbox)
    assert _same_objects(session.alerts.entries, [second]), "an outcome reached the twin"
    assert _same_objects(inbox.peek_last(5), [first]), "an outcome reached the twin"
    assert len(store.requests) == 2

    inbox_clock.now = failed_at + _retry_delay() + 0.01
    drain_and_render_incoming(session, _quiet_console(), inbox)
    assert _same_objects(session.alerts.entries, [second, first])
    assert inbox.qsize == 0
    assert len(store.requests) == 3


def test_render_failure_leaves_the_alert_recorded_once(inbox_clock: _InboxClock) -> None:
    store = _FlakyStore()
    session = Session(store=store)
    inbox = AlertInbox()
    alert = IncomingAlert(text="latency")
    inbox.put(alert)
    console = _RaisingConsole()

    assert drain_and_render_incoming(session, console, inbox) == 1
    assert console.print_calls == 1
    assert _same_objects(session.alerts.entries, [alert])
    assert inbox.qsize == 0

    for _ in range(2):
        inbox_clock.now += _retry_delay() + 0.01
        drain_and_render_incoming(session, console, inbox)
    assert store.requests == ["latency"], "a rendered-but-failed alert was persisted again"
    assert len(session.alerts.entries) == 1


def test_payload_no_stream_can_encode_is_recorded_once_and_not_retried(
    inbox_clock: _InboxClock, tmp_path: Path
) -> None:
    alert = IncomingAlert.model_validate(json.loads('{"text": "bad \\ud800 char"}'))
    store = _FlakyStore()
    session = Session(store=store)
    inbox = AlertInbox()
    inbox.put(alert)

    with (tmp_path / "terminal.txt").open("w", encoding="utf-8") as stream:
        console = Console(file=stream, force_terminal=True, highlight=False)
        drain_and_render_incoming(session, console, inbox)
        inbox_clock.now += _retry_delay() + 0.01
        drain_and_render_incoming(session, console, inbox)

    assert store.requests == [alert.text]
    assert _same_objects(session.alerts.entries, [alert])
    assert inbox.qsize == 0


def test_one_failing_alert_does_not_stop_the_rest_of_the_drain() -> None:
    store = _FlakyStore({"second": _ALWAYS})
    session = Session(store=store)
    inbox = AlertInbox()
    first, second, third = (IncomingAlert(text=text) for text in ("first", "second", "third"))
    for alert in (first, second, third):
        inbox.put(alert)
    console = _RaisingConsole(fail_calls={1})

    assert drain_and_render_incoming(session, console, inbox) == 2
    assert store.requests == ["first", "second", "third"]
    assert _same_objects(session.alerts.entries, [first, third])
    assert _same_objects(inbox.peek_last(5), [second])
    assert "third" in console.output.getvalue()


@pytest.mark.asyncio
async def test_watcher_does_not_spin_while_an_alert_waits_for_its_retry(
    watcher_drain: _WatcherDrain,
) -> None:
    store = _FlakyStore({"stuck": _ALWAYS})
    session = Session(store=store)
    inbox = AlertInbox()
    inbox.put(IncomingAlert(text="stuck"))

    async with _running_watcher(session, inbox) as task:
        assert await _wait_for(lambda: watcher_drain.calls >= 1)
        assert store.requests == ["stuck"]
        assert inbox.qsize == 1, "precondition: the failed alert is still queued"
        assert not task.done(), "precondition: the watcher is still running"

        drains_before = watcher_drain.calls
        await asyncio.sleep(0.6)
        assert watcher_drain.calls - drains_before <= 2, "the watcher busy-loops"
        assert not task.done()


@pytest.mark.asyncio
async def test_watcher_retries_a_failed_alert_after_the_delay_without_a_new_arrival(
    watcher_drain: _WatcherDrain, inbox_clock: _InboxClock
) -> None:
    store = _FlakyStore({"pager": 1})
    session = Session(store=store)
    inbox = AlertInbox()
    alert = IncomingAlert(text="pager")
    inbox.put(alert)

    async with _running_watcher(session, inbox) as task:
        assert await _wait_for(lambda: watcher_drain.calls >= 2), "no timeout pass after a failure"
        assert store.requests == ["pager"], "the timeout pass retried before the delay"
        assert inbox.qsize == 1

        inbox_clock.now += _retry_delay() + 0.01
        recorded = await _wait_for(lambda: _same_objects(session.alerts.entries, [alert]))
        assert recorded, "the watcher did not retry after the delay"
        assert store.requests == ["pager", "pager"]
        assert inbox.qsize == 0
        assert not task.done()


@pytest.mark.asyncio
async def test_watcher_survives_a_failing_drain_on_its_first_and_a_later_pass(
    watcher_drain: _WatcherDrain,
) -> None:
    watcher_drain.fail_next = True
    session = Session(store=_FlakyStore())
    inbox = AlertInbox()

    async with _running_watcher(session, inbox) as task:
        survived_first = await _wait_for(lambda: watcher_drain.calls >= 2)
        assert survived_first, "the watcher stopped after its first drain failed"
        assert not task.done()

        watcher_drain.fail_next = True
        alert = IncomingAlert(text="woken by arrival")
        inbox.put(alert)
        recorded = await _wait_for(lambda: _same_objects(session.alerts.entries, [alert]))
        assert recorded, "the watcher stopped draining after a later drain failed"
        assert not watcher_drain.fail_next
        assert not task.done()
