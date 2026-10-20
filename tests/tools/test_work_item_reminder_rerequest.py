"""Re-requesting a work-item reminder: which row stays enabled, what is delivered, what is reported."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from config.constants import OPENSRE_WORK_ITEMS_DIR_ENV
from config.constants.work_items import WORK_ITEM_REMINDER_RUN_AT_PARAM
from core.domain.work_items import WorkItemChannelTarget, complete_work_items, make_work_item
from infrastructure.scheduling.scheduler import delivery_bundle
from infrastructure.scheduling.scheduler.loops import set_loop_enabled
from infrastructure.scheduling.scheduler.runner import _scheduled_job, run_task_now
from infrastructure.scheduling.scheduler.storage import (
    add_task,
    get_task,
    list_tasks,
    try_queue_run,
)
from infrastructure.scheduling.scheduler.types import Provider, ScheduledTask, TaskKind, TaskReport
from tests.scheduler._bundle import AgentPayload, runners_with_agent
from tools.system.work_items.reminders import schedule_item_reminder
from tools.system.work_items.tool import work_task_add, work_task_complete, work_task_update

_RUN_AT = "2027-08-24T10:00:00Z"
_RUN_AT_TIME = datetime(2027, 8, 24, 10, 0, tzinfo=UTC)
_FIRE_TIME = "2027-08-24T10:00:00Z"
_SEOUL_RUN_AT = "2027-08-24T19:00:00"
_BOTH_TARGETS = frozenset({("telegram", "T1"), ("slack", "C1")})
_NOTHING_SENT: dict[str, list[str]] = {"slack": [], "telegram": []}
_SENT_ONCE: dict[str, list[str]] = {"slack": ["C1"], "telegram": ["T1"]}
_C999 = [WorkItemChannelTarget(provider="slack", chat_id="C999")]


def _no_agent_turn(payload: AgentPayload) -> TaskReport:
    raise AssertionError(f"a work-item reminder must not start an agent turn: {payload}")


_RUNNERS = runners_with_agent(_no_agent_turn)


@dataclass
class _FakeChannel:
    ok: bool = True
    sent: list[str] = field(default_factory=list)

    def deliver(self, task: ScheduledTask, message: str) -> tuple[bool, str, str]:
        del message
        self.sent.append(task.chat_id)
        return (True, "", "posted") if self.ok else (False, "unavailable", "")


@dataclass
class _Channels:
    slack: _FakeChannel = field(default_factory=_FakeChannel)
    telegram: _FakeChannel = field(default_factory=_FakeChannel)

    def sent(self) -> dict[str, list[str]]:
        return {"slack": list(self.slack.sent), "telegram": list(self.telegram.sent)}

    def clear(self) -> None:
        self.slack.sent.clear()
        self.telegram.sent.clear()


def _use_stores(directory: Path, monkeypatch: pytest.MonkeyPatch) -> _Channels:
    # The run database must sit beside the task store: disabling a row completes its
    # queued ticks in the paired database, while the executor reads the default one.
    directory.mkdir(parents=True)
    monkeypatch.setattr(
        "infrastructure.scheduling.scheduler.storage.task_store.default_task_store_path",
        lambda: directory / "scheduler_tasks.json",
    )
    monkeypatch.setattr(
        "infrastructure.scheduling.scheduler.storage.database.default_run_database_path",
        lambda: directory / "scheduler.db",
    )
    monkeypatch.setenv(OPENSRE_WORK_ITEMS_DIR_ENV, str(directory / "work"))
    channels = _Channels()
    delivery_bundle.ScheduledDeliveryAdapters(
        {Provider.SLACK: channels.slack, Provider.TELEGRAM: channels.telegram}
    ).install()
    return channels


@pytest.fixture()
def channels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Channels:
    monkeypatch.setattr(
        "infrastructure.scheduling.scheduler.reload_signal.request_scheduler_reload",
        lambda: None,
    )
    monkeypatch.setattr(delivery_bundle, "_installed", None)
    return _use_stores(tmp_path / "stores", monkeypatch)


def _add_item(**request: str) -> tuple[str, str]:
    """Create an item reminding Telegram T1 then Slack C1; return its id and its reminder's id."""
    arguments: dict[str, Any] = {
        "title": "Check clusters",
        "remind_at": _RUN_AT,
        "timezone": "UTC",
        "channel_provider": "slack",
        "channel_id": "C1",
        "channel_targets": [{"provider": "telegram", "chat_id": "T1"}],
        **request,
    }
    added = work_task_add(**arguments)
    assert "error" not in added, added
    return added["task"]["id"], added["scheduled_task_id"]


def _rerequest(item_id: str, **changes: str) -> dict[str, Any]:
    arguments: dict[str, Any] = {"notes": "re-saved", "remind_at": _RUN_AT, **changes}
    payload = work_task_update(selector=item_id, **arguments)
    assert "error" not in payload, payload
    return payload


def _reminders(item_id: str) -> list[ScheduledTask]:
    return [task for task in list_tasks() if task.params.get("work_item_id") == item_id]


def _rows(item_id: str) -> list[tuple[str, bool]]:
    return [(task.id, task.enabled) for task in _reminders(item_id)]


def _targets(task: ScheduledTask) -> frozenset[tuple[str, str]]:
    entries = json.loads(task.params["delivery_targets"])
    return frozenset((entry["provider"], entry["chat_id"]) for entry in entries)


def _tick(task_id: str) -> None:
    _scheduled_job(task_id, _RUNNERS, scheduled_run_time=_RUN_AT_TIME)


def _tick_every_enabled_reminder(item_id: str) -> None:
    for task in _reminders(item_id):
        if task.enabled:
            _tick(task.id)


def _assert_one_enabled_reminder(item_id: str, reported_id: str) -> None:
    enabled = [task for task in _reminders(item_id) if task.enabled]
    assert [task.id for task in enabled] == [reported_id], (
        "the reported reminder must be the item's only enabled one"
    )
    assert datetime.fromisoformat(enabled[0].params[WORK_ITEM_REMINDER_RUN_AT_PARAM]) == (
        _RUN_AT_TIME
    )
    assert _targets(enabled[0]) == _BOTH_TARGETS


def test_the_same_reminder_requested_again_stays_enabled(channels: _Channels) -> None:
    item = make_work_item(title="Check clusters", remind_at=_RUN_AT)
    schedule_item_reminder(item, targets=_C999, timezone="UTC")

    renamed = dataclasses.replace(item, title="Check clusters v2")
    scheduled = schedule_item_reminder(renamed, targets=_C999, timezone="UTC")

    assert scheduled is not None
    assert scheduled.task.enabled is True
    assert [task.enabled for task in list_tasks()] == [True]
    assert scheduled.task.id == list_tasks()[0].id


@pytest.mark.parametrize(
    ("other_time", "other_targets"),
    [
        pytest.param("2027-08-24T14:00:00Z", _C999, id="time"),
        pytest.param(_RUN_AT, [WorkItemChannelTarget(provider="slack", chat_id="C2")], id="target"),
    ],
)
def test_a_reminder_changed_and_changed_back_is_the_one_enabled_reminder(
    channels: _Channels, other_time: str, other_targets: list[WorkItemChannelTarget]
) -> None:
    item = make_work_item(title="Check clusters", remind_at=_RUN_AT)
    schedule_item_reminder(item, targets=_C999, timezone="UTC")
    changed = dataclasses.replace(item, remind_at=other_time)
    schedule_item_reminder(changed, targets=other_targets, timezone="UTC")

    scheduled = schedule_item_reminder(item, targets=_C999, timezone="UTC")

    assert scheduled is not None
    assert scheduled.task.enabled is True
    assert [task.id for task in list_tasks() if task.enabled] == [scheduled.task.id]
    assert scheduled.task.params[WORK_ITEM_REMINDER_RUN_AT_PARAM] == "2027-08-24T10:00:00+00:00"
    assert _targets(scheduled.task) == {("slack", "C999")}


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"notes": "canary passed"}, id="notes-changed"),
        pytest.param({"timezone": "Asia/Kolkata"}, id="offset-aware-time-other-timezone"),
    ],
)
def test_an_update_with_the_same_remind_at_reports_the_enabled_reminder(
    channels: _Channels, changes: dict[str, str]
) -> None:
    item_id, _task_id = _add_item()

    payload = _rerequest(item_id, **changes)

    _assert_one_enabled_reminder(item_id, payload.get("scheduled_task_id", ""))
    assert "reminder_status" not in payload, "a pending reminder is not reported as delivered"


def test_reopening_a_completed_item_with_the_same_remind_at_reports_the_enabled_reminder(
    channels: _Channels,
) -> None:
    item_id, _task_id = _add_item()
    assert "error" not in work_task_complete(selectors=[item_id])

    payload = _rerequest(item_id, status="open")

    _assert_one_enabled_reminder(item_id, payload.get("scheduled_task_id", ""))


def test_a_stopped_reminder_requested_again_is_enabled_and_delivered_at_its_time(
    channels: _Channels,
) -> None:
    item_id, task_id = _add_item()
    _mutation, error = set_loop_enabled(task_id, enabled=False)
    assert error == ""

    payload = _rerequest(item_id)

    _assert_one_enabled_reminder(item_id, payload.get("scheduled_task_id", ""))
    _tick(payload["scheduled_task_id"])
    assert channels.sent() == _SENT_ONCE


def _fail_at_the_tick(channels: _Channels, item_id: str, task_id: str) -> None:
    del item_id
    channels.slack.ok = channels.telegram.ok = False
    _tick(task_id)
    channels.slack.ok = channels.telegram.ok = True


def _skip_a_queued_tick(channels: _Channels, item_id: str, task_id: str) -> None:
    del channels, item_id
    assert try_queue_run(task_id, _FIRE_TIME)
    _mutation, error = set_loop_enabled(task_id, enabled=False)
    assert error == ""


def _complete_from_the_cli_then_tick(channels: _Channels, item_id: str, task_id: str) -> None:
    del channels
    assert complete_work_items([item_id]).completed
    _tick(task_id)


@pytest.mark.parametrize(
    ("spend", "reopen"),
    [
        pytest.param(_fail_at_the_tick, {}, id="every-target-failed"),
        pytest.param(_skip_a_queued_tick, {}, id="queued-tick-skipped-by-loops-stop"),
        pytest.param(
            _complete_from_the_cli_then_tick, {"status": "open"}, id="quiet-tick-after-completion"
        ),
    ],
)
def test_an_undelivered_reminder_whose_fire_time_was_spent_is_delivered_at_its_time(
    channels: _Channels,
    spend: Callable[[_Channels, str, str], None],
    reopen: dict[str, str],
) -> None:
    item_id, task_id = _add_item()
    spend(channels, item_id, task_id)
    channels.clear()

    payload = _rerequest(item_id, **reopen)

    _assert_one_enabled_reminder(item_id, payload.get("scheduled_task_id", ""))
    _tick(payload["scheduled_task_id"])
    assert channels.sent() == _SENT_ONCE, "each target receives the reminder exactly once"


def _deliver_at_the_tick(channels: _Channels, task_id: str) -> None:
    del channels
    _tick(task_id)


def _deliver_early(channels: _Channels, task_id: str) -> None:
    del channels
    assert run_task_now(task_id, _RUNNERS) is True


def _deliver_partially_at_the_tick(channels: _Channels, task_id: str) -> None:
    channels.telegram.ok = False
    _tick(task_id)


@pytest.mark.parametrize(
    "deliver",
    [
        pytest.param(_deliver_at_the_tick, id="scheduled-tick"),
        pytest.param(_deliver_early, id="cron-run-before-the-time"),
    ],
)
def test_a_fully_delivered_reminder_is_not_sent_again_by_a_rerequest(
    channels: _Channels, deliver: Callable[[_Channels, str], None]
) -> None:
    item_id, task_id = _add_item()
    deliver(channels, task_id)
    channels.clear()
    before = _rows(item_id)

    _rerequest(item_id)
    _tick(task_id)

    assert channels.sent() == _NOTHING_SENT
    assert _rows(item_id) == before, "the delivered reminder is left as it was and none is added"


def test_a_partially_delivered_reminder_keeps_its_failed_only_retry_after_a_rerequest(
    channels: _Channels,
) -> None:
    item_id, task_id = _add_item()
    _deliver_partially_at_the_tick(channels, task_id)
    channels.clear()
    before = _rows(item_id)

    _rerequest(item_id)
    assert channels.sent() == _NOTHING_SENT, "the re-request itself sends nothing"
    assert _rows(item_id) == before, "the partially delivered reminder is left as it was"
    _tick(task_id)
    assert channels.sent() == _NOTHING_SENT, "its spent fire time does not run again"

    channels.telegram.ok = True
    assert run_task_now(task_id, _RUNNERS, only_failed=True) is True
    assert channels.sent() == {"slack": [], "telegram": ["T1"]}


def _early_partial_delivery_then_tick(
    channels: _Channels, *, rerequest: bool
) -> tuple[str, dict[str, list[str]]]:
    item_id, task_id = _add_item()
    channels.telegram.ok = False
    assert run_task_now(task_id, _RUNNERS) is True
    channels.clear()
    if rerequest:
        _rerequest(item_id)
    _tick(task_id)
    return task_id, channels.sent()


def test_an_early_partial_delivery_keeps_its_tick_and_retry_after_a_rerequest(
    channels: _Channels, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_id, rerequested_tick = _early_partial_delivery_then_tick(channels, rerequest=True)
    channels.telegram.ok = True
    channels.clear()
    assert run_task_now(task_id, _RUNNERS, only_failed=True) is True
    assert channels.sent() == {"slack": [], "telegram": ["T1"]}

    control = _use_stores(tmp_path / "control", monkeypatch)
    _control_id, control_tick = _early_partial_delivery_then_tick(control, rerequest=False)
    assert rerequested_tick == control_tick, (
        "the tick at the reminder time delivers as it would have without the re-request"
    )


@pytest.mark.parametrize(
    "deliver",
    [
        pytest.param(_deliver_at_the_tick, id="scheduled-tick"),
        pytest.param(_deliver_early, id="cron-run-before-the-time"),
        pytest.param(_deliver_partially_at_the_tick, id="partial-at-the-tick"),
    ],
)
def test_an_already_delivered_reminder_is_reported_without_a_scheduled_id(
    channels: _Channels, deliver: Callable[[_Channels, str], None]
) -> None:
    item_id, task_id = _add_item()
    deliver(channels, task_id)

    payload = _rerequest(item_id)

    assert not payload.get("scheduled_task_id"), "nothing is pending, so no reminder id is reported"
    assert payload.get("reminder_status") == "already_delivered"


_OTHER_FORMS = [
    pytest.param(
        {}, {"channel_provider": "slack", "channel_id": "C1"}, id="same-targets-in-another-order"
    ),
    pytest.param(
        {"remind_at": _SEOUL_RUN_AT, "timezone": "Asia/Seoul"},
        {},
        id="naive-zone-time-then-the-same-instant-with-an-offset",
    ),
]


@pytest.mark.parametrize(("added_as", "requested_as"), _OTHER_FORMS)
def test_a_delivered_reminder_requested_in_another_form_is_not_sent_again(
    channels: _Channels, added_as: dict[str, str], requested_as: dict[str, str]
) -> None:
    item_id, task_id = _add_item(**added_as)
    _tick(task_id)
    channels.clear()
    before = _rows(item_id)

    payload = _rerequest(item_id, **requested_as)
    _tick_every_enabled_reminder(item_id)

    assert channels.sent() == _NOTHING_SENT
    assert _rows(item_id) == before, "the delivered reminder is left as it was and none is added"
    assert not payload.get("scheduled_task_id")
    assert payload.get("reminder_status") == "already_delivered"


@pytest.mark.parametrize(("added_as", "requested_as"), _OTHER_FORMS)
def test_a_pending_reminder_requested_in_another_form_is_delivered_once(
    channels: _Channels, added_as: dict[str, str], requested_as: dict[str, str]
) -> None:
    item_id, _task_id = _add_item(**added_as)

    payload = _rerequest(item_id, **requested_as)

    _assert_one_enabled_reminder(item_id, payload.get("scheduled_task_id", ""))
    _tick_every_enabled_reminder(item_id)
    assert channels.sent() == _SENT_ONCE


def _store_reordered_copy(task_id: str) -> ScheduledTask:
    """Store the same reminder again with its targets reversed, as a re-request once could."""
    original = get_task(task_id)
    assert original is not None
    targets = json.loads(original.params["delivery_targets"])[::-1]
    copy = add_task(
        ScheduledTask(
            kind=TaskKind.WORK_ITEM_REMINDER,
            cron="",
            timezone=original.timezone,
            provider=Provider(targets[0]["provider"]),
            chat_id=targets[0]["chat_id"],
            params={
                **original.params,
                "delivery_targets": json.dumps(targets, separators=(",", ":")),
            },
        )
    )
    assert copy.id != original.id
    return copy


def test_a_delivered_reminder_outweighs_an_undelivered_copy_of_it(channels: _Channels) -> None:
    item_id, task_id = _add_item()
    _tick(task_id)
    copy = _store_reordered_copy(task_id)
    channels.clear()

    payload = _rerequest(item_id)
    _tick_every_enabled_reminder(item_id)

    assert channels.sent() == _NOTHING_SENT
    assert _rows(item_id) == [(task_id, False), (copy.id, False)], (
        "the undelivered copy is turned off, the delivered reminder is left as it was"
    )
    assert not payload.get("scheduled_task_id")
    assert payload.get("reminder_status") == "already_delivered"


def test_two_undelivered_copies_of_a_reminder_leave_one_enabled(channels: _Channels) -> None:
    item_id, task_id = _add_item()
    copy = _store_reordered_copy(task_id)

    payload = _rerequest(item_id)

    _assert_one_enabled_reminder(item_id, payload.get("scheduled_task_id", ""))
    assert payload["scheduled_task_id"] == copy.id, "a tie goes to the most recently stored copy"
    _tick_every_enabled_reminder(item_id)
    assert channels.sent() == _SENT_ONCE
