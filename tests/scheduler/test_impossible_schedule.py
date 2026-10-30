"""A cron schedule that can never fire: the check, editing, summaries and scheduler counts."""

from __future__ import annotations

import logging
import signal
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from apscheduler.schedulers.blocking import BlockingScheduler

import bootstrap.adapters as adapters
from config.constants.scheduler import SCHEDULER_SCHEDULE_CHANGED_AT_PARAM
from config.constants.work_items import WORK_ITEM_REMINDER_RUN_AT_PARAM
from gateway.core.lifecycle.controller import GatewayController
from infrastructure.scheduling.scheduler import runner
from infrastructure.scheduling.scheduler.cron_expression import build_cron_trigger
from infrastructure.scheduling.scheduler.loop_constants import LOOP_GROUP_ID_PARAM
from infrastructure.scheduling.scheduler.loop_edit import edit_loop
from infrastructure.scheduling.scheduler.loops import (
    LoopSummary,
    create_manual_loop,
    delete_loop,
    list_loop_summaries,
    set_loop_enabled,
    summarize_loop,
    summarize_loops,
)
from infrastructure.scheduling.scheduler.runners import SchedulerRunners
from infrastructure.scheduling.scheduler.storage import get_task, list_tasks, remove_task
from infrastructure.scheduling.scheduler.types import Provider, ScheduledTask, TaskKind
from tests.scheduler._bundle import runners_with_agent
from tests.scheduler._stored_loops import (
    FIRING_CRON,
    IMPOSSIBLE_CRON,
    REFUSAL_PHRASE,
    isolate_scheduler_home,
    store_loop,
    stored_row,
)

_STORED_NEXT_RUN = "2030-01-01T08:00:00+00:00"


@pytest.fixture(autouse=True)
def _scheduler_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_scheduler_home(monkeypatch, tmp_path)


def _summary(task_id: str) -> LoopSummary:
    [loop] = [loop for loop in list_loop_summaries() if task_id in loop.task_ids]
    return loop


@pytest.mark.parametrize(
    "cron",
    ["0 8 31 2 *", "0 0 30 2 *", "0 8 31 4 *", "0 0 31 2 mon", "*/30 0 8 31 2 *"],
)
def test_a_schedule_with_no_next_fire_is_refused_with_a_message_that_says_so(cron: str) -> None:
    with pytest.raises(ValueError, match=REFUSAL_PHRASE):
        build_cron_trigger(cron, "UTC")


def test_a_malformed_cron_keeps_its_own_message() -> None:
    with pytest.raises(ValueError) as refused:
        build_cron_trigger("not a cron", "UTC")
    assert REFUSAL_PHRASE not in str(refused.value)


@pytest.mark.parametrize("cron", ["0 0 29 2 *", "0 0 29 2 mon", "0 8 5 2 sun", "0 8 last 2 *"])
def test_a_rare_schedule_stays_accepted(cron: str) -> None:
    build_cron_trigger(cron, "UTC")

    created = create_manual_loop(name="Rare", prompt="check", cron=cron)

    assert created.task.cron == cron
    assert _summary(created.task.id).status == "active"
    assert _summary(created.task.id).next_run


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize(
    "changes",
    [{"cron": IMPOSSIBLE_CRON}, {"cron": IMPOSSIBLE_CRON, "timezone": "Asia/Seoul"}],
)
def test_an_edit_to_an_impossible_cron_writes_nothing(
    changes: dict[str, str], enabled: bool
) -> None:
    store_loop("editedloop", cron=FIRING_CRON, enabled=enabled)
    before = stored_row("editedloop")

    mutation, error = edit_loop(("editedloop",), changes, loop_id="editedloop")

    assert mutation is None
    assert REFUSAL_PHRASE in error
    assert stored_row("editedloop") == before, "a refused edit left the row changed"


@pytest.mark.parametrize("enabled", [True, False])
def test_a_timezone_only_edit_of_a_stored_impossible_loop_is_applied(enabled: bool) -> None:
    store_loop("kept", enabled=enabled)

    mutation, error = edit_loop(("kept",), {"timezone": "Asia/Seoul"}, loop_id="kept")

    assert error == ""
    assert mutation is not None
    after = stored_row("kept")
    assert after["timezone"] == "Asia/Seoul"
    assert after["cron"] == IMPOSSIBLE_CRON
    assert after["enabled"] is enabled
    assert after["next_run"] is None


def test_a_bad_zone_on_a_stored_impossible_loop_is_still_refused() -> None:
    store_loop("kept")
    before = stored_row("kept")

    mutation, error = edit_loop(("kept",), {"timezone": "Not/AZone"}, loop_id="kept")

    assert mutation is None
    assert error
    assert stored_row("kept") == before


@pytest.mark.parametrize(
    ("stored", "resent"),
    [(IMPOSSIBLE_CRON, IMPOSSIBLE_CRON), ("0 8  31 2 *", IMPOSSIBLE_CRON)],
)
def test_resending_the_stored_cron_beside_a_rename_is_no_schedule_change(
    stored: str, resent: str
) -> None:
    store_loop("kept", cron=stored)

    mutation, error = edit_loop(("kept",), {"cron": resent, "name": "Renamed"}, loop_id="kept")

    assert error == ""
    assert mutation is not None
    after = stored_row("kept")
    assert after["name"] == "Renamed"
    assert after["cron"] == stored
    assert SCHEDULER_SCHEDULE_CHANGED_AT_PARAM not in after["params"]


def test_resending_a_cron_that_can_fire_is_still_capped() -> None:
    store_loop("uncapped", cron="*/15 * * * *")

    mutation, error = edit_loop(("uncapped",), {"cron": "*/15 * * * *"}, loop_id="uncapped")

    assert error == ""
    assert mutation is not None
    assert stored_row("uncapped")["cron"] == "15 * * * *"


def test_repairing_an_enabled_impossible_loop_makes_it_active() -> None:
    store_loop("repaired")

    mutation, error = edit_loop(("repaired",), {"cron": FIRING_CRON}, loop_id="repaired")

    assert error == ""
    assert mutation is not None
    assert stored_row("repaired")["cron"] == FIRING_CRON
    loop = _summary("repaired")
    assert (loop.status, loop.schedule_error) == ("active", "")
    assert loop.next_run


def test_repairing_a_disabled_impossible_loop_leaves_it_disabled_until_started() -> None:
    store_loop("repaired", enabled=False)

    mutation, error = edit_loop(("repaired",), {"cron": FIRING_CRON}, loop_id="repaired")

    assert error == ""
    assert mutation is not None
    assert stored_row("repaired")["cron"] == FIRING_CRON
    assert stored_row("repaired")["enabled"] is False
    assert _summary("repaired").schedule_error == ""

    started, error = set_loop_enabled("repaired", enabled=True)

    assert error == ""
    assert started is not None
    assert started.summary.status == "active"
    assert started.summary.next_run


def test_starting_a_disabled_impossible_loop_is_refused_and_it_stays_disabled() -> None:
    store_loop("stopped", enabled=False)

    mutation, error = set_loop_enabled("stopped", enabled=True)

    assert mutation is None
    assert REFUSAL_PHRASE in error
    assert stored_row("stopped")["enabled"] is False
    assert stored_row("stopped")["next_run"] is None


def test_stop_and_delete_stay_available_for_an_impossible_loop() -> None:
    store_loop("removable")

    stopped, error = set_loop_enabled("removable", enabled=False)
    assert error == ""
    assert stopped is not None
    assert stored_row("removable")["enabled"] is False

    deleted, error = delete_loop("removable")
    assert error == ""
    assert deleted is not None
    assert get_task("removable") is None


def test_an_enabled_impossible_loop_is_unschedulable_with_the_sentence_and_no_next_run() -> None:
    loop = summarize_loop(
        store_loop("impossible", next_run=_STORED_NEXT_RUN),
    )

    assert loop.status == "unschedulable"
    assert REFUSAL_PHRASE in loop.schedule_error
    assert loop.next_run is None, "a stored date must not stand beside the warning"
    assert loop.enabled is True


@pytest.mark.parametrize(
    ("last_run", "status"), [("2026-01-01T00:00:00+00:00", "paused"), (None, "draft")]
)
def test_a_disabled_impossible_loop_keeps_its_word_and_carries_the_sentence(
    last_run: str | None, status: str
) -> None:
    loop = summarize_loop(store_loop("impossible", enabled=False, last_run=last_run))

    assert loop.status == status
    assert REFUSAL_PHRASE in loop.schedule_error


def _group_member(task_id: str, cron: str) -> ScheduledTask:
    return store_loop(task_id, cron=cron, params={LOOP_GROUP_ID_PARAM: "group"})


def test_a_group_of_impossible_members_is_one_unschedulable_loop() -> None:
    [loop] = summarize_loops(
        [_group_member("first", IMPOSSIBLE_CRON), _group_member("second", "0 0 30 2 *")]
    )

    assert loop.status == "unschedulable"
    assert loop.next_run is None


def test_a_group_with_a_firing_member_stays_active_with_its_next_run() -> None:
    [loop] = summarize_loops(
        [_group_member("first", IMPOSSIBLE_CRON), _group_member("second", FIRING_CRON)]
    )

    assert loop.status == "active"
    assert loop.next_run


def test_a_malformed_stored_cron_keeps_its_status_and_gets_no_phrase() -> None:
    loop = summarize_loop(store_loop("malformed", cron="not a cron"))

    assert loop.status == "active"
    assert loop.schedule_error
    assert REFUSAL_PHRASE not in loop.schedule_error


def test_a_reminder_whose_time_has_passed_is_not_flagged() -> None:
    past = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    loop = summarize_loop(
        ScheduledTask(
            kind=TaskKind.WORK_ITEM_REMINDER,
            cron=FIRING_CRON,
            provider=Provider.INTERACTIVE_SHELL,
            params={"work_item_id": "item", WORK_ITEM_REMINDER_RUN_AT_PARAM: past},
            next_run=past,
        )
    )

    assert loop.status == "active"
    assert loop.schedule_error == ""
    assert loop.next_run


@pytest.fixture
def background() -> Iterator[list[Any]]:
    started: list[Any] = []
    yield started
    for scheduler in started:
        scheduler.shutdown(wait=False)


def test_registration_counts_only_the_loop_that_can_fire(
    background: list[Any], caplog: pytest.LogCaptureFixture
) -> None:
    store_loop("firing", cron=FIRING_CRON)
    store_loop("impossible")
    store_loop("malformed", cron="not a cron")

    with caplog.at_level(logging.INFO, logger=runner.logger.name):
        scheduler, count = runner.start_background_scheduler(runners_with_agent(lambda _p: ""))
    background.append(scheduler)

    assert count == 1
    assert scheduler.get_job("firing") is not None
    assert scheduler.get_job("impossible") is None
    levels = {
        record.args[0]: record.levelno
        for record in caplog.records
        if record.getMessage().startswith("Skipping task")
    }
    assert levels == {"impossible": logging.WARNING, "malformed": logging.ERROR}


def test_a_store_of_only_an_impossible_loop_starts_no_background_scheduler(
    background: list[Any],
) -> None:
    store_loop("impossible")

    scheduler, count = runner.start_background_scheduler(runners_with_agent(lambda _p: ""))
    if scheduler is not None:
        background.append(scheduler)

    assert (scheduler, count) == (None, 0)


@pytest.fixture
def no_block(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int]]:
    """Let a blocking scheduler return at once, and record what it reports about itself."""
    recorded: list[tuple[str, int]] = []

    def record(operation: str, *, task_count: int = 0, **_kwargs: Any) -> None:
        recorded.append((operation, task_count))

    monkeypatch.setattr(BlockingScheduler, "start", lambda _self: None)
    monkeypatch.setattr(runner, "record_scheduler_service_operation", record)
    monkeypatch.setattr(signal, "signal", lambda *_a, **_k: None)
    return recorded


def test_cron_start_exits_when_only_an_impossible_loop_is_stored(
    no_block: list[tuple[str, int]],
) -> None:
    store_loop("impossible")

    with pytest.raises(SystemExit):
        runner.start_scheduler(runners_with_agent(lambda _p: ""))

    assert no_block == [("scheduler_idle", 0)]


def test_a_scheduler_service_idles_when_only_an_impossible_loop_is_stored(
    no_block: list[tuple[str, int]],
) -> None:
    store_loop("impossible")

    runner.start_scheduler(runners_with_agent(lambda _p: ""), idle_when_empty=True)

    assert no_block[0] == ("scheduler_idle", 0)


@pytest.fixture
def gateway(monkeypatch: pytest.MonkeyPatch) -> Iterator[GatewayController]:
    """A controller whose threads, adapters and web report are off; its scheduler is real."""
    for method in (
        "_start_scheduler_reload_watcher",
        "_start_webapp_loop_sync",
        "_start_knowledge",
        "_report_loops_to_webapp",
        "_publish_status",
    ):
        monkeypatch.setattr(GatewayController, method, lambda *_a, **_k: None)
    monkeypatch.setattr(adapters, "install_scheduled_delivery_adapters", lambda: None)
    monkeypatch.setattr(adapters, "scheduler_runners", _gateway_runners)
    controller = GatewayController()
    yield controller
    if controller.scheduler is not None:
        controller.scheduler.shutdown(wait=False)


def _gateway_runners() -> SchedulerRunners:
    return runners_with_agent(lambda _p: "")


def test_the_gateway_counts_only_the_loop_that_can_fire(gateway: GatewayController) -> None:
    store_loop("firing", cron=FIRING_CRON)
    store_loop("impossible")

    gateway.start_scheduler(logger=logging.getLogger(__name__))

    assert gateway.components["scheduler"] == "running 1 scheduled task(s)"

    remove_task("firing")
    gateway._reload_scheduler(logging.getLogger(__name__))

    assert gateway.components["scheduler"] == "idle (no scheduled tasks)"
    assert [task.id for task in list_tasks()] == ["impossible"]


def test_the_gateway_is_idle_when_only_an_impossible_loop_is_stored(
    gateway: GatewayController,
) -> None:
    store_loop("impossible")

    gateway.start_scheduler(logger=logging.getLogger(__name__))

    assert gateway.components["scheduler"] == "idle (no scheduled tasks)"
