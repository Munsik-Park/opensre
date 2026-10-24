"""A stored task whose cron can never fire is not registered or counted as running.

The count ``_register_jobs`` returns is what the shell (``scheduler: running N
loop(s)``), ``opensre cron start`` and the gateway report. A one-shot reminder
whose time has passed still fires, so it stays registered.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from apscheduler.schedulers.background import BackgroundScheduler

from config.constants.work_items import WORK_ITEM_REMINDER_RUN_AT_PARAM
from infrastructure.scheduling.scheduler.loop_constants import (
    LOOP_PROMPT_PARAM,
    LOOP_STATUS_ACTIVE,
)
from infrastructure.scheduling.scheduler.loops import list_loop_summaries
from infrastructure.scheduling.scheduler.runner import _register_jobs
from infrastructure.scheduling.scheduler.storage import add_task, task_store
from infrastructure.scheduling.scheduler.types import Provider, ScheduledTask, TaskKind
from tests.scheduler._bundle import real_runners


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "tasks.json"
    monkeypatch.setattr(task_store, "default_task_store_path", lambda: path)
    return path


@pytest.fixture
def scheduler() -> Iterator[BackgroundScheduler]:
    """A real scheduler, started paused so no registered job runs during the test."""
    background = BackgroundScheduler(timezone=UTC)
    background.start(paused=True)
    try:
        yield background
    finally:
        background.shutdown(wait=False)


def _enabled(store: Path, *, kind: TaskKind, cron: str, **params: str) -> ScheduledTask:
    return add_task(
        ScheduledTask(
            kind=kind,
            cron=cron,
            provider=Provider.INTERACTIVE_SHELL,
            enabled=True,
            params=params,
        ),
        store,
    )


def test_a_never_firing_task_is_neither_registered_nor_counted(
    store: Path, scheduler: BackgroundScheduler
) -> None:
    firing = _enabled(
        store, kind=TaskKind.MANUAL_LOOP, cron="0 9 * * *", **{LOOP_PROMPT_PARAM: "Daily"}
    )
    _enabled(
        store, kind=TaskKind.MANUAL_LOOP, cron="0 8 31 2 *", **{LOOP_PROMPT_PARAM: "February 31"}
    )

    count = _register_jobs(scheduler, real_runners())

    assert count == 1
    assert [job.id for job in scheduler.get_jobs()] == [firing.id]


def test_an_overdue_reminder_is_still_registered_and_reads_as_active(
    store: Path, scheduler: BackgroundScheduler
) -> None:
    run_at = datetime.now(UTC) - timedelta(days=3)
    reminder = _enabled(
        store,
        kind=TaskKind.WORK_ITEM_REMINDER,
        cron="",
        work_item_id="w1",
        **{WORK_ITEM_REMINDER_RUN_AT_PARAM: run_at.isoformat()},
    )

    count = _register_jobs(scheduler, real_runners())
    [summary] = list_loop_summaries()

    assert count == 1
    assert [job.id for job in scheduler.get_jobs()] == [reminder.id]
    assert summary.schedule_error == ""
    assert summary.status == LOOP_STATUS_ACTIVE
