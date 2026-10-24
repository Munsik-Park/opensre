"""A loop's status word: active, invalid, paused after it ran, draft when it never ran."""

from __future__ import annotations

from pathlib import Path

from infrastructure.scheduling.scheduler.loop_constants import (
    LOOP_STATUS_ACTIVE,
    LOOP_STATUS_DRAFT,
    LOOP_STATUS_PAUSED,
)
from infrastructure.scheduling.scheduler.loops import LoopSummary, summarize_loops
from infrastructure.scheduling.scheduler.storage import add_task, list_tasks
from infrastructure.scheduling.scheduler.types import Provider, ScheduledTask, TaskKind


def _loop(*, enabled: bool, last_run: str | None) -> LoopSummary:
    return LoopSummary(
        id="a1",
        task_ids=("a1",),
        name="CI repair: o/r",
        description="",
        prompt="",
        kind=TaskKind.MANUAL_LOOP,
        cron="*/30 * * * * *",
        timezone="UTC",
        provider=Provider.INTERACTIVE_SHELL,
        chat_id="",
        channels=(),
        enabled=enabled,
        window_hours=24,
        last_run=last_run,
        next_run=None,
    )


def test_a_finished_repair_loop_reads_as_paused_not_draft() -> None:
    """A repair loop stopped after its run was listed as "draft", as if it had never run."""
    # Arrange
    ran_then_stopped = _loop(enabled=False, last_run="2026-09-24T16:51:00+00:00")
    never_ran = _loop(enabled=False, last_run=None)
    running = _loop(enabled=True, last_run="2026-09-24T16:51:00+00:00")

    # Act / Assert
    assert ran_then_stopped.status == LOOP_STATUS_PAUSED
    assert never_ran.status == LOOP_STATUS_DRAFT
    assert running.status == LOOP_STATUS_ACTIVE


def test_an_enabled_loop_whose_schedule_the_scheduler_refuses_is_not_active(
    tmp_path: Path,
) -> None:
    store = tmp_path / "tasks.json"
    for task_id, cron, timezone in (
        ("bad-cron", "99 99 * * *", "UTC"),
        ("bad-zone", "0 8 * * *", "Mars/Olympus"),
        ("valid", "0 8 * * *", "UTC"),
    ):
        add_task(
            ScheduledTask(
                id=task_id,
                name=task_id,
                kind=TaskKind.MANUAL_LOOP,
                cron=cron,
                timezone=timezone,
                provider=Provider.INTERACTIVE_SHELL,
            ),
            store,
        )

    statuses = {loop.id: loop.status for loop in summarize_loops(list_tasks(store))}

    assert statuses["bad-cron"] != LOOP_STATUS_ACTIVE
    assert statuses["bad-zone"] != LOOP_STATUS_ACTIVE
    assert statuses["valid"] == LOOP_STATUS_ACTIVE
