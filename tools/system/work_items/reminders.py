"""Scheduled reminder helpers for work items."""

from __future__ import annotations

import json
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime

from config.constants.work_items import WORK_ITEM_REMINDER_RUN_AT_PARAM
from core.domain.work_items import (
    WorkItem,
    WorkItemChannelTarget,
    parse_work_item_datetime,
    resolve_work_item_datetime,
    work_items_path,
)
from infrastructure.scheduling.scheduler.delivery_plan import configured_target_keys
from infrastructure.scheduling.scheduler.run_history import run_trigger
from infrastructure.scheduling.scheduler.storage import add_task as add_scheduled_task
from infrastructure.scheduling.scheduler.storage import (
    get_all_runs,
    list_tasks,
    remove_task,
    update_task,
)
from infrastructure.scheduling.scheduler.task_visibility import visible_to_current_scope
from infrastructure.scheduling.scheduler.types import (
    Provider,
    ScheduledTask,
    TaskKind,
    TaskRun,
    TaskStatus,
)
from tools.system.work_items.validation import validate_provider

_UNFINISHED_RUN_STATUSES = frozenset({TaskStatus.PENDING, TaskStatus.RUNNING})


@dataclass(frozen=True, slots=True)
class ScheduledReminder:
    """The reminder task a scheduling request resolved to, as stored after the call.

    ``delivered``: a run of this reminder reached at least one of its targets.
    ``pending``: a scheduled delivery of ``task`` is still ahead.
    """

    task: ScheduledTask
    delivered: bool
    pending: bool


def disable_existing_item_reminders(item_id: str, *, keep: Collection[str] = ()) -> int:
    """Disable enabled one-shot reminders for ``item_id``, except the task ids in ``keep``."""
    disabled = 0
    for task in list_tasks():
        if task.kind is not TaskKind.WORK_ITEM_REMINDER:
            continue
        if not task.enabled or task.id in keep:
            continue
        if task.params.get("work_item_id", "").strip() != item_id:
            continue
        task.enabled = False
        if update_task(task):
            disabled += 1
    return disabled


def _equivalent_reminders(requested: ScheduledTask, remind_at: datetime) -> list[ScheduledTask]:
    """Stored reminders of the requested item at the same instant with the same target set."""
    item_id = requested.params["work_item_id"]
    store_path = requested.params["store_path"]
    target_set = frozenset(configured_target_keys(requested))
    return [
        task
        for task in list_tasks()
        if task.kind is TaskKind.WORK_ITEM_REMINDER
        and task.params.get("work_item_id", "").strip() == item_id
        and task.params.get("store_path", "") == store_path
        and visible_to_current_scope(task)
        and parse_work_item_datetime(task.params.get(WORK_ITEM_REMINDER_RUN_AT_PARAM, ""))
        == remind_at
        and frozenset(configured_target_keys(task)) == target_set
    ]


def _delivered(runs: Sequence[TaskRun]) -> bool:
    return any(outcome.ok for run in runs for outcome in run.targets)


def _fire_time_spent(runs: Sequence[TaskRun]) -> bool:
    """Whether the newest run at a scheduled fire time has finished; ``runs`` is newest first."""
    scheduled = next((run for run in runs if run_trigger(run.fire_time) == "schedule"), None)
    return scheduled is not None and scheduled.status not in _UNFINISHED_RUN_STATUSES


def _reuse_equivalents(
    equivalents: list[ScheduledTask], requested: ScheduledTask
) -> tuple[ScheduledReminder, set[str]]:
    """Resolve a request the store already holds; return it and the ids the disable must skip.

    A delivered reminder is never written to. An undelivered one is re-enabled
    while its scheduled fire time is unspent and replaced by ``requested`` once it
    is spent. Ties go to the most recently stored row.
    """
    histories = {task.id: get_all_runs(task.id) for task in equivalents}
    delivered = [task for task in equivalents if _delivered(histories[task.id])]
    if delivered:
        live = [
            task for task in delivered if task.enabled and not _fire_time_spent(histories[task.id])
        ]
        resolved = ScheduledReminder((live or delivered)[-1], delivered=True, pending=bool(live))
        return resolved, {task.id for task in delivered}
    unspent = [task for task in equivalents if not _fire_time_spent(histories[task.id])]
    if unspent:
        chosen = ([task for task in unspent if task.enabled] or unspent)[-1]
        if not chosen.enabled:
            chosen = chosen.model_copy(update={"enabled": True})
            update_task(chosen)
    else:
        for task in equivalents:
            remove_task(task.id)
        chosen = add_scheduled_task(requested)
    return ScheduledReminder(chosen, delivered=False, pending=True), {chosen.id}


def schedule_item_reminder(
    item: WorkItem,
    *,
    targets: list[WorkItemChannelTarget],
    timezone: str,
) -> ScheduledReminder | None:
    """Schedule ``item``'s one-shot reminder and disable its other reminders.

    Returns ``None`` when nothing can be scheduled: no ``remind_at``, a time that
    does not resolve, or no supported target.
    """
    if not item.remind_at:
        return None
    schedule_timezone = timezone.strip() or "UTC"
    parsed_remind_at = parse_work_item_datetime(item.remind_at)
    remind_at = resolve_work_item_datetime(item.remind_at, schedule_timezone)
    if remind_at is None:
        return None
    if parsed_remind_at is not None and parsed_remind_at.tzinfo is not None:
        schedule_timezone = "UTC"
    valid_targets = [target for target in targets if validate_provider(target.provider) is not None]
    if not valid_targets:
        return None
    primary = valid_targets[0]
    parsed_provider = Provider(primary.provider)
    task = ScheduledTask(
        kind=TaskKind.WORK_ITEM_REMINDER,
        cron="",
        timezone=schedule_timezone,
        provider=parsed_provider,
        chat_id=primary.chat_id,
        params={
            "work_item_id": item.id,
            "store_path": str(work_items_path()),
            "disable_after_success": "true",
            WORK_ITEM_REMINDER_RUN_AT_PARAM: remind_at.isoformat(),
            "delivery_targets": json.dumps(
                [target.to_dict() for target in valid_targets], separators=(",", ":")
            ),
        },
    )
    equivalents = _equivalent_reminders(task, remind_at)
    if equivalents:
        reminder, keep = _reuse_equivalents(equivalents, task)
    else:
        reminder = ScheduledReminder(add_scheduled_task(task), delivered=False, pending=True)
        keep = {reminder.task.id}
    disable_existing_item_reminders(item.id, keep=keep)
    return reminder


_disable_existing_item_reminders = disable_existing_item_reminders
_schedule_item_reminder = schedule_item_reminder

__all__ = [
    "ScheduledReminder",
    "_disable_existing_item_reminders",
    "_schedule_item_reminder",
    "disable_existing_item_reminders",
    "schedule_item_reminder",
]
