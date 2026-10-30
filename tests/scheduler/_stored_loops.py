"""Stored loops for the tests of a cron schedule that can never fire.

``add_task`` validates no schedule, so it can store a row every write path refuses.
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Any

import pytest

import infrastructure.scheduling.scheduler as scheduler_pkg
from infrastructure.scheduling.scheduler.loop_constants import LOOP_PROMPT_PARAM
from infrastructure.scheduling.scheduler.storage import add_task, get_task
from infrastructure.scheduling.scheduler.types import Provider, ScheduledTask, TaskKind

IMPOSSIBLE_CRON = "0 8 31 2 *"
FIRING_CRON = "0 9 * * *"
REFUSAL_PHRASE = "no possible future execution time"
NEVER_RUNS = "Never runs"


def isolate_scheduler_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point every scheduler module that holds the home directory at ``tmp_path``."""
    home = tmp_path / "opensre-home"
    for info in pkgutil.walk_packages(scheduler_pkg.__path__, f"{scheduler_pkg.__name__}."):
        module = importlib.import_module(info.name)
        if hasattr(module, "OPENSRE_HOME_DIR"):
            monkeypatch.setattr(module, "OPENSRE_HOME_DIR", home)


def store_loop(
    task_id: str,
    *,
    cron: str = IMPOSSIBLE_CRON,
    enabled: bool = True,
    kind: TaskKind = TaskKind.MANUAL_LOOP,
    name: str = "",
    **fields: Any,
) -> ScheduledTask:
    """Store a row whose prompt is its id, because ``add_task`` merges equivalent rows."""
    return add_task(
        ScheduledTask(
            id=task_id,
            name=name or f"Loop {task_id}",
            kind=kind,
            cron=cron,
            provider=Provider.INTERACTIVE_SHELL,
            enabled=enabled,
            params={LOOP_PROMPT_PARAM: f"prompt of {task_id}", **fields.pop("params", {})},
            **fields,
        )
    )


def stored_row(task_id: str) -> dict[str, Any]:
    task = get_task(task_id)
    assert task is not None, f"{task_id} is no longer stored"
    return task.model_dump()


def flat(output: str) -> str:
    """Terminal output with its line wrapping undone, for a phrase a console may break."""
    return " ".join(output.split())
