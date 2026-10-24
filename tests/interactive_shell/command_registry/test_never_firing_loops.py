"""A cron schedule that can never fire is refused in the shell and never shown as running.

``0 8 31 2 *`` (February 31) builds an APScheduler trigger, so ``/loops add``
reported ``loop created`` and every view showed the loop as active although it
had no possible execution time.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from infrastructure.scheduling.scheduler.loop_constants import LOOP_PROMPT_PARAM
from infrastructure.scheduling.scheduler.storage import add_task, get_task, list_tasks, task_store
from infrastructure.scheduling.scheduler.types import Provider, ScheduledTask, TaskKind
from surfaces.interactive_shell.command_registry import dispatch_slash, loops_cmds
from surfaces.interactive_shell.session import Session

_NEVER_FIRES = "0 8 31 2 *"
_REASON = "no possible future execution time"
_WIDTH = 100


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A temp task store, and no scheduler started by a command under test."""
    path = tmp_path / "tasks.json"
    monkeypatch.setattr(task_store, "default_task_store_path", lambda: path)
    monkeypatch.setattr(loops_cmds, "_reload_loop_scheduler", lambda *_args: None)
    # Rich renders "dumb" terminals at 80 columns whatever width the console asks for.
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("COLUMNS", str(_WIDTH))
    return path


def _dispatch(command: str) -> str:
    output = io.StringIO()
    with create_app_session(output=DummyOutput()):
        dispatch_slash(command, Session(), Console(file=output, width=_WIDTH), is_tty=False)
    return " ".join(output.getvalue().split())


def _stored_loop(store: Path, *, cron: str, enabled: bool) -> ScheduledTask:
    """A loop written by the store layer, which accepts any schedule (tasks saved before the check)."""
    return add_task(
        ScheduledTask(
            name="NeverFire",
            kind=TaskKind.MANUAL_LOOP,
            cron=cron,
            provider=Provider.INTERACTIVE_SHELL,
            enabled=enabled,
            params={LOOP_PROMPT_PARAM: "Impossible date audit"},
        ),
        store,
    )


def test_loops_add_refuses_a_schedule_that_can_never_fire(store: Path) -> None:
    output = _dispatch(
        f'/loops add --name NeverFire --cron "{_NEVER_FIRES}" '
        '--prompt "Impossible date audit" --channel interactive_shell'
    )

    assert "could not add loop" in output and _REASON in output
    assert "loop created" not in output
    assert list_tasks(store) == []


def test_loops_edit_refuses_a_schedule_that_can_never_fire_and_keeps_the_loop(
    store: Path,
) -> None:
    loop = _stored_loop(store, cron="0 9 * * *", enabled=True)
    before = get_task(loop.id, store)

    output = _dispatch(f'/loops edit {loop.id} --cron "{_NEVER_FIRES}"')

    assert "could not edit loop" in output and _REASON in output
    assert "loop updated" not in output
    assert get_task(loop.id, store) == before


def test_loops_start_refuses_a_loop_that_can_never_fire_and_leaves_it_off(store: Path) -> None:
    loop = _stored_loop(store, cron=_NEVER_FIRES, enabled=False)

    output = _dispatch(f"/loops start {loop.id}")

    assert _REASON in output
    assert "loop started" not in output
    stored = get_task(loop.id, store)
    assert stored is not None and not stored.enabled


def test_loops_next_reports_a_never_firing_loop_as_invalid_with_its_reason(store: Path) -> None:
    loop = _stored_loop(store, cron=_NEVER_FIRES, enabled=True)

    output = _dispatch(f"/loops next {loop.id}")

    assert "status: invalid" in output
    assert "schedule error:" in output and _REASON in output


def test_loops_show_reports_a_never_firing_loop_as_invalid_with_its_reason(store: Path) -> None:
    loop = _stored_loop(store, cron=_NEVER_FIRES, enabled=True)

    output = _dispatch(f"/loops show {loop.id}")

    assert "Status: Invalid" in output and "Status: Active" not in output
    assert "Schedule error:" in output and _REASON in output


@pytest.mark.parametrize("command", ["/loops", "/loops active"])
def test_loop_lists_mark_a_never_firing_loop_invalid_and_say_why(store: Path, command: str) -> None:
    _stored_loop(store, cron=_NEVER_FIRES, enabled=True)

    output = _dispatch(command)

    assert "NeverFire" in output
    assert "Invalid" in output and "Active" not in output
    assert _REASON in output
