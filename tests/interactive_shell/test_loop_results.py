"""Loop result presentation and full-report navigation through isolated real stores."""

from __future__ import annotations

import io
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from rich.console import Console

from infrastructure.scheduling.scheduler.loop_constants import LOOP_PROMPT_PARAM
from infrastructure.scheduling.scheduler.loops import LoopSummary, summarize_loop
from infrastructure.scheduling.scheduler.storage import (
    add_task,
    complete_run,
    get_runs,
    get_task,
    record_run_report,
    try_claim,
)
from infrastructure.scheduling.scheduler.types import (
    DeliveryOutcome,
    Provider,
    ScheduledTask,
    TaskKind,
    TaskReport,
    TaskRun,
    TaskStatus,
)
from surfaces.interactive_shell.command_registry import loop_show, loops_cmds
from surfaces.interactive_shell.session import Session
from surfaces.interactive_shell.ui.loops import render_loops


def _loop() -> LoopSummary:
    return summarize_loop(
        ScheduledTask(
            id="hidden-id",
            name="CI check · org/repo",
            kind=TaskKind.MANUAL_LOOP,
            cron="0 8 * * *",
            provider=Provider.SLACK,
        )
    )


@pytest.mark.parametrize("width", [40, 80, 114])
def test_schedule_records_preserve_findings_and_full_report_navigation(
    width: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Rich renders "dumb" terminals at a fixed 80 columns regardless of the
    # explicit console width, which breaks the geometry assertions below.
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("COLUMNS", str(width))
    output = io.StringIO()
    console = Console(file=output, width=width, color_system=None)
    loop = replace(_loop(), next_run="2026-09-14T06:00:00+00:00")
    run = TaskRun(
        task_id=loop.id,
        fire_time="2026-09-12T08:00",
        started_at="2026-09-12T08:00:00+00:00",
        status=TaskStatus.SUCCESS,
        report="## Report title\n\n" + "Long content " * 100,
        report_summary="1 fix; 3 failing workflows",
    )
    render_loops(
        console,
        [loop],
        {loop.id: run},
        now=datetime(2026, 9, 12, 8, 10, tzinfo=UTC),
        local_timezone=ZoneInfo("Europe/Berlin"),
    )
    text = output.getvalue()
    assert "Latest result" in text
    assert "1 fix" in text
    assert "Done" in text
    assert "Mon 08:00" in text
    assert "ID: hidden-id" in text
    assert "Report title" not in text
    assert "slack" in text
    assert "0 8 * * *" in text
    # Secondary fields reflow below the summary instead of squeezing table cells.
    assert "/loops show <name-or-id>" in text
    assert all(len(line) <= width for line in text.splitlines())


def test_run_states_and_local_dst_are_visible_without_reusing_old_findings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "114")
    output = io.StringIO()
    console = Console(file=output, width=114, color_system=None)
    loop = replace(_loop(), next_run="2026-10-25T01:30:00+00:00")
    runs = [
        TaskRun(task_id="failed", fire_time="x", status=TaskStatus.FAILED),
        TaskRun(task_id="quiet", fire_time="x", status=TaskStatus.SUCCESS, report=""),
        TaskRun(task_id="running", fire_time="x", status=TaskStatus.RUNNING),
        TaskRun(task_id="old", fire_time="x", status=TaskStatus.SUCCESS),
        TaskRun(
            task_id="partial",
            fire_time="x",
            status=TaskStatus.SUCCESS,
            report="[red]literal finding[/red]",
            targets=(DeliveryOutcome(provider=Provider.SLACK, ok=False),),
        ),
    ]
    loops = [replace(loop, id=run.task_id) for run in runs]
    render_loops(
        console,
        loops,
        {run.task_id: run for run in runs},
        now=datetime(2026, 10, 24, 8, tzinfo=UTC),
        local_timezone=ZoneInfo("Europe/Berlin"),
    )
    text = output.getvalue()
    assert "Tomorrow 02:30" in text
    assert "Failed" in text
    assert "No report" in text
    assert "Running" in text
    assert "Report not retained" in text
    assert "Delivery issue" in text
    assert "[red]literal finding[/red]" in text


@pytest.fixture()
def loop_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "infrastructure.scheduling.scheduler.storage.database.default_run_database_path",
        lambda: tmp_path / "scheduler.db",
    )
    monkeypatch.setattr(
        "infrastructure.scheduling.scheduler.storage.task_store.default_task_store_path",
        lambda: tmp_path / "tasks.json",
    )


def _stored_loop(task_id: str, name: str, report: str) -> ScheduledTask:
    task = ScheduledTask(
        id=task_id,
        name=name,
        kind=TaskKind.MANUAL_LOOP,
        cron="0 8 * * *",
        provider=Provider.SLACK,
        chat_id=task_id,
    )
    add_task(task)
    claim = try_claim(task.id, "2026-09-12T08:00")
    assert claim is not None
    assert record_run_report(claim, TaskReport(report, summary="1 workflow fixed"))
    assert complete_run(claim, status=TaskStatus.SUCCESS)
    return task


@pytest.mark.usefixtures("loop_stores")
def test_commands_show_latest_failure_and_allow_opening_the_earlier_full_report() -> None:
    report = "## Findings\n\n" + "The complete report continues. " * 30 + "FINAL EVIDENCE"
    task = _stored_loop("first-task", "Morning ops", report)
    prior_id = get_runs(task.id)[0].run_id
    claim = try_claim(task.id, "2026-09-13T08:00")
    assert claim is not None
    assert complete_run(claim, status=TaskStatus.FAILED, error="Provider unavailable")
    output = io.StringIO()
    console = Console(file=output, width=100, color_system=None)
    session = Session()

    loops_cmds._cmd_loops(session, console, [])
    assert "Failed" in output.getvalue()
    assert "1 workflow fixed" not in output.getvalue()

    output.seek(0)
    output.truncate()
    loops_cmds._cmd_loops(session, console, ["show", "Morning", "ops", "--run", str(prior_id)])
    text = " ".join(output.getvalue().split())
    assert "FINAL EVIDENCE" in text
    assert text.index("FINAL EVIDENCE") < text.index("Recent runs") < text.index("Configuration")
    assert "0 8 * * *" in text
    assert (
        "Provider unavailable" not in text
    )  # Selected run's details, not another attempt's error.
    # The opened attempt is recent enough to be in the history query; it must
    # not be listed again under its own report.
    recent = text[text.index("Recent runs") : text.index("Configuration")]
    assert f" {prior_id} Done " not in recent
    assert f" {get_runs(task.id)[0].run_id} Failed " in recent
    assert f"Run {prior_id} " in text


@pytest.mark.usefixtures("loop_stores")
def test_full_report_shows_link_urls() -> None:
    # Arrange: a stored report that links its pull request.
    url = "https://github.com/o/r/pull/1"
    _stored_loop("linked-task", "Linked loop", f"Opened [PR #1]({url}).")
    output = io.StringIO()
    console = Console(file=output, width=100, color_system=None)

    # Act
    loops_cmds._cmd_loops(Session(), console, ["show", "Linked", "loop"])

    # Assert: the report shows the URL, not only the link text.
    assert f"Opened PR #1 ({url})." in output.getvalue()


@pytest.mark.usefixtures("loop_stores")
def test_picker_disambiguates_duplicate_names_and_noninteractive_show_gives_a_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stored_loop("first-task", "Same name", "First report")
    _stored_loop("second-task", "Same name", "Second report")
    session = Session()
    output = io.StringIO()
    console = Console(file=output, width=100, color_system=None)
    monkeypatch.setattr(loop_show, "repl_tty_interactive", lambda: False)
    loop_show.show_loop(session, console, [])
    assert "Use /loops show <name-or-id>" in output.getvalue()
    loop_show.show_loop(session, console, ["Same name"])
    assert "ambiguous" in output.getvalue()

    def choose(**kwargs: Any) -> str:
        choices = kwargs["choices"]
        assert {choice[0] for choice in choices} == {"first-task", "second-task"}
        assert all(choice[0] in choice[1] for choice in choices)
        return "second-task"

    output.seek(0)
    output.truncate()
    session.terminal.exclusive_stdin_active = True
    monkeypatch.setattr(loop_show, "repl_tty_interactive", lambda: True)
    monkeypatch.setattr(loop_show, "repl_choose_one", choose)
    loop_show.show_loop(session, console, [])
    assert "Second report" in output.getvalue()
    assert "First report" not in output.getvalue()


@pytest.mark.parametrize("width", [40, 80, 120])
def test_loop_list_uses_responsive_schedule_records(
    monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    from rich.cells import cell_len

    monkeypatch.setenv("COLUMNS", str(width))
    loop = replace(
        _loop(), description="Fix failing checkout workflows.", next_run="2026-10-09T08:00:00Z"
    )
    output = io.StringIO()
    render_loops(
        Console(file=output, width=width),
        [loop],
        {},
        now=datetime(2026, 10, 9, tzinfo=UTC),
        local_timezone=UTC,
    )
    text = output.getvalue()
    assert "ID: hidden-id" in text
    assert "0 8 * * *" in text
    assert "Fix failing checkout workflows." in " ".join(text.split())
    assert "Active" in text
    assert "Not run yet" in text
    assert "/loops show" in text
    assert max(cell_len(line) for line in text.splitlines()) <= width


@pytest.mark.parametrize("width", [40, 80, 160])
def test_recent_runs_reflow_with_full_timestamp_and_bounded_findings(
    monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    from rich.cells import cell_len

    from surfaces.interactive_shell.ui.loops import _exact_time, render_loop_details

    monkeypatch.setenv("COLUMNS", str(width))
    monkeypatch.setenv("TERM", "xterm")
    run = TaskRun(
        task_id="hidden-id",
        fire_time="2026-10-09T09:00:00Z",
        started_at="2026-10-09T09:00:00Z",
        run_id=123456789,
        status=TaskStatus.FAILED,
        report_summary="[literal] " + "verified evidence " * 20 + "HIDDEN TAIL",
    )
    output = io.StringIO()
    render_loop_details(Console(file=output, width=width), _loop(), [run], None)
    text = output.getvalue()
    recent = text[text.index("Recent runs") : text.index("Configuration")]
    assert "123456789" in recent
    assert _exact_time(run.started_at) in " ".join(recent.split())
    assert "[literal]" in recent and "Failed" in recent
    assert "\n\n    [literal] verified" in recent
    assert "HIDDEN TAIL" not in recent and "…" in recent
    assert "/loops show hidden-id --run" in recent
    assert max(cell_len(line) for line in recent.splitlines()) <= width
    if width == 40:
        assert "Result:" in recent and "Started:" in recent


@pytest.mark.parametrize("width", [40, 160])
def test_invalid_loop_keeps_recovery_guidance_in_details_only(
    monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    from surfaces.interactive_shell.ui.loops import render_loop_details

    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("COLUMNS", str(width))
    error = "Legacy task kind 'daily_summary' was disabled. Recreate with opensre cron add."
    loop = replace(_loop(), enabled=False, schedule_error=error)
    output = io.StringIO()
    console = Console(file=output, width=width, color_system=None)
    render_loops(console, [loop], {})
    listing = output.getvalue()
    assert "Invalid" in listing and "Paused" in listing
    assert "Requires action" not in listing
    assert "daily_summary" not in listing
    assert "/loops show <name-or-id>" in listing

    output.seek(0)
    output.truncate()
    render_loop_details(console, loop, [], None)
    assert error in " ".join(output.getvalue().split())


@pytest.mark.parametrize("width", [40, 160])
def test_loop_description_is_a_spaced_preview_with_full_details(
    monkeypatch: pytest.MonkeyPatch, width: int
) -> None:
    from surfaces.interactive_shell.ui.loops import render_loop_details

    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("COLUMNS", str(width))
    description = "[literal] " + "Checkout evidence " * 8 + "FULL DESCRIPTION END"
    loop = replace(_loop(), description=description)
    output = io.StringIO()
    console = Console(file=output, width=width)
    render_loops(console, [loop], {})
    listing = output.getvalue()
    assert "FULL DESCRIPTION END" not in listing and "…" in listing
    assert "What it does:" not in listing
    lines = listing.splitlines()
    index = next(i for i, line in enumerate(lines) if "[literal]" in line)
    assert not lines[index - 1].strip()
    output.seek(0)
    output.truncate()
    render_loop_details(console, loop, [], None)
    assert description in " ".join(output.getvalue().split())


_NEVER_FIRES = "this cron schedule has no possible future execution time"


def _stored_never_firing_loop(task_id: str, name: str, *, enabled: bool) -> ScheduledTask:
    return add_task(
        ScheduledTask(
            id=task_id,
            name=name,
            kind=TaskKind.MANUAL_LOOP,
            cron="0 8 31 2 *",
            provider=Provider.INTERACTIVE_SHELL,
            enabled=enabled,
            params={LOOP_PROMPT_PARAM: f"Impossible date audit: {name}"},
        )
    )


def _command_text(args: list[str]) -> str:
    output = io.StringIO()
    loops_cmds._cmd_loops(Session(), Console(file=output, width=100, color_system=None), args)
    return " ".join(output.getvalue().split())


@pytest.mark.usefixtures("loop_stores")
def test_starting_a_stored_loop_that_can_never_fire_is_refused_and_it_stays_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stored_never_firing_loop("neverfire-off", "NeverFire", enabled=False)
    reloads: list[tuple[Any, ...]] = []
    monkeypatch.setattr(loops_cmds, "_reload_loop_scheduler", lambda *args: reloads.append(args))

    text = _command_text(["start", "neverfire-off"])

    assert _NEVER_FIRES in text
    assert "loop started" not in text
    assert not reloads
    stored = get_task("neverfire-off")
    assert stored is not None and stored.enabled is False


@pytest.mark.usefixtures("loop_stores")
def test_loop_lists_warn_about_a_loop_that_can_never_fire_and_never_show_it_active() -> None:
    add_task(
        ScheduledTask(
            id="daily-ops",
            name="Morning ops",
            kind=TaskKind.MANUAL_LOOP,
            cron="0 8 * * *",
            provider=Provider.INTERACTIVE_SHELL,
            params={LOOP_PROMPT_PARAM: "Check open incidents"},
        )
    )
    _stored_never_firing_loop("neverfire-on", "NeverFire", enabled=True)
    _stored_never_firing_loop("neverfire-off", "NeverFire stopped", enabled=False)

    every_loop = _command_text([])
    active_loops = _command_text(["active"])

    assert every_loop.count(f"Not running: {_NEVER_FIRES}") == 2
    assert every_loop.count("Active") == 1
    assert "ID: daily-ops" in active_loops
    assert "ID: neverfire-on" not in active_loops
    assert f"not running: NeverFire — {_NEVER_FIRES}" in active_loops


@pytest.mark.usefixtures("loop_stores")
def test_loop_details_warn_about_a_loop_that_can_never_fire_and_never_show_it_active() -> None:
    _stored_never_firing_loop("neverfire-on", "NeverFire", enabled=True)

    next_run = _command_text(["next", "neverfire-on"])
    details = _command_text(["show", "neverfire-on"])

    assert _NEVER_FIRES in next_run
    assert "status: active" not in next_run
    assert _NEVER_FIRES in details
    assert "Status: Active" not in details
