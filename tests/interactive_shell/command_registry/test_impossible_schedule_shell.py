from __future__ import annotations

import io
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import click
import pytest
from rich.console import Console

import bootstrap.adapters as adapters
import surfaces.interactive_shell.runtime.loop_scheduler as loop_scheduler
from infrastructure.scheduling.scheduler.loops import list_loop_summaries, summarize_loop
from infrastructure.scheduling.scheduler.runners import SchedulerRunners
from infrastructure.scheduling.scheduler.storage import list_tasks
from surfaces.interactive_shell.command_registry import (
    cron_cmds,
    loop_edit,
    loop_show,
    loops_cmds,
    target_collection,
)
from surfaces.interactive_shell.session import Session
from surfaces.interactive_shell.ui import loop_edit_input, target_input
from surfaces.interactive_shell.ui.cron_input import target as cron_target
from surfaces.interactive_shell.ui.cron_input.arguments import (
    parse_cron_draft,
    validate_cron_draft,
)
from surfaces.interactive_shell.ui.loop_input import LoopInput, _form_values, _validate
from surfaces.shared.terminal.components import choice_menu
from surfaces.shared.terminal.components.input_form import InputFormError
from surfaces.shared.terminal.components.target_picker import TargetChoice
from tests.scheduler._stored_loops import (
    FIRING_CRON,
    IMPOSSIBLE_CRON,
    NEVER_RUNS,
    REFUSAL_PHRASE,
    flat,
    isolate_scheduler_home,
    store_loop,
    stored_row,
)

_ANSI = re.compile(r"\x1b\[[0-9;:?]*[A-Za-z]")
_LONG_NAME = "Investigate checkout errors in the payments service"
_STORED_NEXT_RUN = "2030-01-01T08:00:00+00:00"


@pytest.fixture(autouse=True)
def _scheduler_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_scheduler_home(monkeypatch, tmp_path)


def _console() -> tuple[Console, io.StringIO]:
    output = io.StringIO()
    return Console(file=output, width=160, color_system=None), output


def _no_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loops_cmds, "_reload_loop_scheduler", lambda *_args: None)


def test_loops_add_refuses_an_impossible_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_reload(monkeypatch)
    console, output = _console()

    loops_cmds._cmd_loops_add(
        Session(),
        console,
        ["--name", "Never", "--cron", *IMPOSSIBLE_CRON.split(), "--prompt", "check"],
    )

    assert REFUSAL_PHRASE in flat(output.getvalue())
    assert list_tasks() == []


def test_the_loops_add_form_reports_an_impossible_schedule_on_its_field() -> None:
    draft = LoopInput(name="Never", prompt="check", cron=IMPOSSIBLE_CRON, cron_selected=True)

    with pytest.raises(InputFormError) as refused:
        _validate(_form_values(draft))

    assert refused.value.field == "schedule"
    assert REFUSAL_PHRASE in str(refused.value)


def test_the_cron_add_form_refuses_an_impossible_schedule() -> None:
    draft = parse_cron_draft(
        [
            "--kind",
            "manual_loop",
            "--provider",
            "interactive_shell",
            "--cron",
            IMPOSSIBLE_CRON,
            "--prompt",
            "check",
        ]
    )

    with pytest.raises(click.ClickException, match=REFUSAL_PHRASE):
        validate_cron_draft(draft)


def test_loops_edit_arguments_refuse_a_change_to_an_impossible_cron() -> None:
    store_loop("editedloop", cron=FIRING_CRON)
    before = stored_row("editedloop")
    console, output = _console()
    changed: list[bool] = []

    loop_edit.edit_loop_command(
        Session(),
        console,
        ["editedloop", "--cron", *IMPOSSIBLE_CRON.split()],
        on_schedule_change=lambda: changed.append(True),
    )

    assert REFUSAL_PHRASE in flat(output.getvalue())
    assert changed == []
    assert stored_row("editedloop") == before


def _edit_form_validator(monkeypatch: pytest.MonkeyPatch, task_id: str) -> Any:
    captured: dict[str, Any] = {}

    def capture(current: dict[str, str], **kwargs: Any) -> None:
        captured["values"] = dict(current)
        captured["validate"] = kwargs["validate"]

    monkeypatch.setattr(loop_edit_input, "build_input_form", capture)
    [loop] = [loop for loop in list_loop_summaries() if task_id in loop.task_ids]
    loop_edit_input.build_loop_edit_form(loop, prompt_editable=True)
    return captured["values"], captured["validate"]


def test_the_edit_form_reports_an_impossible_cron_on_schedule_even_with_a_new_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_loop("editedloop", cron="0 9 * * tue")
    values, validate = _edit_form_validator(monkeypatch, "editedloop")

    with pytest.raises(InputFormError) as refused:
        validate({**values, "schedule": IMPOSSIBLE_CRON, "timezone": "Asia/Seoul"})

    assert refused.value.field == "schedule"
    assert REFUSAL_PHRASE in str(refused.value)


def test_the_edit_form_lets_a_new_zone_through_beside_a_stored_impossible_cron(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_loop("kept")
    values, validate = _edit_form_validator(monkeypatch, "kept")

    assert validate({**values, "timezone": "Asia/Seoul"}) == {"timezone": "Asia/Seoul"}

    with pytest.raises(InputFormError) as refused:
        validate({**values, "timezone": "Not/AZone"})
    assert refused.value.field == "timezone"


def test_the_edit_form_lets_the_stored_impossible_cron_through_when_it_is_retyped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_loop("kept", cron="0 8  31 2 *")
    values, validate = _edit_form_validator(monkeypatch, "kept")

    changes = validate({**values, "schedule": IMPOSSIBLE_CRON, "name": "Renamed"})

    assert changes["name"] == "Renamed"


@pytest.mark.parametrize("enabled", [True, False])
def test_the_edit_form_lets_a_stored_impossible_loop_be_repaired(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    store_loop("repaired", enabled=enabled)
    values, validate = _edit_form_validator(monkeypatch, "repaired")

    assert validate({**values, "schedule": FIRING_CRON}) == {"cron": FIRING_CRON}


@pytest.mark.parametrize("enabled", [True, False])
def test_loops_start_refuses_an_impossible_loop_and_it_keeps_its_flag(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    _no_reload(monkeypatch)
    store_loop("stoppedloop", enabled=enabled)
    console, output = _console()

    loops_cmds._cmd_loops_set_enabled(Session(), console, ["stoppedloop"], enabled=True)

    assert REFUSAL_PHRASE in flat(output.getvalue())
    assert "loop started" not in output.getvalue()
    assert stored_row("stoppedloop")["enabled"] is enabled
    assert stored_row("stoppedloop")["next_run"] is None


def test_loops_warns_about_both_rows_and_loops_active_shows_only_the_enabled_one() -> None:
    store_loop("enabledloop", name="Enabled audit", next_run=_STORED_NEXT_RUN)
    store_loop("stoppedloop", name="Stopped audit", enabled=False)

    console, output = _console()
    loops_cmds._cmd_loops_list(Session(), console, [])

    listing = output.getvalue()
    assert listing.count(NEVER_RUNS) >= 2, listing
    assert "Active" not in listing
    assert "Invalid" not in listing, "the next-run cell alone must not read as malformed"

    console, output = _console()
    loops_cmds._cmd_loops_list(Session(), console, ["active"])

    active = output.getvalue()
    assert NEVER_RUNS in active
    assert "Enabled audit" in active
    assert "Stopped audit" not in active


@pytest.mark.parametrize("enabled", [True, False])
def test_loops_show_states_it_in_status_and_shows_the_sentence(enabled: bool) -> None:
    store_loop("impossible", enabled=enabled, next_run=_STORED_NEXT_RUN)
    console, output = _console()

    loop_show.show_loop(Session(), console, ["impossible"])

    shown = output.getvalue()
    assert NEVER_RUNS in shown
    assert REFUSAL_PHRASE in flat(shown)
    assert "Active" not in shown


def _painted_rows(monkeypatch: pytest.MonkeyPatch, columns: int) -> list[str]:
    monkeypatch.setattr(loop_show, "repl_tty_interactive", lambda: True)
    monkeypatch.setattr(choice_menu, "repl_tty_interactive", lambda: True)
    monkeypatch.setattr(choice_menu, "_read_action", lambda **_kwargs: "cancel")
    monkeypatch.setattr(choice_menu, "_cols", lambda: columns)
    painted = io.StringIO()
    monkeypatch.setattr(sys, "stdout", painted)
    session = Session()
    session.terminal.exclusive_stdin_active = True
    loop_show.show_loop(session, Console(file=io.StringIO()), [])
    plain = _ANSI.sub("", painted.getvalue()).replace("\r", "\n")
    return [row for row in plain.split("\n") if re.search(r"\d\. ", row)]


@pytest.mark.parametrize("columns", [40, 80])
def test_the_show_chooser_keeps_the_label_on_every_never_firing_row_at_width(
    monkeypatch: pytest.MonkeyPatch, columns: int
) -> None:
    store_loop("neverenabled", name="Daily audit")
    store_loop("neverlongnam", name=_LONG_NAME)
    store_loop("neverstopped", name="Stopped audit", enabled=False)
    store_loop("firingloop00", name="Fires daily", cron=FIRING_CRON)

    rows = _painted_rows(monkeypatch, columns)

    for identifier in ("neverenabled", "neverlongnam", "neverstopped"):
        row = next(row for row in rows if identifier in row)
        assert NEVER_RUNS in row, f"label clipped at {columns} columns: {row!r}"
    assert all(NEVER_RUNS not in row for row in rows if "firingloop00" in row)


def test_loops_next_reports_the_status_and_the_sentence_and_no_date() -> None:
    store_loop("impossible", next_run=_STORED_NEXT_RUN)
    console, output = _console()

    loops_cmds._cmd_loops_next(Session(), console, ["impossible"])

    shown = output.getvalue()
    assert "status: unschedulable" in shown
    assert REFUSAL_PHRASE in flat(shown)
    assert "2030" not in shown


def test_the_shell_cron_list_warns_and_does_not_say_active() -> None:
    store_loop("enabledloop")
    console, output = _console()

    cron_cmds.cmd_cron(Session(), console, ["list"], run_cli=lambda *_a, **_k: True)

    assert NEVER_RUNS in output.getvalue()
    assert "Active" not in output.getvalue()


def _picker_choices(monkeypatch: pytest.MonkeyPatch, module: Any) -> list[TargetChoice]:
    captured: list[TargetChoice] = []

    def capture(choices: Any, **_kwargs: Any) -> None:
        captured.extend(choices)

    monkeypatch.setattr(module, "build_target_picker", capture)
    return captured


def _cell_text(choice: TargetChoice) -> list[str]:
    return [cell.plain for cell in choice.cells]


@pytest.mark.parametrize("enabled", [True, False])
def test_the_loops_picker_labels_a_never_firing_row_everywhere_it_states_the_state(
    monkeypatch: pytest.MonkeyPatch, enabled: bool
) -> None:
    store_loop("impossible", enabled=enabled)
    choices = _picker_choices(monkeypatch, target_input)

    target_input.build_loop_picker(
        list_loop_summaries(), command="/loops run", action_label="Run now"
    )

    [choice] = choices
    assert choice.summary.startswith(NEVER_RUNS)
    assert choice.compact_summary.startswith(NEVER_RUNS)
    assert NEVER_RUNS in _cell_text(choice)
    assert not {"Active", "Invalid"}.intersection(_cell_text(choice))


@pytest.mark.parametrize(("sub", "listed"), [("start", "stoppedloop"), ("stop", "enabledloop")])
def test_loops_start_and_stop_pickers_list_only_what_their_filter_keeps(
    monkeypatch: pytest.MonkeyPatch, sub: str, listed: str
) -> None:
    store_loop("enabledloop")
    store_loop("stoppedloop", enabled=False)
    choices = _picker_choices(monkeypatch, target_input)
    monkeypatch.setattr(target_collection, "can_collect_input", lambda *_a: True)
    monkeypatch.setattr(target_collection, "run_command_input", lambda _app: None)

    target_collection.collect_loop_target(Session(), _console()[0], sub, [])

    assert [choice.value for choice in choices] == [listed]


def test_the_cron_picker_labels_a_never_firing_row(monkeypatch: pytest.MonkeyPatch) -> None:
    task = store_loop("impossible")
    choices = _picker_choices(monkeypatch, cron_target)

    cron_target.build_cron_target_picker([task], subcommand="run")

    [choice] = choices
    assert NEVER_RUNS in _cell_text(choice)
    assert not {"Active", "Invalid"}.intersection(_cell_text(choice))
    assert summarize_loop(task).next_run is None


def _shell_runners() -> SchedulerRunners:
    return SchedulerRunners(agent=lambda _payload: "")


@pytest.fixture
def real_shell_scheduler(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(loop_scheduler, "configure_process", lambda _profile: None)
    monkeypatch.setattr(adapters, "scheduler_runners", _shell_runners)
    monkeypatch.setattr(loop_scheduler, "_scheduler", None)
    monkeypatch.setattr(loop_scheduler, "_task_count", 0)
    monkeypatch.setattr(loop_scheduler, "_host_session", None)
    yield
    loop_scheduler.shutdown_loop_scheduler()


@pytest.mark.usefixtures("real_shell_scheduler")
def test_the_shell_counts_only_the_loop_that_can_fire() -> None:
    store_loop("firing", cron=FIRING_CRON)
    store_loop("impossible")
    console, output = _console()

    loops_cmds._reload_loop_scheduler(Session(), console)

    assert "scheduler: running 1 loop(s)" in output.getvalue()


@pytest.mark.usefixtures("real_shell_scheduler")
def test_the_shell_is_idle_when_only_an_impossible_loop_is_stored() -> None:
    store_loop("impossible")
    console, output = _console()

    loops_cmds._reload_loop_scheduler(Session(), console)

    assert "scheduler: idle" in output.getvalue()
