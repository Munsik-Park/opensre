"""CLI commands refuse a cron schedule that can never fire and warn about a stored one."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

import infrastructure.process.runtime_flags as runtime_flags
from config.constants.work_items import OPENSRE_WORK_ITEMS_DIR_ENV
from infrastructure.scheduling.scheduler.storage import list_tasks
from infrastructure.scheduling.scheduler.types import TaskKind
from infrastructure.terminal.theme import GLYPH_SUCCESS
from surfaces.cli.commands.cron import cron_command
from surfaces.cli.commands.posthog_report import posthog_command
from surfaces.cli.commands.sentry_digest import sentry_command
from surfaces.cli.commands.work import work_command
from tests.scheduler._stored_loops import (
    FIRING_CRON,
    IMPOSSIBLE_CRON,
    NEVER_RUNS,
    REFUSAL_PHRASE,
    flat,
    isolate_scheduler_home,
    store_loop,
)

_POSTHOG = "integrations.posthog.report_prerequisites"
_SENTRY = "integrations.sentry.digest_prerequisites"
_DELIVERY = ("--provider", "telegram", "--chat-id", "-100200300")
_SIX_FIELD_IMPOSSIBLE = "*/30 0 8 31 2 *"


@pytest.fixture(autouse=True)
def _isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_scheduler_home(monkeypatch, tmp_path)
    monkeypatch.setenv(OPENSRE_WORK_ITEMS_DIR_ENV, str(tmp_path / "work_items"))
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setattr(runtime_flags, "_flags", runtime_flags.RuntimeFlags())


@pytest.fixture
def integrations_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(f"{_POSTHOG}.require_posthog_integration", lambda: None)
    monkeypatch.setattr(f"{_POSTHOG}.require_report_delivery_provider", lambda _provider: None)
    monkeypatch.setattr(f"{_SENTRY}.require_sentry_integration", lambda: None)
    monkeypatch.setattr(f"{_SENTRY}.require_digest_delivery_provider", lambda _provider: None)


def _assert_refused_with_nothing_stored(result: Any) -> None:
    assert result.exit_code != 0, result.output
    assert REFUSAL_PHRASE in flat(result.output)
    assert list_tasks() == []


@pytest.mark.parametrize(
    "kind_args",
    [
        ("--kind", "manual_loop", "--prompt", "check"),
        (
            "--kind",
            "recurring_skill",
            "--skill",
            "reporting-github-ci-failures",
            "--owner",
            "acme",
            "--repo",
            "api",
            "--pr",
            "42",
        ),
    ],
)
def test_cron_add_refuses_an_impossible_schedule(kind_args: tuple[str, ...]) -> None:
    result = CliRunner().invoke(
        cron_command,
        ["add", *kind_args, "--cron", IMPOSSIBLE_CRON, "--provider", "interactive_shell"],
    )

    _assert_refused_with_nothing_stored(result)


@pytest.mark.usefixtures("integrations_ready")
@pytest.mark.parametrize(
    ("command", "path"),
    [
        (sentry_command, ("digest", "schedule", "add")),
        (posthog_command, ("report", "schedule", "add")),
    ],
)
def test_digest_and_report_schedules_refuse_an_impossible_schedule(
    command: Any, path: tuple[str, ...]
) -> None:
    result = CliRunner().invoke(command, [*path, "--cron", IMPOSSIBLE_CRON, *_DELIVERY])

    _assert_refused_with_nothing_stored(result)


@pytest.mark.usefixtures("integrations_ready")
def test_an_uptime_watch_on_an_impossible_schedule_announces_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[str] = []

    def deliver(_task: Any, message: str) -> tuple[bool, str, str]:
        sent.append(message)
        return True, "", "1"

    monkeypatch.setattr(
        "infrastructure.scheduling.scheduler.executor.deliver_scheduled_message", deliver
    )

    result = CliRunner().invoke(
        sentry_command, ["uptime", "watch", "add", "--cron", IMPOSSIBLE_CRON, *_DELIVERY]
    )

    _assert_refused_with_nothing_stored(result)
    assert sent == [], "a watch that can never fire announced itself active"


@pytest.mark.parametrize("cron", [IMPOSSIBLE_CRON, _SIX_FIELD_IMPOSSIBLE])
def test_work_schedule_checkin_says_why_an_impossible_schedule_is_refused(cron: str) -> None:
    result = CliRunner().invoke(work_command, ["schedule-checkin", "--cron", cron, *_DELIVERY])

    _assert_refused_with_nothing_stored(result)


def test_work_schedule_checkin_keeps_its_field_limit_for_a_six_field_cron_that_fires() -> None:
    result = CliRunner().invoke(
        work_command, ["schedule-checkin", "--cron", "*/30 * * * * *", *_DELIVERY]
    )

    assert result.exit_code != 0
    assert "cron must have exactly 5 fields" in flat(result.output)
    assert list_tasks() == []


def test_cron_list_warns_about_enabled_and_disabled_impossible_loops() -> None:
    store_loop("enabledloop")
    store_loop("stoppedloop", enabled=False)

    result = CliRunner().invoke(cron_command, ["list"])

    assert result.exit_code == 0, result.output
    assert result.output.count(NEVER_RUNS) >= 2, result.output
    assert "Active" not in result.output


def test_cron_list_json_reports_the_state_and_keeps_the_stored_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_loop("enabledloop", next_run="2030-01-01T08:00:00+00:00")
    monkeypatch.setattr(runtime_flags, "_flags", runtime_flags.RuntimeFlags(json=True))

    result = CliRunner().invoke(cron_command, ["list"])

    assert result.exit_code == 0, result.output
    [row] = json.loads(result.output)
    assert row["status"] == "unschedulable"
    assert REFUSAL_PHRASE in row["schedule_error"]
    assert row["enabled"] is True
    assert row["next_run"] is None


@pytest.mark.parametrize(
    ("kind", "command", "path"),
    [
        (TaskKind.SENTRY_MORNING_DIGEST, sentry_command, ("digest", "schedule", "list")),
        (TaskKind.SENTRY_UPTIME_WATCH, sentry_command, ("uptime", "watch", "list")),
        (TaskKind.POSTHOG_METRIC_REPORT, posthog_command, ("report", "schedule", "list")),
    ],
)
def test_per_kind_lists_warn_on_the_impossible_row_only(
    kind: TaskKind, command: Any, path: tuple[str, ...]
) -> None:
    store_loop("neverfiring", kind=kind)
    store_loop("firingtask", kind=kind, cron=FIRING_CRON)

    result = CliRunner().invoke(command, list(path))

    assert result.exit_code == 0, result.output
    rows = result.output.splitlines()
    [impossible] = [row for row in rows if "neverfiring" in row]
    [firing] = [row for row in rows if "firingtask" in row]
    assert NEVER_RUNS in impossible
    assert GLYPH_SUCCESS not in impossible
    assert GLYPH_SUCCESS in firing
    assert NEVER_RUNS not in firing


@pytest.mark.parametrize("columns", ["40", "80"])
@pytest.mark.parametrize(
    ("kind", "command", "path"),
    [
        (TaskKind.SENTRY_MORNING_DIGEST, sentry_command, ("digest", "schedule", "list")),
        (TaskKind.SENTRY_UPTIME_WATCH, sentry_command, ("uptime", "watch", "list")),
        (TaskKind.POSTHOG_METRIC_REPORT, posthog_command, ("report", "schedule", "list")),
    ],
)
def test_per_kind_lists_keep_the_label_whole_on_a_narrow_terminal(
    monkeypatch: pytest.MonkeyPatch,
    kind: TaskKind,
    command: Any,
    path: tuple[str, ...],
    columns: str,
) -> None:
    store_loop("neverfiring", kind=kind)
    monkeypatch.setenv("COLUMNS", columns)

    result = CliRunner().invoke(command, list(path))

    assert result.exit_code == 0, result.output
    assert NEVER_RUNS in result.output, result.output
