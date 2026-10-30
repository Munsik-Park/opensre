from __future__ import annotations

from pathlib import Path

import pytest

from infrastructure.scheduling.scheduler.storage import list_tasks
from tests.scheduler._stored_loops import (
    FIRING_CRON,
    IMPOSSIBLE_CRON,
    REFUSAL_PHRASE,
    isolate_scheduler_home,
    store_loop,
    stored_row,
)
from tools.system.scheduled_loops import manage
from tools.system.scheduled_loops.tool import list_scheduled_loops
from tools.system.work_items import work_task_schedule_checkin


@pytest.fixture(autouse=True)
def _scheduler_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_scheduler_home(monkeypatch, tmp_path)


@pytest.fixture
def reloads(monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    requested: list[bool] = []
    monkeypatch.setattr(manage, "request_scheduler_reload", lambda: requested.append(True))
    return requested


def test_the_loop_tool_refuses_an_impossible_schedule(reloads: list[bool]) -> None:
    result = manage.schedule_prompt_loop(prompt="check", cron=IMPOSSIBLE_CRON)

    assert result["ok"] is False
    assert REFUSAL_PHRASE in result["error"]
    assert reloads == []
    assert list_tasks() == []


@pytest.mark.parametrize("cron", [IMPOSSIBLE_CRON, "*/30 0 8 31 2 *"])
def test_the_work_item_tool_says_why_an_impossible_schedule_is_refused(cron: str) -> None:
    result = work_task_schedule_checkin(cron=cron, provider="slack", chat_id="C12345")

    assert result.get("error") == "invalid_cron"
    assert REFUSAL_PHRASE in result["detail"]
    assert list_tasks() == []


def test_the_edit_tool_refuses_a_change_to_an_impossible_cron(reloads: list[bool]) -> None:
    store_loop("editedloop", cron=FIRING_CRON)
    before = stored_row("editedloop")

    result = manage.edit_scheduled_loop(loop="editedloop", cron=IMPOSSIBLE_CRON)

    assert result["ok"] is False
    assert REFUSAL_PHRASE in result["error"]
    assert reloads == []
    assert stored_row("editedloop") == before


@pytest.mark.parametrize("enabled", [True, False])
def test_the_stop_tool_refuses_to_resume_an_impossible_loop(
    reloads: list[bool], enabled: bool
) -> None:
    store_loop("stoppedloop", enabled=enabled)

    result = manage.stop_scheduled_loop(loop="stoppedloop", action="resume")

    assert result["ok"] is False
    assert REFUSAL_PHRASE in result["error"]
    assert reloads == []
    assert stored_row("stoppedloop")["enabled"] is enabled


def test_the_listing_tool_does_not_report_an_impossible_loop_as_active() -> None:
    store_loop("impossible")

    result = list_scheduled_loops()

    [row] = result["loops"]
    assert row["status"] == "unschedulable"
    assert REFUSAL_PHRASE in row["health"]
    assert row["needs_attention"] is True
    assert result["response_text"].startswith("1 scheduled loop, 0 active")


def test_the_listing_tools_headline_still_counts_a_firing_and_a_stopped_loop_by_status() -> None:
    store_loop("impossible")
    store_loop("firing", cron=FIRING_CRON)
    store_loop("stopped", cron=FIRING_CRON, enabled=False)

    result = list_scheduled_loops()

    assert result["response_text"].startswith("3 scheduled loops, 1 active")


def test_filtering_the_listing_tool_keeps_an_enabled_impossible_loop_and_not_a_disabled_one() -> (
    None
):
    store_loop("impossible")
    store_loop("stopped", enabled=False)

    result = list_scheduled_loops(include_disabled=False)

    [row] = result["loops"]
    assert row["status"] == "unschedulable"
    assert "stopped" not in row["id"]
