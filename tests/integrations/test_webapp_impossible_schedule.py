"""The web app's loop sync refuses a cron schedule that can never fire and reports a stored one."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from infrastructure.scheduling.scheduler.storage import list_tasks
from integrations.webapp_scheduled_loops import WebappLoopSync
from tests.integrations.test_webapp_scheduled_loops import _ORG, _create_request, _sync, _Webapp
from tests.scheduler._stored_loops import (
    FIRING_CRON,
    IMPOSSIBLE_CRON,
    REFUSAL_PHRASE,
    isolate_scheduler_home,
    store_loop,
    stored_row,
)

_REQUEST_ID = "6c1d2e3f-0000-4000-8000-00000000000a"
_LAST_RUN = "2026-01-01T00:00:00+00:00"


@pytest.fixture(autouse=True)
def _scheduler_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    isolate_scheduler_home(monkeypatch, tmp_path)


def _request(action: str, loop_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "id": _REQUEST_ID,
        "action": action,
        "loop_id": loop_id,
        "payload": payload,
        "requested_by": "user_2abc",
    }


def _applied(*requests: dict[str, Any]) -> tuple[_Webapp, WebappLoopSync]:
    webapp = _Webapp(pending=list(requests))
    sync = _sync(webapp)
    sync.sync()
    return webapp, sync


def test_a_create_with_an_impossible_schedule_fails_and_stores_nothing() -> None:
    create = _create_request()
    create["payload"]["cron"] = IMPOSSIBLE_CRON

    webapp, _ = _applied(create)

    [(_, ack)] = webapp.acks
    assert ack["status"] == "failed"
    assert REFUSAL_PHRASE in ack["error"]
    assert list_tasks() == []


def test_an_update_to_an_impossible_cron_fails_and_leaves_the_row() -> None:
    store_loop("editedloop", cron=FIRING_CRON, organization=_ORG)
    before = stored_row("editedloop")

    webapp, _ = _applied(_request("update", "editedloop", {"cron": IMPOSSIBLE_CRON}))

    [(_, ack)] = webapp.acks
    assert ack["status"] == "failed"
    assert REFUSAL_PHRASE in ack["error"]
    assert stored_row("editedloop") == before


def test_a_rename_resending_the_stored_impossible_cron_is_applied() -> None:
    store_loop("kept", cron="0 8  31 2 *", organization=_ORG)

    webapp, _ = _applied(_request("update", "kept", {"name": "Renamed", "cron": IMPOSSIBLE_CRON}))

    [(_, ack)] = webapp.acks
    assert ack["status"] == "applied"
    assert stored_row("kept")["name"] == "Renamed"
    assert stored_row("kept")["cron"] == "0 8  31 2 *"


@pytest.mark.parametrize("enabled", [True, False])
def test_a_resume_of_an_impossible_loop_fails_and_it_keeps_its_flag(enabled: bool) -> None:
    store_loop("stoppedloop", enabled=enabled, organization=_ORG)

    webapp, _ = _applied(_request("resume", "stoppedloop"))

    [(_, ack)] = webapp.acks
    assert ack["status"] == "failed"
    assert REFUSAL_PHRASE in ack["error"]
    assert stored_row("stoppedloop")["enabled"] is enabled


@pytest.mark.parametrize(("last_run", "word"), [(_LAST_RUN, "paused"), (None, "draft")])
@pytest.mark.parametrize("enabled", [True, False])
def test_the_row_stays_within_the_apps_words_and_carries_the_warning(
    last_run: str | None, word: str, enabled: bool
) -> None:
    store_loop(
        "impossible",
        enabled=enabled,
        organization=_ORG,
        last_run=last_run,
        next_run="2030-01-01T08:00:00+00:00",
    )

    webapp, _ = _applied()

    [row] = webapp.reports[-1]
    assert row["status"] == word
    assert REFUSAL_PHRASE in row["schedule_error"]
    assert row["next_run"] is None
