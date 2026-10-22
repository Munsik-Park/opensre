"""Private-key handling of scheduled Buzz delivery."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from infrastructure.scheduling.scheduler.types import Provider, ScheduledTask, TaskKind
from integrations.buzz.scheduled_delivery import BuzzScheduledDelivery

_KEY = "nsec1scheduledbuzzkeyunique"


def _task() -> ScheduledTask:
    return ScheduledTask(
        id="buzz-redaction",
        kind=TaskKind.MANUAL_LOOP,
        cron="0 9 * * *",
        provider=Provider.BUZZ,
        chat_id="3f6c2a8e-buzz-channel",
        params={"private_key": _KEY, "relay_url": "http://relay.test"},
    )


def test_scheduled_buzz_keeps_the_private_key_off_argv_and_out_of_the_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The binary must resolve, or the send path fails before any child runs.
    buzz = tmp_path / "buzz"
    buzz.write_text("")
    for name in ("BUZZ_PRIVATE_KEY", "BUZZ_RELAY_URL", "BUZZ_AUTH_TAG", "BUZZ_DEFAULT_CHANNEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BUZZ_PATH", str(buzz))
    children: list[dict[str, Any]] = []

    def _buzz_child(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        children.append({"argv": cmd, "env": kwargs["env"]})
        stderr = json.dumps({"error": "auth", "message": f"auth rejected for {_KEY}"})
        return subprocess.CompletedProcess(cmd, 3, stdout="", stderr=stderr)

    monkeypatch.setattr("integrations.buzz.client.subprocess.run", _buzz_child)

    ok, error, message_id = BuzzScheduledDelivery().deliver(_task(), "Daily report")

    assert len(children) == 1, "the buzz CLI must be run once"
    assert all(_KEY not in arg for arg in children[0]["argv"]), "private key on the buzz argv"
    assert children[0]["env"]["BUZZ_PRIVATE_KEY"] == _KEY
    assert ok is False
    assert _KEY not in error
    assert "<redacted>" in error
    assert message_id == ""
