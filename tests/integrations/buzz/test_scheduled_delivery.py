from __future__ import annotations

import json
import logging
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
        params={"private_key": f"  {_KEY}  ", "relay_url": "http://relay.test"},
    )


def test_scheduled_buzz_keeps_a_padded_private_key_off_argv_and_out_of_the_error_and_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
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

    with caplog.at_level(logging.WARNING):
        ok, error, message_id = BuzzScheduledDelivery().deliver(_task(), "Daily report")

    assert len(children) == 1, "the buzz CLI must be run once"
    assert all(_KEY not in arg for arg in children[0]["argv"]), "private key on the buzz argv"
    assert children[0]["env"]["BUZZ_PRIVATE_KEY"] == _KEY
    assert ok is False
    assert _KEY not in error
    assert "<redacted>" in error
    assert message_id == ""
    assert "<redacted>" in caplog.text, "the send failure must be logged"
    assert _KEY not in caplog.text, "private key in the log"
