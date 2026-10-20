"""Drain of the alert inbox into the interactive-shell session and terminal."""

from __future__ import annotations

import logging

from rich.console import Console

from core.domain.alerts.inbox import AlertInbox
from surfaces.interactive_shell.session import Session
from surfaces.interactive_shell.ui.alerts import format_incoming_alert

log = logging.getLogger(__name__)


def drain_and_render_incoming(
    session: Session,
    console: Console,
    inbox: AlertInbox,
) -> int:
    """Record, acknowledge, then render each due alert; return how many were recorded.

    An alert whose recording raises stays in the inbox for a later retry; one whose
    rendering raises stays recorded. Neither failure stops the remaining alerts.
    """
    recorded = 0
    for due in inbox.due():
        try:
            session.record_incoming_alert(due.alert)
        except Exception as exc:
            inbox.defer(due.handle)
            log.warning("Error recording incoming alert; will retry: %s", exc)
            continue
        inbox.acknowledge(due.handle)
        recorded += 1
        try:
            console.print(format_incoming_alert(due.alert), end="\n")
        except Exception as exc:
            log.warning("Error rendering incoming alert: %s", exc)
    return recorded


__all__ = ["drain_and_render_incoming"]
