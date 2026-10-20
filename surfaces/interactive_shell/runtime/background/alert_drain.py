"""Drain of the alert inbox into the interactive-shell session and terminal."""

from __future__ import annotations

from rich.console import Console

from core.domain.alerts.inbox import AlertInbox
from surfaces.interactive_shell.session import Session
from surfaces.interactive_shell.ui.alerts import format_incoming_alert


def drain_and_render_incoming(
    session: Session,
    console: Console,
    inbox: AlertInbox,
) -> int:
    """Pop all queued alerts, render each one, and record them in session.

    Returns the number of alerts rendered.
    """
    alerts = inbox.iter_pending()
    count = 0

    for alert in alerts:
        console.print(format_incoming_alert(alert), end="\n")
        session.record_incoming_alert(alert)
        count += 1

    return count


__all__ = ["drain_and_render_incoming"]
