"""In-process alert inbox — the domain queue for external alert pushes.

Owns the queue and the process-wide current-inbox handle. HTTP intake is
``POST /alerts`` on the gateway web app.
"""

from __future__ import annotations

import itertools
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from time import monotonic

from config.strict_config import StrictConfigModel

_DEFAULT_MAX_INBOX = 256
_RETRY_DELAY_SECONDS = 5.0


class IncomingAlert(StrictConfigModel):
    text: str
    alert_name: str | None = None
    severity: str | None = None
    source: str | None = None
    received_at: datetime | None = None


@dataclass(frozen=True)
class DueAlert:
    """A queued alert handed to a drain, with the handle naming its inbox entry."""

    handle: int
    alert: IncomingAlert


@dataclass
class _Entry:
    alert: IncomingAlert
    retry_at: float | None = None


class AlertInbox:
    """Bounded queue of received alerts; an alert stays queued until a drain acknowledges it.

    Every queued alert, including one waiting to retry, counts against ``maxsize``.
    """

    def __init__(self, maxsize: int = _DEFAULT_MAX_INBOX) -> None:
        self._entries: OrderedDict[int, _Entry] = OrderedDict()
        self._handles = itertools.count()
        self._maxsize = maxsize
        self._dropped: int = 0
        self._lock = threading.Lock()
        self._pending_event = threading.Event()

    def put(self, alert: IncomingAlert) -> bool:
        """Return True if queued without eviction, False if an old alert was dropped."""
        with self._lock:
            evicted = len(self._entries) >= self._maxsize
            if evicted:
                self._entries.popitem(last=False)
                self._dropped += 1
            self._entries[next(self._handles)] = _Entry(alert)
            self._pending_event.set()
        return not evicted

    def pop_nowait(self) -> IncomingAlert | None:
        with self._lock:
            if not self._entries:
                return None
            return self._entries.popitem(last=False)[1].alert

    def due(self) -> list[DueAlert]:
        """Return the alerts not waiting out a retry delay, oldest first; clears the wake event."""
        now = monotonic()
        with self._lock:
            self._pending_event.clear()
            return [
                DueAlert(handle, entry.alert)
                for handle, entry in self._entries.items()
                if entry.retry_at is None or entry.retry_at <= now
            ]

    def acknowledge(self, handle: int) -> None:
        """Remove a recorded alert; a no-op once ``put`` has evicted its entry."""
        with self._lock:
            self._entries.pop(handle, None)

    def defer(self, handle: int) -> None:
        """Leave a failed alert queued, not due again until the retry delay has passed."""
        retry_at = monotonic() + _RETRY_DELAY_SECONDS
        with self._lock:
            entry = self._entries.get(handle)
            if entry is not None:
                entry.retry_at = retry_at

    def peek_last(self, n: int) -> list[IncomingAlert]:
        with self._lock:
            items = [entry.alert for entry in self._entries.values()]
            return items[-n:]

    @property
    def qsize(self) -> int:
        return len(self._entries)

    @property
    def dropped(self) -> int:
        return self._dropped

    @property
    def pending_event(self) -> threading.Event:
        """Event set by ``put`` and cleared by ``due``; deferring an alert never sets it."""
        return self._pending_event


_current_inbox: AlertInbox | None = None


def set_current_inbox(inbox: AlertInbox | None) -> None:
    global _current_inbox
    _current_inbox = inbox


def get_current_inbox() -> AlertInbox | None:
    return _current_inbox


__all__ = [
    "AlertInbox",
    "DueAlert",
    "IncomingAlert",
    "get_current_inbox",
    "set_current_inbox",
]
