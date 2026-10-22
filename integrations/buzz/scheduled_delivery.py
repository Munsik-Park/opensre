"""Scheduled-delivery adapter: post a scheduled task's message to Buzz."""

from __future__ import annotations

import logging

from pydantic import ValidationError

from infrastructure.delivery.notifications.limits import MAX_MESSAGE_SIZE
from infrastructure.scheduling.scheduler.credentials import resolve_buzz_credentials
from infrastructure.scheduling.scheduler.types import ScheduledTask
from infrastructure.text.truncation import truncate
from integrations.buzz.delivery import post_buzz_message

logger = logging.getLogger(__name__)


class BuzzScheduledDelivery:
    """Deliver a scheduled task's message to its Buzz channel via the ``buzz`` CLI.

    Buzz renders Markdown, so the message gets no HTML transform. A relay URL, channel,
    message or credential the CLI cannot be handed returns a failure, never raises.
    """

    def deliver(self, task: ScheduledTask, message: str) -> tuple[bool, str, str]:
        creds = resolve_buzz_credentials(task.params)
        private_key = creds.get("private_key", "").strip()
        if not private_key:
            return False, "Missing private_key (BUZZ_PRIVATE_KEY) for Buzz", ""

        try:
            ok, error, event_id = post_buzz_message(
                creds.get("relay_url", ""),
                task.chat_id,
                truncate(message, MAX_MESSAGE_SIZE, suffix="…"),
                private_key,
                auth_tag=creds.get("auth_tag", ""),
                buzz_path=creds.get("buzz_path", ""),
            )
        except ValidationError as exc:
            # With the input included, the rejected relay URL and any credential in it are logged.
            logger.warning(
                "[buzz] scheduled delivery of task %s failed: invalid Buzz config %s",
                task.id,
                exc.errors(include_url=False, include_context=False, include_input=False),
            )
            return False, "Buzz relay_url (BUZZ_RELAY_URL) must start with http:// or https://", ""
        except ValueError:
            # subprocess refuses an argv element or environment value holding a NUL.
            logger.warning("[buzz] scheduled delivery of task %s failed", task.id, exc_info=True)
            return (
                False,
                "Buzz channel, message or credential holds a character the buzz CLI cannot take",
                "",
            )
        return (True, "", event_id) if ok else (False, error, "")
