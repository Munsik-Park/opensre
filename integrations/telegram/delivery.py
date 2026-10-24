"""Telegram delivery helper - posts investigation findings to Telegram Bot API."""

from __future__ import annotations

import bisect
import contextlib
import logging
import re
import string
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from platform.common.truncation import truncate
from platform.notifications.delivery_transport import post_json
from platform.notifications.limits import MAX_MESSAGE_SIZE
from platform.notifications.redaction import redact_token

logger = logging.getLogger(__name__)

_MESSAGE_LIMIT = MAX_MESSAGE_SIZE
_BOT_TOKEN_RE = re.compile(r"(bot)[^/]+(/)")
_TELEGRAM_HTML_TAG_NAMES = frozenset(
    {
        "a",
        "b",
        "strong",
        "i",
        "em",
        "u",
        "ins",
        "s",
        "strike",
        "del",
        "code",
        "pre",
        "span",
        "tg-spoiler",
        "tg-emoji",
        "tg-time",
        "blockquote",
    }
)
# Telegram's parser ends a tag name at these only; ``str.isspace`` also matches
# ``\f`` and Unicode spaces, which Telegram reads as part of the name.
_TELEGRAM_HTML_SPACE = frozenset(" \t\r\n\v\0")
_CHAR_REF_CHARS = frozenset("#" + string.ascii_letters + string.digits)
# Telegram decodes no ``&`` run longer than this (``&#1114110``); a cut through a
# longer run leaves literal text, not a broken reference.
_MAX_CHAR_REF_LEN = 9


@dataclass(frozen=True, slots=True)
class _HtmlTag:
    start: int
    end: int
    name: str
    closing: bool


def _tag_name_end(text: str, pos: int) -> int:
    while pos < len(text) and text[pos] not in _TELEGRAM_HTML_SPACE and text[pos] != ">":
        pos += 1
    return pos


def _start_tag_end(text: str, pos: int) -> int:
    """Return the offset just past the ``>`` ending a start tag, or ``len(text)``.

    A quoted attribute value may hold ``>``; Telegram reads a quote as opening a
    value only right after ``=`` and optional whitespace.
    """
    while pos < len(text):
        char = text[pos]
        if char == ">":
            return pos + 1
        pos += 1
        if char != "=":
            continue
        while pos < len(text) and text[pos] in _TELEGRAM_HTML_SPACE:
            pos += 1
        if pos < len(text) and text[pos] in "\"'":
            close = text.find(text[pos], pos + 1)
            if close == -1:
                return len(text)
            pos = close + 1
    return len(text)


def _read_html_tags(text: str) -> list[_HtmlTag]:
    """Read the tags of *text* as Telegram's ``parse_mode=HTML`` parser does.

    A ``<`` that starts neither a start tag of a supported name nor an end tag
    is one literal character.
    """
    tags: list[_HtmlTag] = []
    pos = text.find("<")
    while pos != -1:
        closing = text.startswith("/", pos + 1)
        name_start = pos + 2 if closing else pos + 1
        name_end = _tag_name_end(text, name_start)
        name = text[name_start:name_end]
        if name.isascii() and (name.lower() in _TELEGRAM_HTML_TAG_NAMES or (closing and not name)):
            if closing:
                gt = text.find(">", name_end)
                end = len(text) if gt == -1 else gt + 1
            else:
                end = _start_tag_end(text, name_end)
            tags.append(_HtmlTag(pos, end, name.lower(), closing))
            pos = text.find("<", end)
        else:
            pos = text.find("<", pos + 1)
    return tags


def _cut_outside_tags(tags: list[_HtmlTag], starts: list[int], cut: int) -> int:
    idx = bisect.bisect_left(starts, cut) - 1
    if idx >= 0 and tags[idx].end > cut:
        return tags[idx].start
    return cut


def _cut_before_split_char_ref(text: str, cut: int) -> int:
    """Move *cut* before a trailing ``&`` run only when the cut split that run."""
    amp = text.rfind("&", max(0, cut - _MAX_CHAR_REF_LEN), cut)
    if amp == -1 or not all(char in _CHAR_REF_CHARS for char in text[amp + 1 : cut]):
        return cut
    if text[cut] in _CHAR_REF_CHARS or text[cut] == ";":
        return amp
    return cut


def _closing_tags(open_and_close: list[_HtmlTag]) -> str:
    """Closing tags for the elements *open_and_close* leaves open, innermost first.

    An end tag closes the innermost open element whatever its name, as ``</>`` does.
    """
    stack: list[str] = []
    for tag in open_and_close:
        if not tag.closing:
            stack.append(tag.name)
        elif stack:
            stack.pop()
    return "".join(f"</{name}>" for name in reversed(stack))


def truncate_for_telegram_html(text: str, max_len: int, suffix: str = "…") -> str:
    """Truncate *text* for Telegram ``parse_mode=HTML`` without breaking markup at the cut."""
    if len(text) <= max_len:
        return text
    if max_len <= len(suffix):
        return suffix[:max_len]
    tags = _read_html_tags(text)
    starts = [tag.start for tag in tags]
    room = max_len - len(suffix)
    while room > 0:
        cut = _cut_before_split_char_ref(text, _cut_outside_tags(tags, starts, room))
        closers = _closing_tags(tags[: bisect.bisect_left(starts, cut)])
        candidate = text[:cut] + closers + suffix
        if len(candidate) <= max_len:
            return candidate
        room -= 1
    return suffix[:max_len]


def _redact_arg(a: object) -> object:
    """Redact bot token from a log arg, preserving the original type if no match."""
    s = str(a)
    redacted = _BOT_TOKEN_RE.sub(r"\1<redacted>\2", s)
    return redacted if redacted != s else a


class _TelegramTokenFilter(logging.Filter):
    """Scrub Telegram bot tokens from httpx/httpcore log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _BOT_TOKEN_RE.sub(r"\1<redacted>\2", str(record.msg))
        if record.args:
            if isinstance(record.args, tuple):
                record.args = tuple(_redact_arg(a) for a in record.args)
            elif isinstance(record.args, dict):
                record.args = {k: _redact_arg(v) for k, v in record.args.items()}
        return True


def _install_httpx_token_filter() -> None:
    _filter = _TelegramTokenFilter()
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).addFilter(_filter)


_install_httpx_token_filter()


def post_telegram_message(
    chat_id: str,
    text: str,
    bot_token: str,
    parse_mode: str = "",
    reply_to_message_id: str = "",
    reply_markup: dict[str, Any] | None = None,
) -> tuple[bool, str, str]:
    """Call Telegram Bot API sendMessage endpoint.

    Returns (success, error, message_id).
    """
    logger.debug("[telegram] post message params chat_id: %s", chat_id)
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup:
        payload["reply_markup"] = reply_markup
    if reply_to_message_id and reply_to_message_id != "0":
        with contextlib.suppress(ValueError, TypeError):
            payload["reply_to_message_id"] = int(reply_to_message_id)
    response = post_json(
        url=f"https://api.telegram.org/bot{bot_token}/sendMessage",
        payload=payload,
    )
    if not response.ok:
        error = redact_token(response.error, bot_token)
        logger.warning("[telegram] post message exception: %s", error)
        return False, error, ""
    if response.status_code != HTTPStatus.OK:
        logger.warning("[telegram] post message failed: %s", response.status_code)
        if response.data:
            error_message = str(
                response.data.get("description", response.data.get("error", "unknown"))
            )
        else:
            error_message = response.text or f"HTTP {response.status_code}"
        logger.warning("[telegram] post message failed: %s", error_message)
        return False, error_message, ""
    result = response.data.get("result", {})
    message_id = str(result.get("message_id") or "") if isinstance(result, dict) else ""
    return True, "", message_id


def send_telegram_report(
    report: str,
    telegram_ctx: dict[str, Any],
    *,
    parse_mode: str = "HTML",
    reply_markup: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Send a truncated report to Telegram. Returns (success, error)."""
    bot_token: str = str(telegram_ctx.get("bot_token") or "")
    chat_id: str = str(telegram_ctx.get("chat_id") or "")
    if not bot_token or not chat_id:
        return False, "Missing bot_token or chat_id"
    reply_to_message_id: str = str(telegram_ctx.get("reply_to_message_id") or "")
    if parse_mode.upper() == "HTML":
        text = truncate_for_telegram_html(report, _MESSAGE_LIMIT, suffix="…")
    else:
        text = truncate(report, _MESSAGE_LIMIT, suffix="…")
    post_success, error, _ = post_telegram_message(
        chat_id,
        text,
        bot_token,
        parse_mode=parse_mode,
        reply_to_message_id=reply_to_message_id,
        reply_markup=reply_markup,
    )
    return (True, "") if post_success else (False, error)
