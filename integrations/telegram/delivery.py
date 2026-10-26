"""Telegram delivery helper - posts messages to the Telegram Bot API."""

from __future__ import annotations

import contextlib
import logging
import re
import string
from http import HTTPStatus
from typing import Any

from infrastructure.delivery.notifications.delivery_transport import post_json
from infrastructure.delivery.notifications.limits import MAX_MESSAGE_SIZE
from infrastructure.delivery.notifications.redaction import redact_token
from infrastructure.text.truncation import truncate

logger = logging.getLogger(__name__)

_MESSAGE_LIMIT = MAX_MESSAGE_SIZE
_BOT_TOKEN_RE = re.compile(r"(bot)[^/]+(/)")
# Every element Telegram's ``parse_mode=HTML`` parser accepts, so a cut taken while
# one is open leaves markup the Bot API rejects.
_BALANCED_TAG_NAMES = frozenset(
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
_LONGEST_TAG_NAME = max(len(name) for name in _BALANCED_TAG_NAMES)
# The parser skips only these between a tag's parts, never form feed or a non-ASCII
# space; a name or value also ends at NUL, which it reads as the end of the text.
_TAG_SPACE = frozenset(" \t\r\n\v")
_TAG_NAME_STOP = _TAG_SPACE | {"\0", ">"}
_ATTRIBUTE_NAME_STOP = _TAG_NAME_STOP | {"=", "/", '"', "'"}
_UNQUOTED_VALUE_CHARS = frozenset(string.ascii_letters + string.digits + ".-")
_CHARACTER_REFERENCE_CHARS = frozenset(string.ascii_letters + string.digits + "#")

_Tag = tuple[int, int, str | None]


def _skip_tag_space(text: str, pos: int) -> int:
    while text[pos] in _TAG_SPACE:
        pos += 1
    return pos


def _tag_name_end(text: str, start: int) -> int:
    """Return where the tag name at *start* ends, reading one past the longest set name at most."""
    pos = start
    limit = start + _LONGEST_TAG_NAME + 1
    while pos < limit and text[pos] not in _TAG_NAME_STOP:
        pos += 1
    return pos


def _set_tag_name(raw: str) -> str | None:
    """Return the element *raw* names, its case folded for ASCII letters only, or None."""
    name = raw.lower() if raw.isascii() else ""
    return name if name in _BALANCED_TAG_NAMES else None


def _start_tag_end(text: str, pos: int, known: dict[int, int | None]) -> int | None:
    """Return the index past the ``>`` that ends a start tag whose attributes begin at *pos*.

    None when the parser refuses the tag before that ``>``. *text* ends with a NUL. *known*
    maps each position where the parser's attribute loop restarts to the result read from
    there, so text that several unfinished tags run over is read once per call.
    """
    restarts: list[int] = []
    end: int | None = None
    while True:
        if pos in known:
            end = known[pos]
            break
        restarts.append(pos)
        pos = _skip_tag_space(text, pos)
        if text[pos] == ">":
            end = pos + 1
            break
        name_start = pos
        while text[pos] not in _ATTRIBUTE_NAME_STOP:
            pos += 1
        if pos == name_start:
            break
        pos = _skip_tag_space(text, pos)
        if text[pos] != "=":
            if text[pos] == "\0":
                break
            continue
        pos = _skip_tag_space(text, pos + 1)
        quote = text[pos]
        if quote in ('"', "'"):
            pos += 1
            while text[pos] != quote and text[pos] != "\0":
                pos += 1
            if text[pos] == quote:
                pos += 1
        else:
            while text[pos] in _UNQUOTED_VALUE_CHARS:
                pos += 1
            if text[pos] not in _TAG_SPACE and text[pos] != ">":
                break
        if text[pos] == "\0":
            break
    for restart in restarts:
        known[restart] = end
    return end


def _read_tag(text: str, pos: int, known: dict[int, int | None]) -> tuple[int, str | None] | None:
    """Read the tag whose ``<`` is at *pos* as Telegram's parser reads it.

    Return ``(end, name)``: *end* is the index past the tag's ``>``, *name* the element a
    start tag opens, or None for an end tag. Return None when that ``<`` is text: it starts
    no tag of the set, or the parser refuses the tag before its ``>``. *text* ends with a NUL.
    """
    if text[pos + 1] == "/":
        name_start = pos + 2
        name_end = _tag_name_end(text, name_start)
        close = _skip_tag_space(text, name_end)
        if text[close] != ">":
            return None
        if name_end > name_start and _set_tag_name(text[name_start:name_end]) is None:
            return None
        return close + 1, None
    name_end = _tag_name_end(text, pos + 1)
    name = _set_tag_name(text[pos + 1 : name_end])
    if name is None or text[name_end] == "\0":
        return None
    end = _start_tag_end(text, name_end, known)
    return None if end is None else (end, name)


def _read_tags(text: str, limit: int) -> list[_Tag]:
    """Return ``(start, end, name)`` for each tag of the set the scan meets before *limit*.

    A tag is read on the whole of *text*, so its end may lie past *limit*. The scan resumes
    right after a ``<`` that is text.
    """
    padded = text + "\0"
    known: dict[int, int | None] = {}
    tags: list[_Tag] = []
    pos = padded.find("<", 0, limit)
    while pos != -1:
        tag = _read_tag(padded, pos, known)
        if tag is None:
            pos = padded.find("<", pos + 1, limit)
            continue
        end, name = tag
        tags.append((pos, end, name))
        pos = padded.find("<", end, limit)
    return tags


def _cut(text: str, room: int, tags: list[_Tag]) -> tuple[str, list[str]]:
    """Return the text kept when *text* is cut at *room*, and the elements open where it ends.

    Only what the cut left behind is dropped: the tag the cut lands inside, or else a trailing
    ``&`` followed only by characters a character reference may hold.
    """
    open_names: list[str] = []
    for start, end, name in tags:
        if start >= room:
            break
        if end > room:
            return text[:start], open_names
        if name is not None:
            open_names.append(name)
        elif open_names:
            open_names.pop()
    amp = text.rfind("&", 0, room)
    if amp != -1 and all(char in _CHARACTER_REFERENCE_CHARS for char in text[amp + 1 : room]):
        return text[:amp], open_names
    return text[:room], open_names


def truncate_for_telegram_html(text: str, max_len: int, suffix: str = "…") -> str:
    """Truncate *text* for Telegram ``parse_mode=HTML`` without breaking markup at the cut."""
    if len(text) <= max_len:
        return text
    if max_len <= len(suffix):
        return suffix[:max_len]
    room = max_len - len(suffix)
    tags = _read_tags(text, room)
    while room > 0:
        kept, open_names = _cut(text, room, tags)
        candidate = kept + "".join(f"</{name}>" for name in reversed(open_names)) + suffix
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
    *,
    link_previews: bool = True,
) -> tuple[bool, str, str]:
    """Call Telegram Bot API sendMessage endpoint.

    ``link_previews=False`` posts without the preview card Telegram adds for
    the first link (a report lists many links; previews bury it).
    Returns (success, error, message_id).
    """
    logger.debug("[telegram] post message params chat_id: %s", chat_id)
    payload: dict[str, Any] = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if reply_markup:
        payload["reply_markup"] = reply_markup
    if not link_previews:
        payload["link_preview_options"] = {"is_disabled": True}
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
        safe_error = redact_token(error_message, bot_token)
        logger.warning("[telegram] post message failed: %s", safe_error)
        return False, safe_error, ""
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
