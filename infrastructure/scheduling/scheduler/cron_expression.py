"""Cron expression parsing shared by the scheduler runner and every ``add`` command.

Two shapes are accepted: the standard five-field crontab line
(``minute hour day month day_of_week``) and a six-field line with a leading
seconds field (``second minute hour day month day_of_week``). The seconds form
exists for polling loops such as the CI repair loop, which must react within
half a minute; a five-field line can never fire more than once per minute.

Weekday numbers mean what they mean in crontab (0 and 7 are Sunday, 1 is
Monday), although APScheduler 3.x counts from 0 = Monday; every trigger is
built through :func:`build_cron_trigger`, which translates them.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from itertools import groupby
from typing import Any

#: A standard crontab line: minute, hour, day, month, day_of_week.
CRON_FIELD_COUNT = 5
#: A crontab line with a leading seconds field.
CRON_FIELD_COUNT_WITH_SECONDS = 6

CRON_FORMAT_HELP = (
    "minute hour day month day_of_week, "
    "or second minute hour day month day_of_week for sub-minute polling"
)
CRON_FIELD_COUNT_ERROR = (
    f"cron expression must have {CRON_FIELD_COUNT} fields"
    f" ({CRON_FIELD_COUNT_WITH_SECONDS} with a leading seconds field)"
)

#: APScheduler 3.x day names, indexed like ``datetime.weekday()`` (0 = Monday).
_WEEKDAY_NAMES = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
#: Crontab weekday numbers run 0-7; both 0 and 7 are Sunday.
_CRONTAB_LAST_WEEKDAY = 7
#: Crontab numbers of the day names, ``sun`` being 0.
_CRONTAB_WEEKDAYS = {
    name: (index + 1) % len(_WEEKDAY_NAMES) for index, name in enumerate(_WEEKDAY_NAMES)
}
_MONTH_NAMES = ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
#: Crontab numbers of the month names, ``jan`` being 1.
_CRONTAB_MONTHS = {name: number for number, name in enumerate(_MONTH_NAMES, start=1)}
#: The text APScheduler 3.x reads as the last day of the month, matched as it matches it.
_LAST_DAY = re.compile(r"last", re.IGNORECASE)


def _item_grammar(names: dict[str, int]) -> re.Pattern[str]:
    """The grammar of one lower-case item of a field that takes names.

    An item is ``*`` or an endpoint with an optional ``-endpoint``, then an
    optional ``/step``; an endpoint is digits or a whole name from ``names``.
    """
    endpoint = "|".join([r"\d+", *names])
    return re.compile(
        rf"(?:(?P<every>\*)|(?P<first>{endpoint})(?:-(?P<last>{endpoint}))?)(?:/(?P<step>\d+))?"
    )


_DAY_OF_WEEK_ITEM = _item_grammar(_CRONTAB_WEEKDAYS)
_MONTH_ITEM = _item_grammar(_CRONTAB_MONTHS)


def _crontab_number(endpoint: str, names: dict[str, int]) -> int:
    """The crontab number of an endpoint its field's grammar matched."""
    return int(endpoint) if endpoint.isdigit() else names[endpoint]


def day_of_week_names(field: str) -> str:
    """Spell a crontab ``day_of_week`` field in the day names APScheduler 3.x reads.

    Each item, in numbers, day names or both (``1-5/2``, ``mon-fri/2``,
    ``mon-5``, ``*/2``), is read the crontab way and its days become names; a
    step is expanded into a list, because APScheduler ignores a step written
    after a name range. Raises ``ValueError``, naming the item as written, for
    an item it cannot read in full, a number above 7, a backwards range, or a
    zero step.
    """
    days: set[int] = set()
    for item in field.split(","):
        match = _DAY_OF_WEEK_ITEM.fullmatch(item.lower())
        if match is None:
            raise ValueError(
                f"day_of_week item is not a day (0-7 or mon-sun), a range or a step: {item!r}"
            )
        days.update(_crontab_weekdays(item, match))
    return ",".join(_weekday_ranges(days))


def _crontab_weekdays(item: str, match: re.Match[str]) -> set[int]:
    """APScheduler weekday indexes for one crontab item."""
    step = int(match["step"] or 1)
    if match["every"]:
        first, last = 0, _CRONTAB_LAST_WEEKDAY
    else:
        first = _crontab_number(match["first"], _CRONTAB_WEEKDAYS)
        if match["last"] is not None:
            last = _crontab_number(match["last"], _CRONTAB_WEEKDAYS)
            # As in crontab, a range that starts above 0 ends on ``sun`` as 7 (``fri-sun``).
            if match["last"] == "sun" and first > 0:
                last = _CRONTAB_LAST_WEEKDAY
        else:
            # ``N/step`` runs to the end of the week, like ``*/step``.
            last = first if match["step"] is None else _CRONTAB_LAST_WEEKDAY
    if max(first, last) > _CRONTAB_LAST_WEEKDAY:
        raise ValueError(f"day_of_week numbers run 0-7, with 0 and 7 for Sunday: {item!r}")
    if first > last:
        raise ValueError(f"day_of_week range runs backwards: {item!r}")
    if step < 1:
        raise ValueError(f"day_of_week step must be at least 1: {item!r}")
    # Crontab 1 (Monday) is APScheduler 0; crontab 0 and 7 (Sunday) are both 6.
    return {(day - 1) % len(_WEEKDAY_NAMES) for day in range(first, last + 1, step)}


def _weekday_ranges(days: set[int]) -> list[str]:
    """Day names for APScheduler weekday indexes, consecutive days joined into a range."""
    if len(days) == len(_WEEKDAY_NAMES):
        return ["*"]
    ranges: list[str] = []
    for _, run in groupby(enumerate(sorted(days)), key=lambda pair: pair[1] - pair[0]):
        run_days = [day for _, day in run]
        first, last = _WEEKDAY_NAMES[run_days[0]], _WEEKDAY_NAMES[run_days[-1]]
        ranges.append(first if first == last else f"{first}-{last}")
    return ranges


def _month_numbers(field: str) -> str:
    """Spell a crontab ``month`` field with numbers in place of month names.

    An item with a letter must read in full, and APScheduler then accepts or
    rejects it as the same item written with numbers (``jan-dec/3`` as
    ``1-12/3``); an item with no letter passes as written. Raises
    ``ValueError`` naming the item as written.
    """
    from apscheduler.triggers.cron.expressions import RangeExpression

    items: list[str] = []
    for item in field.split(","):
        if not any(character.isalpha() for character in item):
            items.append(item)
            continue
        match = _MONTH_ITEM.fullmatch(item.lower())
        if match is None:
            raise ValueError(
                f"month item is not a month (1-12 or jan-dec), a range or a step: {item!r}"
            )
        first = _crontab_number(match["first"], _CRONTAB_MONTHS)
        last = None if match["last"] is None else _crontab_number(match["last"], _CRONTAB_MONTHS)
        step = match["step"]
        # APScheduler's own check of a numeric month item, run here so that its
        # error names this item rather than the numbers it became.
        try:
            RangeExpression(first, last, step).validate_range("month")
        except ValueError as exc:
            raise ValueError(f"month item {item!r}: {exc}") from exc
        numeric = str(first) if last is None else f"{first}-{last}"
        items.append(numeric if step is None else f"{numeric}/{step}")
    return ",".join(items)


def _reject_text_after_last(field: str) -> None:
    """Raise ``ValueError`` for a crontab ``day`` item that continues after ``last``.

    APScheduler reads such an item as ``last`` and ignores the rest.
    """
    for item in field.split(","):
        last = _LAST_DAY.match(item)
        if last is not None and last.end() < len(item):
            raise ValueError(f"day item takes nothing after last: {item!r}")


def build_cron_trigger(cron: str, timezone: str | None) -> Any:
    """Build an APScheduler ``CronTrigger`` for a five- or six-field crontab expression.

    Raises ``ValueError`` when the field count, a field value, or the timezone
    is invalid, so callers reject a schedule before it is stored.
    """
    from apscheduler.triggers.cron import CronTrigger

    parts = cron.split()
    if len(parts) == CRON_FIELD_COUNT:
        second: str | None = None
        minute, hour, day, month, day_of_week = parts
    elif len(parts) == CRON_FIELD_COUNT_WITH_SECONDS:
        second, minute, hour, day, month, day_of_week = parts
    else:
        raise ValueError(f"{CRON_FIELD_COUNT_ERROR}: {cron!r}")

    try:
        _reject_text_after_last(day)
        return CronTrigger(
            second=second,
            minute=minute,
            hour=hour,
            day=day,
            month=_month_numbers(month),
            day_of_week=day_of_week_names(day_of_week),
            timezone=timezone,
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"invalid cron expression or timezone: {exc}") from exc


_MIN_FIRE_GAP_SECONDS = 60 * 60
_FIRE_SAMPLES = 48


def _normalized_cron(cron: str) -> str:
    return " ".join(cron.split())


def _fires_more_than_hourly(cron: str, timezone: str | None) -> bool:
    """Whether two of the next fires are less than an hour apart."""
    trigger = build_cron_trigger(cron, timezone)
    previous = None
    cursor = datetime.now(UTC)
    for _ in range(_FIRE_SAMPLES):
        fire = trigger.get_next_fire_time(previous, cursor)
        if fire is None:
            return False
        if previous is not None and (fire - previous).total_seconds() < _MIN_FIRE_GAP_SECONDS:
            return True
        previous = fire
        cursor = fire
    return False


def _earliest_field_value(field: str, maximum: int) -> str:
    numbers = [int(match) for match in re.findall(r"\d+", field)]
    allowed = [number for number in numbers if number <= maximum]
    return str(min(allowed)) if allowed else "0"


def cap_cron_at_most_hourly(cron: str, timezone: str | None = None) -> str:
    """Return ``cron``, or a coarser expression that fires at most once an hour.

    A minute list or step (``8,23,38,53``, ``*/15``) keeps its earliest minute.
    A leading seconds field is dropped. Day, month, and weekday stay, so a
    daily schedule is not turned into an hourly one. An expression that
    already waits at least an hour is unchanged, including its whitespace
    collapsed to single spaces.
    """
    normalized = _normalized_cron(cron)
    if not _fires_more_than_hourly(normalized, timezone):
        return normalized
    parts = normalized.split()
    if len(parts) == CRON_FIELD_COUNT_WITH_SECONDS:
        parts = parts[1:]
    parts[0] = _earliest_field_value(parts[0], 59)
    capped = " ".join(parts)
    if _fires_more_than_hourly(capped, timezone):
        parts[1] = _earliest_field_value(parts[1], 23)
        capped = " ".join(parts)
    return capped


__all__ = [
    "CRON_FIELD_COUNT",
    "CRON_FIELD_COUNT_ERROR",
    "CRON_FIELD_COUNT_WITH_SECONDS",
    "CRON_FORMAT_HELP",
    "build_cron_trigger",
    "cap_cron_at_most_hourly",
    "day_of_week_names",
]
