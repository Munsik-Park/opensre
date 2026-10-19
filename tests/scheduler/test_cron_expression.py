"""Cron lines are read the crontab way, although APScheduler 3.x counts weekdays from 0 = Monday."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

import pytest

from infrastructure.scheduling.scheduler.cron_expression import build_cron_trigger
from infrastructure.scheduling.scheduler.runner import compute_next_run
from infrastructure.scheduling.scheduler.types import Provider, ScheduledTask, TaskKind

_SUNDAY_NOON = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
_NEW_YEAR_2027 = datetime(2027, 1, 1, tzinfo=UTC)
_EVERY_DAY = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}


def _fire_times(cron: str, start: datetime, end: datetime) -> list[datetime]:
    """Every time ``cron`` fires from ``start`` up to, not including, ``end``."""
    trigger = build_cron_trigger(cron, "UTC")
    fires: list[datetime] = []
    fire = trigger.get_next_fire_time(None, start)
    while fire is not None and fire < end:
        fires.append(fire)
        fire = trigger.get_next_fire_time(fire, fire)
    return fires


def _fire_days(cron: str) -> set[str]:
    """The weekdays ``cron`` fires on during the two weeks after a Sunday noon."""
    two_weeks_later = _SUNDAY_NOON + timedelta(days=14)
    return {f"{fire:%a}" for fire in _fire_times(cron, _SUNDAY_NOON, two_weeks_later)}


def _fire_months(cron: str) -> set[str]:
    """The months ``cron`` fires in during 2027."""
    new_year_2028 = datetime(2028, 1, 1, tzinfo=UTC)
    return {f"{fire:%b}" for fire in _fire_times(cron, _NEW_YEAR_2027, new_year_2028)}


def _fire_dates_in_first_quarter(cron: str) -> list[str]:
    """The dates ``cron`` fires on from January to March 2027, in order."""
    april = datetime(2027, 4, 1, tzinfo=UTC)
    return [f"{fire:%b %d}" for fire in _fire_times(cron, _NEW_YEAR_2027, april)]


@pytest.mark.parametrize(
    ("day_of_week", "fires_on"),
    [
        ("1-5", {"Mon", "Tue", "Wed", "Thu", "Fri"}),
        ("0", {"Sun"}),
        ("7", {"Sun"}),
        ("*/2", {"Sun", "Tue", "Thu", "Sat"}),
        # A step after a name range is ignored by APScheduler, so steps become lists.
        ("1-5/2", {"Mon", "Wed", "Fri"}),
        ("mon-fri", {"Mon", "Tue", "Wed", "Thu", "Fri"}),
    ],
)
def test_a_weekday_field_fires_on_its_crontab_days(day_of_week: str, fires_on: set[str]) -> None:
    assert _fire_days(f"0 10 * * {day_of_week}") == fires_on


def test_a_weekday_number_above_7_is_rejected_before_it_is_stored() -> None:
    with pytest.raises(ValueError, match="0-7"):
        build_cron_trigger("0 10 * * 8", "UTC")


def test_the_stored_next_run_skips_the_weekend() -> None:
    """``next_run`` is what the store keeps and the loop list shows; it must agree with the trigger."""
    # Arrange: a weekday loop, asked after Friday's slot has passed
    task = ScheduledTask(
        kind=TaskKind.MANUAL_LOOP,
        cron="0 8 * * 1-5",
        timezone="UTC",
        provider=Provider.INTERACTIVE_SHELL,
    )
    friday_after_slot = datetime(2026, 10, 9, 9, 0, tzinfo=UTC)

    # Act
    next_run = compute_next_run(task, friday_after_slot)

    # Assert: Monday, not Saturday
    assert next_run == "2026-10-12T08:00:00+00:00"


@pytest.mark.parametrize(
    ("day_of_week", "numeric_twin", "fires_on"),
    [
        ("mon-fri/2", "1-5/2", {"Mon", "Wed", "Fri"}),
        ("mon/2", "1/2", {"Mon", "Wed", "Fri", "Sun"}),
        ("MON-FRI/2", "1-5/2", {"Mon", "Wed", "Fri"}),
        ("Mon/2", "1/2", {"Mon", "Wed", "Fri", "Sun"}),
        ("mon-fri/9", "1-5/9", {"Mon"}),
        ("mon-fri/2,sat", "1-5/2,6", {"Mon", "Wed", "Fri", "Sat"}),
        ("mon-5", "1-5", {"Mon", "Tue", "Wed", "Thu", "Fri"}),
        ("sun-sat/2", "0-6/2", {"Sun", "Tue", "Thu", "Sat"}),
        ("sun/2", "0/2", {"Sun", "Tue", "Thu", "Sat"}),
        ("sun-thu", "0-4", {"Sun", "Mon", "Tue", "Wed", "Thu"}),
        ("sun-sat", "0-6", _EVERY_DAY),
        ("sun-3", "0-3", {"Sun", "Mon", "Tue", "Wed"}),
        ("5-sun", "5-7", {"Fri", "Sat", "Sun"}),
        ("0-sun", "0-0", {"Sun"}),
        ("7-sun", "7-7", {"Sun"}),
    ],
)
def test_a_named_weekday_item_fires_on_its_numeric_twins_days(
    day_of_week: str, numeric_twin: str, fires_on: set[str]
) -> None:
    assert _fire_days(f"0 10 * * {day_of_week}") == fires_on
    assert _fire_days(f"0 10 * * {numeric_twin}") == fires_on


@pytest.mark.parametrize(("cron", "numeric_twin"), [("0 0 10 * * mon-fri/2", "0 0 10 * * 1-5/2")])
def test_a_six_field_line_reads_a_named_weekday_step_like_its_numeric_twin(
    cron: str, numeric_twin: str
) -> None:
    assert _fire_days(cron) == _fire_days(numeric_twin) == {"Mon", "Wed", "Fri"}


@pytest.mark.parametrize(
    ("day_of_week", "numeric_twin"),
    [
        ("mon-fri/0", "1-5/0"),
        ("mon/0", "1/0"),
        ("mon-fri/x", "1-5/x"),
        ("mon-fri@@", "1-5@@"),
        ("mon-8", "1-8"),
    ],
)
def test_a_named_weekday_item_is_rejected_where_its_numeric_twin_is(
    day_of_week: str, numeric_twin: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(day_of_week)):
        build_cron_trigger(f"0 10 * * {day_of_week}", "UTC")
    with pytest.raises(ValueError):
        build_cron_trigger(f"0 10 * * {numeric_twin}", "UTC")


@pytest.mark.parametrize(
    ("month", "numeric_twin", "fires_in"),
    [
        ("jan-dec/3", "1-12/3", {"Jan", "Apr", "Jul", "Oct"}),
        ("jan/3", "1/3", {"Jan", "Apr", "Jul", "Oct"}),
        ("feb-dec/5", "2-12/5", {"Feb", "Jul", "Dec"}),
        ("JAN-DEC/3", "1-12/3", {"Jan", "Apr", "Jul", "Oct"}),
        ("jan-dec/3,feb", "1-12/3,2", {"Jan", "Feb", "Apr", "Jul", "Oct"}),
        ("jan-6", "1-6", {"Jan", "Feb", "Mar", "Apr", "May", "Jun"}),
    ],
)
def test_a_named_month_item_fires_in_its_numeric_twins_months(
    month: str, numeric_twin: str, fires_in: set[str]
) -> None:
    assert _fire_months(f"0 9 1 {month} *") == fires_in
    assert _fire_months(f"0 9 1 {numeric_twin} *") == fires_in


@pytest.mark.parametrize(
    ("month", "numeric_twin"),
    [
        ("jan-dec/12", "1-12/12"),
        ("jan-dec/13", "1-12/13"),
        ("dec/3", "12/3"),
        ("jan/12", "1/12"),
        ("jan-dec/0", "1-12/0"),
        ("jan/0", "1/0"),
        ("jan-dec/x", "1-12/x"),
        ("jan-13", "1-13"),
    ],
)
def test_a_named_month_item_is_rejected_where_its_numeric_twin_is(
    month: str, numeric_twin: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(month)):
        build_cron_trigger(f"0 9 1 {month} *", "UTC")
    with pytest.raises(ValueError):
        build_cron_trigger(f"0 9 1 {numeric_twin} *", "UTC")


@pytest.mark.parametrize("cron", ["0 10 * * mon1", "0 9 1 jan1 *"])
def test_a_name_with_digits_joined_to_it_is_rejected(cron: str) -> None:
    with pytest.raises(ValueError):
        build_cron_trigger(cron, "UTC")


@pytest.mark.parametrize(
    "day", ["last/2", "last-5", "lastx", "last/0", "LAST/2", "1,last/2", "last/2,15"]
)
def test_a_day_of_month_item_continuing_after_last_is_rejected(day: str) -> None:
    with pytest.raises(ValueError):
        build_cron_trigger(f"0 9 {day} * *", "UTC")


@pytest.mark.parametrize(
    ("day_of_week", "fires_on"),
    [
        ("fri-sun", {"Fri", "Sat", "Sun"}),
        ("sat-sun", {"Sat", "Sun"}),
        ("mon-sun", _EVERY_DAY),
        ("sun-sun", {"Sun"}),
    ],
)
def test_a_named_weekday_range_ending_on_sunday_keeps_its_days(
    day_of_week: str, fires_on: set[str]
) -> None:
    assert _fire_days(f"0 10 * * {day_of_week}") == fires_on


@pytest.mark.parametrize(
    "cron",
    [
        "0 10 * * fri-mon",
        "0 10 * * 5-1",
        "0 10 * * 5-0",
        "0 10 * * monday",
        "0 9 1 dec-jan *",
        "0 9 1 12-1 *",
        "0 9 1 january *",
    ],
)
def test_a_backwards_range_or_a_full_day_or_month_name_stays_rejected(cron: str) -> None:
    with pytest.raises(ValueError):
        build_cron_trigger(cron, "UTC")


@pytest.mark.parametrize(
    ("day", "fires_on"),
    [
        ("last", ["Jan 31", "Feb 28", "Mar 31"]),
        ("LAST", ["Jan 31", "Feb 28", "Mar 31"]),
        ("1,last", ["Jan 01", "Jan 31", "Feb 01", "Feb 28", "Mar 01", "Mar 31"]),
    ],
)
def test_a_bare_last_keeps_firing_on_the_last_day_of_each_month(
    day: str, fires_on: list[str]
) -> None:
    assert _fire_dates_in_first_quarter(f"0 9 {day} * *") == fires_on
