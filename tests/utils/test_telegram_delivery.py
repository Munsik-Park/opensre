"""Tests for utils/telegram_delivery.py."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from integrations.telegram import delivery as telegram_delivery
from integrations.telegram.delivery import (
    _balance_telegram_markup_tags,
    _TelegramTokenFilter,
    post_telegram_message,
    send_telegram_report,
    truncate_for_telegram_html,
)
from integrations.telegram.tools.telegram_send_message_tool import (
    delivery as telegram_send_message_delivery,
)
from integrations.telegram.tools.telegram_send_message_tool.models import (
    TelegramDeliveryTarget,
)
from platform.notifications.delivery_transport import DeliveryResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_response(status_code: int, body: dict[str, Any]) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = body
    return resp


# ---------------------------------------------------------------------------
# post_telegram_message
# ---------------------------------------------------------------------------


def test_post_telegram_message_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "platform.notifications.delivery_transport.httpx.post",
        lambda *_a, **_kw: _mock_response(200, {"ok": True, "result": {"message_id": 42}}),
    )
    ok, error, message_id = post_telegram_message("chat-1", "hello", "bot-token")
    assert ok is True
    assert error == ""
    assert message_id == "42"


def test_post_telegram_message_sends_correct_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_post(url: str, *, json: dict[str, Any], **_kw: Any) -> MagicMock:
        captured["url"] = url
        captured["json"] = json
        return _mock_response(200, {"ok": True, "result": {"message_id": 1}})

    monkeypatch.setattr("platform.notifications.delivery_transport.httpx.post", _fake_post)
    post_telegram_message("chat-42", "test text", "my-token")

    assert "my-token" in captured["url"]
    assert "sendMessage" in captured["url"]
    assert captured["json"]["chat_id"] == "chat-42"
    assert captured["json"]["text"] == "test text"


def test_post_telegram_message_with_reply_to(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_post(url: str, *, json: dict[str, Any], **_kw: Any) -> MagicMock:
        captured["json"] = json
        return _mock_response(200, {"ok": True, "result": {"message_id": 2}})

    monkeypatch.setattr("platform.notifications.delivery_transport.httpx.post", _fake_post)
    post_telegram_message("chat-1", "text", "token", reply_to_message_id="99")
    assert captured["json"]["reply_to_message_id"] == 99


def test_post_telegram_message_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "platform.notifications.delivery_transport.httpx.post",
        lambda *_a, **_kw: _mock_response(
            400, {"ok": False, "description": "Bad Request: chat not found"}
        ),
    )
    ok, error, message_id = post_telegram_message("chat-1", "text", "bot-token")
    assert ok is False
    assert "Bad Request" in error
    assert message_id == ""


def test_post_telegram_message_exception_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_a: Any, **_kw: Any) -> None:
        raise ConnectionError("network down")

    monkeypatch.setattr("platform.notifications.delivery_transport.httpx.post", _raise)
    ok, error, message_id = post_telegram_message("chat-1", "text", "bot-token")
    assert ok is False
    assert "network down" in error
    assert message_id == ""


def test_post_telegram_message_exception_redacts_token(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "secret-bot-token-123"

    def _raise(*_a: Any, **_kw: Any) -> None:
        raise ConnectionError(f"failed to connect to api.telegram.org/bot{secret}/sendMessage")

    monkeypatch.setattr("platform.notifications.delivery_transport.httpx.post", _raise)
    ok, error, _ = post_telegram_message("chat-1", "text", secret)
    assert ok is False
    assert secret not in error
    assert "<redacted>" in error


# ---------------------------------------------------------------------------
# send_telegram_report
# ---------------------------------------------------------------------------


def test_send_telegram_report_posts_to_chat(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_post(url: str, *, json: dict[str, Any], **_kw: Any) -> MagicMock:
        captured["url"] = url
        captured["json"] = json
        return _mock_response(200, {"ok": True, "result": {"message_id": 5}})

    monkeypatch.setattr("platform.notifications.delivery_transport.httpx.post", _fake_post)
    ok, error = send_telegram_report("Report text", {"bot_token": "tok", "chat_id": "chat-1"})

    assert ok is True
    assert error == ""
    assert "tok" in captured["url"]
    assert captured["json"]["chat_id"] == "chat-1"
    assert captured["json"]["text"] == "Report text"
    assert captured["json"]["parse_mode"] == "HTML"


def test_send_telegram_report_uses_reply_to_message_id(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_post(url: str, *, json: dict[str, Any], **_kw: Any) -> MagicMock:
        captured["json"] = json
        return _mock_response(200, {"ok": True, "result": {"message_id": 6}})

    monkeypatch.setattr("platform.notifications.delivery_transport.httpx.post", _fake_post)
    send_telegram_report(
        "Report",
        {"bot_token": "tok", "chat_id": "chat-1", "reply_to_message_id": "77"},
    )
    assert captured["json"].get("reply_to_message_id") == 77


def test_send_telegram_report_passes_reply_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _fake_post(url: str, *, json: dict[str, Any], **_kw: Any) -> MagicMock:
        captured["json"] = json
        return _mock_response(200, {"ok": True, "result": {"message_id": 7}})

    monkeypatch.setattr("platform.notifications.delivery_transport.httpx.post", _fake_post)
    markup = {"inline_keyboard": [[{"text": "Open", "url": "https://x"}]]}
    ok, err = send_telegram_report(
        "Hi",
        {"bot_token": "tok", "chat_id": "c1"},
        reply_markup=markup,
    )
    assert ok is True
    assert err == ""
    assert captured["json"]["reply_markup"] == markup


def test_send_telegram_report_returns_false_on_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "platform.notifications.delivery_transport.httpx.post",
        lambda *_a, **_kw: _mock_response(403, {"ok": False, "description": "Forbidden"}),
    )
    ok, error = send_telegram_report("Report", {"bot_token": "tok", "chat_id": "chat-1"})
    assert ok is False
    assert "Forbidden" in error


# ---------------------------------------------------------------------------
# fix 1 – httpx log token filter
# ---------------------------------------------------------------------------


def test_telegram_token_filter_scrubs_url_from_msg() -> None:
    f = _TelegramTokenFilter()
    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg="HTTP Request: POST https://api.telegram.org/botSECRET123/sendMessage",
        args=(),
        exc_info=None,
    )
    f.filter(record)
    assert "SECRET123" not in record.msg
    assert "bot<redacted>/sendMessage" in record.msg


def test_telegram_token_filter_scrubs_token_in_args() -> None:
    f = _TelegramTokenFilter()
    record = logging.LogRecord(
        name="httpcore",
        level=logging.DEBUG,
        pathname="",
        lineno=0,
        msg="send %s",
        args=("https://api.telegram.org/botSECRET123/sendMessage",),
        exc_info=None,
    )
    f.filter(record)
    assert isinstance(record.args, tuple)
    assert "SECRET123" not in record.args[0]
    assert "bot<redacted>/sendMessage" in record.args[0]


def test_telegram_token_filter_scrubs_non_string_url_arg() -> None:
    """httpx passes request.url as an httpx.URL object, not a plain str."""

    class FakeURL:
        def __str__(self) -> str:
            return "https://api.telegram.org/botSECRET123/sendMessage"

    f = _TelegramTokenFilter()
    record = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg='HTTP Request: %s %s "%s %d %s"',
        args=("POST", FakeURL(), "HTTP/1.1", 200, "OK"),
        exc_info=None,
    )
    f.filter(record)
    assert isinstance(record.args, tuple)
    assert "SECRET123" not in str(record.args[1])
    assert "bot<redacted>/sendMessage" in str(record.args[1])


# ---------------------------------------------------------------------------
# fix 2 – send_telegram_report guards missing creds
# ---------------------------------------------------------------------------


def test_send_telegram_report_missing_bot_token() -> None:
    ok, error = send_telegram_report("report", {"bot_token": "", "chat_id": "chat-1"})
    assert ok is False
    assert "Missing" in error


def test_send_telegram_report_missing_chat_id() -> None:
    ok, error = send_telegram_report("report", {"bot_token": "tok", "chat_id": ""})
    assert ok is False
    assert "Missing" in error


def test_send_telegram_report_missing_both_creds() -> None:
    ok, error = send_telegram_report("report", {})
    assert ok is False
    assert "Missing" in error


# ---------------------------------------------------------------------------
# fix 3 – resp.json() only called after status check (non-JSON error body)
# ---------------------------------------------------------------------------


def test_post_telegram_message_non_json_error_body(monkeypatch: pytest.MonkeyPatch) -> None:
    resp = MagicMock()
    resp.status_code = 502
    resp.json.side_effect = ValueError("not JSON")
    resp.text = "Bad Gateway"

    monkeypatch.setattr(
        "platform.notifications.delivery_transport.httpx.post", lambda *_a, **_kw: resp
    )
    ok, error, message_id = post_telegram_message("chat-1", "text", "tok")

    assert ok is False
    assert "Bad Gateway" in error
    assert message_id == ""


# ---------------------------------------------------------------------------
# fix 4 – reply_to_message_id="0" must not be sent to the API
# ---------------------------------------------------------------------------


def test_post_telegram_message_reply_to_zero_string_not_sent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _fake_post(url: str, *, json: dict[str, Any], **_kw: Any) -> MagicMock:
        captured["json"] = json
        return _mock_response(200, {"ok": True, "result": {"message_id": 1}})

    monkeypatch.setattr("platform.notifications.delivery_transport.httpx.post", _fake_post)
    post_telegram_message("chat-1", "text", "tok", reply_to_message_id="0")
    assert "reply_to_message_id" not in captured["json"]


def test_send_telegram_report_truncates_to_4096(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    monkeypatch.setattr(
        "platform.notifications.delivery_transport.httpx.post",
        lambda *_a, **kw: (
            captured.update({"text": kw["json"].get("text", "")})
            or _mock_response(200, {"ok": True, "result": {"message_id": 7}})
        ),  # type: ignore[misc]
    )
    long_report = "x" * 5000
    send_telegram_report(long_report, {"bot_token": "tok", "chat_id": "chat-1"})
    assert len(captured["text"]) == 4096
    assert captured["text"].endswith("…")


def test_truncate_for_telegram_html_strips_incomplete_trailing_tag() -> None:
    # Narrow limit so slice lands after ``<b>`` opens but well before ``</b>``.
    src = "<b>" + ("m" * 80)
    out = truncate_for_telegram_html(src, 18, suffix="…")
    assert len(out) <= 18
    assert out.endswith("…")
    assert not out[:-1].replace("…", "").endswith("<")


def test_truncate_for_telegram_html_balances_open_tags_within_limit() -> None:
    long_inner = "z" * 6000
    src = f"<b>start {long_inner}</b>"
    out = truncate_for_telegram_html(src, 4096, suffix="…")
    assert len(out) == 4096
    assert out.endswith("…")
    assert out.count("<b>") == out.count("</b>")


def test_truncate_for_telegram_html_noop_when_under_limit() -> None:
    s = "<i>ok</i>"
    assert truncate_for_telegram_html(s, 50) == s


# ---------------------------------------------------------------------------
# Shared-transport delegation (regression coverage for the #864 refactor)
# ---------------------------------------------------------------------------


class TestDelegatesToSharedTransport:
    """After #864 the telegram helper uses ``delivery_transport.post_json``
    rather than calling httpx directly. Pins the contract so re-importing
    httpx into ``telegram_delivery`` regresses loudly."""

    def test_module_does_not_import_httpx(self) -> None:
        # Reuse the module-level ``from integrations.telegram_delivery import delivery as telegram_delivery`` (renamed for cleanliness)
        # to avoid importing the same module via both ``import`` and
        # ``from import`` styles (CodeQL py/import-and-import-from).
        assert not hasattr(telegram_delivery, "httpx"), (
            "telegram_delivery should not import httpx directly — "
            "it must go through delivery_transport.post_json"
        )

    def test_post_message_uses_post_json_helper(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from platform.notifications.delivery_transport import DeliveryResponse

        captured: dict[str, Any] = {}

        def _stub_post_json(url: str, payload: dict, **kw: Any) -> DeliveryResponse:
            captured["url"] = url
            captured["payload"] = payload
            return DeliveryResponse(
                ok=True, status_code=200, data={"ok": True, "result": {"message_id": 42}}
            )

        monkeypatch.setattr("integrations.telegram.delivery.post_json", _stub_post_json)
        ok, err, mid = post_telegram_message("chat-1", "hello", "secret-bot-tok")
        assert ok is True
        assert err == ""
        assert mid == "42"
        assert "/bot" in captured["url"]
        assert captured["payload"]["chat_id"] == "chat-1"
        assert captured["payload"]["text"] == "hello"

    def test_transport_failure_redacts_token_in_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the shared helper returns ``ok=False`` because the transport
        raised, telegram must still scrub the bot token out of the error
        string before propagating it."""
        from platform.notifications.delivery_transport import DeliveryResponse

        bot_token = "1234567890:ABCDEFverysecretvalue"
        leak_msg = f"connect failed for url=https://api.telegram.org/bot{bot_token}/sendMessage"

        monkeypatch.setattr(
            "integrations.telegram.delivery.post_json",
            lambda *_a, **_kw: DeliveryResponse(ok=False, error=leak_msg),
        )
        ok, err, _ = post_telegram_message("chat-1", "hi", bot_token)
        assert ok is False
        assert bot_token not in err
        assert "<redacted>" in err


# ---------------------------------------------------------------------------
# Issue #1 — unclosed <pre>/<s> at truncation cut, IndexError on bare <>
# Traces: .autoflow/issue-1-verification-design.md,
# .autoflow/issue-1-feature-design.md > Acceptance criteria
# ---------------------------------------------------------------------------


def test_balance_tags_closes_open_pre_at_cut() -> None:
    """AC-pre-block-closed."""
    out = _balance_telegram_markup_tags("<pre>some code")
    assert out.count("<pre>") == out.count("</pre>")
    assert out.endswith("</pre>")


def test_balance_tags_closes_open_strike_at_cut() -> None:
    """AC-strike-closed."""
    out = _balance_telegram_markup_tags("<s>struck")
    assert out.count("<s>") == out.count("</s>")
    assert out.endswith("</s>")


@pytest.mark.parametrize(
    "fragment",
    ["WHERE x <> 5", "WHERE x < > 5", "WHERE x <   > 5"],
)
def test_balance_tags_bare_angle_span_is_inert(fragment: str) -> None:
    """AC-bare-brackets-inert: bare-angle-pair-inert and whitespace-only-span-inert —
    a span whose interior yields no name token does not raise and passes through
    byte-identical."""
    assert _balance_telegram_markup_tags(fragment) == fragment


def test_balance_tags_real_bold_pair_survives_alongside_bare_brackets() -> None:
    """AC-real-tags-still-balanced: real-tag-survives-non-tag-neighbour — a genuine
    open tag in the same input as a bare <> is still balanced, and the bare span is
    left untouched."""
    out = _balance_telegram_markup_tags("<b>bold <> text")
    assert out == "<b>bold <> text</b>"


def test_balance_tags_nested_pre_code_close_innermost_first() -> None:
    """AC-real-tags-still-balanced: nested-close-order-preserved, pre/code."""
    assert _balance_telegram_markup_tags("<pre><code>x") == "<pre><code>x</code></pre>"


def test_balance_tags_nested_bold_italic_close_innermost_first() -> None:
    """AC-real-tags-still-balanced: nested-close-order-preserved, b/i (already-tracked
    names — pins the exact closer sequence, not just membership)."""
    assert _balance_telegram_markup_tags("<b>x<i>y") == "<b>x<i>y</i></b>"


@pytest.mark.parametrize("fragment", ["<pre>ok</pre> tail", "</pre> stray"])
def test_balance_tags_widened_close_branch_is_inert_on_already_balanced_pre(
    fragment: str,
) -> None:
    """AC-real-tags-still-balanced: widened-close-branch-inert — the push/pop path
    newly reached for a widened name (`pre`) is a no-op on already-balanced or
    stray-closer input."""
    assert _balance_telegram_markup_tags(fragment) == fragment


def test_balance_tags_anchor_still_closes_at_cut() -> None:
    """AC-real-tags-still-balanced: anchor-branch-untouched."""
    out = _balance_telegram_markup_tags('<a href="https://x">link')
    assert out == '<a href="https://x">link</a>'


def test_balance_tags_anchor_nested_in_open_pre_still_closes() -> None:
    """AC-real-tags-still-balanced: anchor-branch-untouched, nested inside <pre>."""
    out = _balance_telegram_markup_tags('<pre><a href="https://x">link')
    assert out == '<pre><a href="https://x">link</a></pre>'


def test_balance_tags_raw_span_already_tracked_name_unchanged() -> None:
    """AC-no-new-spurious-closer (a): no class change — an already-tracked name's
    multi-token raw span balances byte-identically before and after the fix."""
    assert _balance_telegram_markup_tags("attr <b foo> x") == "attr <b foo> x</b>"


def test_balance_tags_raw_span_widened_name_gains_closer() -> None:
    """AC-no-new-spurious-closer (b): membership change is the accepted outcome — a
    widened name's raw span gains the matching closer after the fix."""
    assert _balance_telegram_markup_tags("size <s m> ok") == "size <s m> ok</s>"


def test_truncate_tight_limit_forces_room_retreat_with_balanced_pre() -> None:
    """length-budget-holds: a limit tight enough to force the room-search retry loop
    to retreat still converges on balanced <pre> within the limit."""
    src = "<pre>" + ("x" * 50)
    out = truncate_for_telegram_html(src, 15, suffix="…")
    assert len(out) <= 15
    assert out.count("<pre>") == out.count("</pre>")


def test_truncate_at_production_limit_preserves_content_inside_pre() -> None:
    """length-budget-holds: at the production 4096 limit, the payload carries
    truncated content rather than collapsing to the suffix or content-free markup."""
    src = "<pre>" + ("code line\n" * 500) + "</pre>"
    out = truncate_for_telegram_html(src, 4096, suffix="…")
    assert len(out) <= 4096
    assert out.count("<pre>") == out.count("</pre>")
    content = out.replace("<pre>", "").replace("</pre>", "").rstrip("…")
    assert len(content.strip()) > 100


# ---------------------------------------------------------------------------
# Issue #1 — composition oracle: real renderer/passthrough into real truncator,
# driven through the production integrations/telegram/tools/telegram_send_message_tool
# wiring (dispatch_message -> markdown_to_telegram_html -> send_telegram_report ->
# truncate_for_telegram_html). Only the outbound HTTP transport is substituted.
# ---------------------------------------------------------------------------


def test_composition_renderer_fenced_block_truncated_mid_block_is_balanced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """renderer-to-truncator-composition: real renderer output, truncated mid-fenced
    block, comes back with balanced <pre>."""
    captured: dict[str, Any] = {}

    def _stub_post_json(url: str, payload: dict, **kw: Any) -> DeliveryResponse:
        captured["text"] = payload["text"]
        return DeliveryResponse(ok=True, status_code=200, data={"ok": True, "result": {}})

    monkeypatch.setattr("integrations.telegram.delivery.post_json", _stub_post_json)
    message = "Report\n\n```\n" + ("code line\n" * 600) + "```\n"
    target = TelegramDeliveryTarget(bot_token="tok", chat_id="chat-1")

    ok, err = telegram_send_message_delivery.dispatch_message(message, target)

    assert ok is True
    assert err == ""
    assert captured["text"].count("<pre>") == captured["text"].count("</pre>")


def test_composition_passthrough_bare_brackets_and_open_strike_survive_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """passthrough-html-composition: pre-rendered HTML admitted by the passthrough
    guard does not raise on a bare SQL <> predicate and closes an <s> left open at
    the cut."""
    captured: dict[str, Any] = {}

    def _stub_post_json(url: str, payload: dict, **kw: Any) -> DeliveryResponse:
        captured["text"] = payload["text"]
        return DeliveryResponse(ok=True, status_code=200, data={"ok": True, "result": {}})

    monkeypatch.setattr("integrations.telegram.delivery.post_json", _stub_post_json)
    message = "WHERE x <> 5 " + "<s>" + ("z" * 5000)
    target = TelegramDeliveryTarget(bot_token="tok", chat_id="chat-1")

    ok, err = telegram_send_message_delivery.dispatch_message(message, target)

    assert ok is True
    assert err == ""
    assert "<> 5" in captured["text"]
    # AC-strike-closed is a property of the balancer's output, not of the
    # truncator's tail byte: truncate_for_telegram_html appends the suffix
    # after the balanced chunk (delivery.py:88, pinned by
    # test_truncate_for_telegram_html_balances_open_tags_within_limit and
    # test_truncate_for_telegram_html_strips_incomplete_trailing_tag), so the
    # closer sits before the suffix, not at the very end of the string.
    assert captured["text"].rstrip("…").endswith("</s>")
