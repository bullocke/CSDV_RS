"""Ledger, cache and budget guard of the model client. No network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from csdv_core.label import clients
from csdv_core.label.ledger import BudgetExceeded, Ledger, request_key


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.status_code = 200
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


@pytest.fixture()
def fake_post(monkeypatch: pytest.MonkeyPatch) -> list:
    calls: list = []

    def post(url: str, **kwargs) -> _Response:
        calls.append(url)
        if "openrouter" in url:
            return _Response(
                {
                    "choices": [{"message": {"content": '{"states": ["forest"]}'}}],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 10,
                        "cost": 0.004,
                    },
                }
            )
        return _Response(
            {
                "candidates": [
                    {"content": {"parts": [{"text": '```json\n{"a": 1}\n```'}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 1000,
                    "candidatesTokenCount": 100,
                },
            }
        )

    import httpx

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("OpenRouter_API_KEY", "y")
    return calls


def test_call_is_recorded_and_cached(tmp_path: Path, fake_post: list) -> None:
    ledger = Ledger(tmp_path, cap=1.0)
    tag = {"cand_id": "c1"}
    reply = clients.ask(ledger, "terra", "prompt", [b"img"], stage="verdict", tag=tag)
    assert reply.data == {"states": ["forest"]} and reply.cost == 0.004
    assert not reply.cached and len(fake_post) == 1
    again = clients.ask(ledger, "terra", "prompt", [b"img"], stage="verdict", tag=tag)
    assert again.cached and again.data == reply.data and len(fake_post) == 1
    assert ledger.spent() == pytest.approx(0.004)
    reopened = Ledger(tmp_path, cap=1.0)
    assert reopened.spent("verdict") == pytest.approx(0.004)
    assert reopened.rows[0]["cand_id"] == "c1"


def test_gemini_cost_from_tokens_and_fenced_json(
    tmp_path: Path, fake_post: list
) -> None:
    ledger = Ledger(tmp_path, cap=1.0)
    reply = clients.ask(ledger, "gem37", "prompt", [b"img"], stage="verdict", tag={})
    assert reply.data == {"a": 1}
    assert reply.cost == pytest.approx((1000 * 0.75 + 100 * 3.75) / 1e6)


def test_cap_refuses_before_calling(tmp_path: Path, fake_post: list) -> None:
    ledger = Ledger(tmp_path, cap=0.001)
    with pytest.raises(BudgetExceeded):
        clients.ask(ledger, "terra", "prompt", [b"img"], stage="verdict", tag={})
    assert fake_post == [] and ledger.spent() == 0.0


def test_ask_many_stops_at_the_cap(tmp_path: Path, fake_post: list) -> None:
    one = clients.estimate_cost("terra", 1)
    ledger = Ledger(tmp_path, cap=0.004 + one * 1.2)
    jobs = [(f"prompt {i}", [b"img"], {"cand_id": str(i)}) for i in range(4)]
    replies = clients.ask_many(ledger, "terra", jobs, stage="verdict", workers=1)
    assert [r.error for r in replies] == ["", "", "budget", "budget"]
    assert len(fake_post) == 2 and ledger.spent() <= ledger.cap


def test_request_key_depends_on_every_part() -> None:
    key = request_key("m", "p", [b"a", b"b"])
    assert key != request_key("m", "p", [b"b", b"a"])
    assert key != request_key("m", "p2", [b"a", b"b"])
    assert key == request_key("m", "p", [b"a", b"b"])


def test_parse_json_tolerates_prose() -> None:
    assert clients.parse_json('Here: {"a": [1, 2]} done') == {"a": [1, 2]}
    assert clients.parse_json("nothing") is None
