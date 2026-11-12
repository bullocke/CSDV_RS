"""csdv_core.label.clients: calls to the vision models, with cost tracking.

Gemini is called directly. GPT models go through OpenRouter, which reports
the billed cost of each call. Every call passes the response cache and the
budget guard of a :class:`~csdv_core.label.ledger.Ledger`.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from csdv_core.label.env import api_key
from csdv_core.label.ledger import (
    PRICES,
    BudgetExceeded,
    Ledger,
    request_key,
    token_cost,
)

logger = logging.getLogger(__name__)

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

#: HTTP statuses that say "too many requests" or "overloaded". A call that
#: meets one waits and is sent again without using up its attempts, because
#: the fault is the pace of the run and not the request.
BUSY_STATUS = (429, 503, 529)
#: Waits on a busy status before the call is given up.
BUSY_TRIES = 6
#: First wait on a busy status, seconds. It doubles up to BUSY_MAX_WAIT_S.
BUSY_WAIT_S = 15.0
BUSY_MAX_WAIT_S = 120.0

#: Short names used in tables, and the provider that serves each model.
MODELS: dict[str, tuple[str, str]] = {
    "gem37": ("gemini", "gemini-3.7-flash"),
    "terra": ("openrouter", "openai/gpt-6.1-terra"),
    "sol": ("openrouter", "openai/gpt-6.1-sol"),
}

__all__ = ["MODELS", "Reply", "ask", "ask_many", "estimate_cost", "parse_json"]


@dataclass
class Reply:
    """One model answer and what it cost."""

    model: str
    text: str
    data: Any  # parsed JSON, or None when the text could not be parsed
    cost: float
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    cached: bool = False
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def parse_json(text: str) -> Any:
    """Parse a JSON answer, tolerating code fences and surrounding prose."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min((i for i in (text.find("{"), text.find("[")) if i >= 0), default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    return None


def estimate_cost(
    name: str,
    n_images: int,
    tokens_per_image: float = 1100.0,
    out_tokens: float = 1500.0,
) -> float:
    """Upper estimate of one call, held against the cap until the bill is known."""
    price_in, price_out = PRICES[MODELS[name][1]]
    return (
        (n_images * tokens_per_image + 900) * price_in + out_tokens * price_out
    ) / 1e6


def _gemini(
    model: str, prompt: str, images: list[bytes], max_out: int, effort: str
) -> tuple[str, int, int, float | None, dict]:
    import httpx

    parts: list[dict[str, Any]] = [{"text": prompt}]
    for img in images:
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img).decode(),
                }
            }
        )
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "maxOutputTokens": max_out,
            "temperature": 0,
            "responseMimeType": "application/json",
            "thinkingConfig": {"thinkingLevel": effort},
        },
    }
    resp = httpx.post(
        GEMINI_URL.format(model),
        json=body,
        timeout=180,
        headers={"x-goog-api-key": api_key("GEMINI_API_KEY")},
    )
    resp.raise_for_status()
    out = resp.json()
    cand = (out.get("candidates") or [{}])[0]
    text = "".join(
        p.get("text", "")
        for p in cand.get("content", {}).get("parts", [])
        if not p.get("thought")
    )
    usage = out.get("usageMetadata", {})
    tokens_out = usage.get("candidatesTokenCount", 0) + usage.get(
        "thoughtsTokenCount", 0
    )
    return text, usage.get("promptTokenCount", 0), tokens_out, None, out


def _openrouter(
    model: str, prompt: str, images: list[bytes], max_out: int, effort: str
) -> tuple[str, int, int, float | None, dict]:
    import httpx

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for img in images:
        url = "data:image/jpeg;base64," + base64.b64encode(img).decode()
        content.append({"type": "image_url", "image_url": {"url": url}})
    body = {
        "model": model,
        "max_tokens": max_out,
        "temperature": 0,
        "reasoning": {"effort": effort},
        "usage": {"include": True},
        "messages": [{"role": "user", "content": content}],
    }
    resp = httpx.post(
        OPENROUTER_URL,
        json=body,
        timeout=240,
        headers={"Authorization": "Bearer " + api_key("OpenRouter_API_KEY")},
    )
    resp.raise_for_status()
    out = resp.json()
    if "choices" not in out:
        raise RuntimeError(str(out)[:300])
    text = out["choices"][0]["message"].get("content") or ""
    usage = out.get("usage", {})
    return (
        text,
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        usage.get("cost"),
        out,
    )


def _busy_wait(response: Any, n: int) -> float:
    """Seconds to wait after the ``n``-th busy answer.

    The server's ``Retry-After`` (seconds) is used when it gives one.
    """
    try:
        asked = float(response.headers.get("retry-after", ""))
    except (AttributeError, TypeError, ValueError):
        asked = 0.0
    if asked > 0:
        return min(asked, BUSY_MAX_WAIT_S)
    return min(BUSY_WAIT_S * 2 ** (n - 1), BUSY_MAX_WAIT_S)


def ask(
    ledger: Ledger,
    name: str,
    prompt: str,
    images: list[bytes],
    *,
    stage: str,
    tag: dict[str, Any],
    max_out: int = 4000,
    effort: str = "low",
    attempts: int = 3,
    repeat: int = 0,
) -> Reply:
    """Send one request, through the cache and the budget guard.

    Args:
        ledger: Ledger of the run.
        name: Short model name, a key of :data:`MODELS`.
        prompt: Prompt text.
        images: JPEG bytes, in the order the prompt describes.
        stage: Label of the step, copied into the ledger row.
        tag: Further fields copied into the ledger row (candidate, window).
        repeat: Number of the repeat of an identical request. A repeat has
            its own cache entry, so it is sent again.

    Raises:
        BudgetExceeded: If the call would pass the ledger's cap.
    """
    import httpx

    provider, model = MODELS[name]
    key = request_key(model, prompt, images, f"{max_out}|{effort}|{repeat}")
    if repeat:
        tag = {**tag, "repeat": repeat}
    hit = ledger.cached(key)
    if hit is not None:
        return Reply(
            model=name,
            text=hit["text"],
            data=parse_json(hit["text"]),
            cost=hit["cost"],
            tokens_in=hit["tokens_in"],
            tokens_out=hit["tokens_out"],
            seconds=hit["seconds"],
            cached=True,
        )
    ledger.reserve(key, estimate_cost(name, len(images)))
    call = _gemini if provider == "gemini" else _openrouter
    last = ""
    attempt, busy = 0, 0
    while attempt < attempts:
        start = time.time()
        try:
            text, t_in, t_out, cost, raw = call(model, prompt, images, max_out, effort)
        except Exception as exc:  # noqa: BLE001
            last = repr(exc)[:300]
            status = None
            if isinstance(exc, httpx.HTTPStatusError):
                last += " " + exc.response.text[:300]
                status = exc.response.status_code
            if status in (400, 401, 403, 404):
                break
            if status in BUSY_STATUS and busy < BUSY_TRIES:
                busy += 1
                wait = _busy_wait(exc.response, busy)
                logger.warning(
                    "%s is busy (HTTP %d), waiting %.0f s (%d of %d)",
                    name,
                    status,
                    wait,
                    busy,
                    BUSY_TRIES,
                )
                time.sleep(wait)
                continue
            attempt += 1
            logger.warning("%s attempt %d failed: %s", name, attempt, last)
            time.sleep(4 * attempt)
            continue
        seconds = time.time() - start
        if cost is None:
            cost = token_cost(model, t_in, t_out)
        ledger.record(
            key,
            {
                "stage": stage,
                "model": name,
                **tag,
                "tokens_in": t_in,
                "tokens_out": t_out,
                "cost": cost,
                "seconds": round(seconds, 1),
            },
        )
        data = parse_json(text)
        if data is not None:
            ledger.store(
                key,
                {
                    "text": text,
                    "cost": cost,
                    "tokens_in": t_in,
                    "tokens_out": t_out,
                    "seconds": seconds,
                    "model": name,
                    **tag,
                },
            )
        return Reply(
            model=name,
            text=text,
            data=data,
            cost=cost,
            tokens_in=t_in,
            tokens_out=t_out,
            seconds=seconds,
            raw=raw,
            error="" if data is not None else "unparseable",
        )
    ledger.release(key)
    return Reply(model=name, text="", data=None, cost=0.0, error=last or "failed")


def ask_many(
    ledger: Ledger,
    name: str,
    jobs: list[tuple[str, list[bytes], dict[str, Any]]],
    *,
    stage: str,
    workers: int = 6,
    repeat: int = 0,
) -> list[Reply]:
    """Send ``(prompt, images, tag)`` jobs concurrently, in order.

    Sending stops at the first refused call. The jobs after it come back with
    ``error="budget"`` and cost nothing.
    """
    stopped = {"flag": False}

    def send(job: tuple[str, list[bytes], dict[str, Any]]) -> Reply:
        prompt, images, tag = job
        if stopped["flag"]:
            return Reply(model=name, text="", data=None, cost=0.0, error="budget")
        try:
            return ask(
                ledger, name, prompt, images, stage=stage, tag=tag, repeat=repeat
            )
        except BudgetExceeded as exc:
            stopped["flag"] = True
            logger.error("Budget stop: %s", exc)
            return Reply(model=name, text="", data=None, cost=0.0, error="budget")

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(send, jobs))
