"""csdv_core.label.ledger: cost ledger, budget guard and response cache.

Every paid call appends one row to ``ledger.jsonl`` in the run folder. A call
is refused when its estimated cost would take the run past its cap, which the
caller must pass in: there is no default cap, so a run can never inherit one
by accident. Responses are cached on a hash of the request, so repeating a
finished run costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

#: List prices, US dollars per million tokens (input, output), read from the
#: provider price lists on 2026-11-11.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.7-flash": (0.75, 3.75),
    "openai/gpt-6.1-sol": (2.00, 10.00),
    "openai/gpt-6.1-terra": (1.00, 5.00),
}

__all__ = ["PRICES", "BudgetExceeded", "Ledger", "request_key", "token_cost"]


class BudgetExceeded(RuntimeError):
    """A call was refused because it would pass the cap."""


class Ledger:
    """Thread-safe ledger with a response cache beside it.

    Args:
        root: Run folder. Holds ``ledger.jsonl`` and ``responses/``.
        cap: Hard stop for everything recorded in this ledger, US dollars.
    """

    def __init__(self, root: Path, cap: float) -> None:
        self.path = Path(root) / "ledger.jsonl"
        self.cache_dir = Path(root) / "responses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cap = float(cap)
        self._lock = threading.Lock()
        self._pending: dict[str, float] = {}
        self.rows: list[dict[str, Any]] = []
        if self.path.exists():
            self.rows = [
                json.loads(line) for line in self.path.read_text().splitlines() if line
            ]

    def spent(self, stage: str | None = None) -> float:
        return sum(r["cost"] for r in self.rows if stage is None or r["stage"] == stage)

    def reserve(self, key: str, estimate: float) -> None:
        """Hold ``estimate`` against the cap, or refuse the call."""
        with self._lock:
            pending = sum(self._pending.values())
            total = self.spent() + pending + estimate
            if total > self.cap:
                raise BudgetExceeded(f"{total:.3f} would pass the cap {self.cap:.2f}")
            self._pending[key] = estimate

    def record(self, key: str, row: dict[str, Any]) -> None:
        with self._lock:
            self._pending.pop(key, None)
            row = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **row}
            self.rows.append(row)
            with self.path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")

    def release(self, key: str) -> None:
        with self._lock:
            self._pending.pop(key, None)

    def cache_path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def cached(self, key: str) -> dict[str, Any] | None:
        path = self.cache_path(key)
        return json.loads(path.read_text()) if path.exists() else None

    def store(self, key: str, payload: dict[str, Any]) -> None:
        path = self.cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))


def request_key(model: str, prompt: str, images: list[bytes], extra: str = "") -> str:
    """Hash of a request: model, prompt, settings and every image."""
    digest = hashlib.sha256()
    for part in (model.encode(), prompt.encode(), extra.encode(), *images):
        digest.update(hashlib.sha256(part).digest())
    return digest.hexdigest()


def token_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    price_in, price_out = PRICES[model]
    return (tokens_in * price_in + tokens_out * price_out) / 1e6
