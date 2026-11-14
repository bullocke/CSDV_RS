"""csdv_core.audit.server: serve the audit page and store the answers.

A small standard-library HTTP server for one interpreter on one machine. It
serves the static ``site`` folder and two routes: ``GET /state`` returns the
interpreter and every answer stored so far, and ``POST /answer`` appends one
answer as a line of ``answers/<interpreter>.jsonl``. The file is only ever
appended to. An undo is a further line that voids the item, so the last line
of an item is its answer.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

log = logging.getLogger(__name__)

MAX_BODY = 1 << 20

__all__ = ["latest_answers", "make_server", "read_answers"]


def read_answers(path: Path) -> list[dict]:
    """Every stored line of an answers file, oldest first."""
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def latest_answers(records: list[dict]) -> dict[str, dict]:
    """The answer in force per item: its last line, unless that voids it."""
    latest: dict[str, dict] = {}
    for record in records:
        if record.get("void"):
            latest.pop(record["item"], None)
        else:
            latest[record["item"]] = record
    return latest


class _Handler(SimpleHTTPRequestHandler):
    answers_path: Path
    interpreter: str
    lock: threading.Lock

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        log.debug("%s %s", self.address_string(), format % args)

    def end_headers(self) -> None:
        if not self.path.startswith("/img/"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.split("?")[0] == "/state":
            with self.lock:
                records = read_answers(self.answers_path)
            self._json({"interpreter": self.interpreter, "answers": records})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/answer":
            self._json({"error": "unknown route"}, HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", 0))
        if not 0 < length <= MAX_BODY:
            self._json({"error": "bad length"}, HTTPStatus.BAD_REQUEST)
            return
        try:
            record = json.loads(self.rfile.read(length))
            item = record["item"]
        except (ValueError, KeyError, TypeError):
            self._json({"error": "bad answer"}, HTTPStatus.BAD_REQUEST)
            return
        if not isinstance(item, str) or not item:
            self._json({"error": "bad item"}, HTTPStatus.BAD_REQUEST)
            return
        record["interpreter"] = self.interpreter
        record["saved"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.lock:
            self.answers_path.parent.mkdir(parents=True, exist_ok=True)
            with self.answers_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
                handle.flush()
        self._json({"ok": True, "saved": record["saved"]})


def make_server(
    site: Path,
    answers_path: Path,
    interpreter: str,
    host: str = "localhost",
    port: int = 8766,
) -> ThreadingHTTPServer:
    """A server for ``site`` that appends answers to ``answers_path``.

    Args:
        site: The built page folder.
        answers_path: The interpreter's answers file (JSON lines).
        interpreter: Name stored with every answer.
        host: Address to bind. Keep ``localhost`` unless the page must be
            reached from another machine.
        port: Port to bind. 0 picks a free one.

    Returns:
        The server, not yet serving. Call ``serve_forever``.
    """
    handler = type(
        "AuditHandler",
        (_Handler,),
        {
            "answers_path": Path(answers_path),
            "interpreter": interpreter,
            "lock": threading.Lock(),
        },
    )
    return ThreadingHTTPServer((host, port), partial(handler, directory=str(site)))
