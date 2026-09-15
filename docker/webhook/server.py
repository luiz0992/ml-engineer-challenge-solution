"""Local alert delivery sink.

Alertmanager refuses to load a config that references an unset webhook URL, so
the committed routing tree has no endpoints. This process is the missing
delivery half for local and CI: it accepts Alertmanager's webhook payload and
writes it to stdout and a JSONL file, which is enough to prove routing works
without committing a Slack or PagerDuty credential.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LISTEN_HOST = os.environ.get("WEBHOOK_HOST", "0.0.0.0")
LISTEN_PORT = int(os.environ.get("WEBHOOK_PORT", "8089"))
LOG_PATH = Path(os.environ.get("WEBHOOK_LOG", "/var/tmp/alerts.jsonl"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args: object) -> None:
        return

    def do_GET(self) -> None:
        if self.path in {"/health", "/-/healthy"}:
            self._respond(200, b"ok")
            return
        self._respond(404, b"not found")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self._respond(400, b"invalid json")
            return
        record = {
            "received_at": datetime.now(UTC).isoformat(),
            "path": self.path,
            "status": payload.get("status"),
            "receiver": payload.get("receiver"),
            "alerts": [
                {
                    "status": alert.get("status"),
                    "alertname": (alert.get("labels") or {}).get("alertname"),
                    "severity": (alert.get("labels") or {}).get("severity"),
                    "summary": (alert.get("annotations") or {}).get("summary"),
                }
                for alert in payload.get("alerts", [])
            ],
        }
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        self._respond(200, b'{"status":"ok"}')

    def _respond(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer((LISTEN_HOST, LISTEN_PORT), Handler)
    print(f"alert webhook listening on {LISTEN_HOST}:{LISTEN_PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
