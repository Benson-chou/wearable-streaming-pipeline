#!/usr/bin/env python3
"""
agent_stub.py — Minimal AI-agent stand-in for local testing.

Listens on :8000 and logs every POST body to stdout, so you can watch
spark_processor.py's anomaly alerts arrive end-to-end without a real agent.

    python agent_stub.py            # serves http://localhost:8000/anomaly
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.getenv("AGENT_PORT", "8000"))


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            pretty = json.dumps(json.loads(body), indent=2)
        except json.JSONDecodeError:
            pretty = body
        print(f"\n=== ANOMALY @ {self.path} ===\n{pretty}\n", flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"received"}')

    def log_message(self, *_args) -> None:  # silence default access logging
        pass


def main() -> None:
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"AI agent stub listening on http://localhost:{PORT}/  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
