"""Serve ONLY laya-multilingual on loopback for the follow-up (isolated .laya-venv)."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import laya


class Handler(BaseHTTPRequestHandler):
    agent = None

    def do_GET(self):
        if self.path != "/health":
            self.send_error(404)
            return
        self.reply({"model": "convaiinnovations/laya-multilingual"})

    def do_POST(self):
        if self.path != "/v1/systemone":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size < 1_000_000:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(size))
            # Explicit model: no language heuristics that might route accentless PT to English.
            result = self.agent.predict(data["state"], data["questions"], max_len=1024)
            self.reply(result)
        except (ValueError, KeyError) as exc:
            self.reply({"error": str(exc)}, status=400)

    def reply(self, data, status=200):
        payload = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    Handler.agent = laya.load("convaiinnovations/laya-multilingual", device="cuda")
    print("Laya multilingual ready on 127.0.0.1:8018", flush=True)
    HTTPServer(("127.0.0.1", 8018), Handler).serve_forever()


if __name__ == "__main__":
    main()
