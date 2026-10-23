"""Minimal fail-closed authorization and rate guard for Flask operator actions.

Keep operator secrets out of URLs, templates, cookies and logs. This module is
intentionally pure so auth paths can be regression tested without Flask.
"""
from __future__ import annotations

import hmac
import re
import time
from collections import deque


PROTECTED = {
    ("/api/generate", "GET"),
    ("/api/generate", "POST"),
    ("/api/group-code", "GET"),
    ("/api/build/code", "GET"),
    ("/api/bots/run", "POST"),
}


def needs_operator_auth(path: str, method: str, args: dict | None = None) -> bool:
    if (path, method.upper()) in PROTECTED:
        return True
    if path == "/api/build" and (args or {}).get("rebuild") in ("1", "true", "yes"):
        return True
    return False


def valid_operator_token(provided: str, configured: str | None) -> bool:
    if not configured or len(configured) < 32:
        return False
    if not provided:
        return False
    return hmac.compare_digest(provided, configured)


class WindowLimiter:
    """Best-effort per-process limiter. VPS reverse proxy must enforce globally."""
    def __init__(self, max_events: int = 8, window_sec: int = 60):
        self.max_events = max_events
        self.window_sec = window_sec
        self.hits: dict[str, deque[float]] = {}

    def allowed(self, key: str, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        events = self.hits.setdefault(key, deque())
        while events and events[0] <= now - self.window_sec:
            events.popleft()
        if len(events) >= self.max_events:
            return False
        events.append(now)
        # Limit memory from anonymous clients; reverse proxy is still required.
        if len(self.hits) > 2048:
            self.hits = {k: v for k, v in self.hits.items() if v}
        return True
