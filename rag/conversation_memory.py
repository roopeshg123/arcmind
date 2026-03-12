"""
Server-side conversation memory for ArcMind.

Stores per-session conversation history in memory so clients do not need to
send the full history on every request (though they may still do so).

Design
------
- Keyed by session_id (a UUID string supplied by the client)
- Maximum MAX_TURNS turns retained per session (older turns evicted)
- Sessions expire after SESSION_TTL seconds of inactivity
- Thread-safe via a simple threading.Lock
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_TURNS    = 10      # maximum question/answer pairs per session
SESSION_TTL  = 3600    # seconds before an inactive session is evicted (1 hour)


# ---------------------------------------------------------------------------
# ConversationMemory
# ---------------------------------------------------------------------------

class ConversationMemory:
    """Thread-safe, TTL-bounded in-memory conversation store."""

    def __init__(
        self,
        max_turns: int = MAX_TURNS,
        ttl:       int = SESSION_TTL,
    ) -> None:
        self._max_turns  = max_turns
        self._ttl        = ttl
        self._sessions:    dict[str, list[dict]] = defaultdict(list)
        self._last_access: dict[str, float]      = {}
        self._lock = Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_history(self, session_id: str) -> list[dict]:
        """
        Return the conversation history for *session_id*.

        Returns an empty list if the session does not exist or has expired.
        """
        with self._lock:
            self._evict_expired()
            return list(self._sessions.get(session_id, []))

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """
        Append a Q/A turn to *session_id*'s history.

        Evicts the oldest turns when MAX_TURNS is exceeded.
        """
        with self._lock:
            history = self._sessions[session_id]
            history.append({"role": "user",      "content": question})
            history.append({"role": "assistant",  "content": answer})

            # Keep only the most recent max_turns turns (2 messages each)
            max_messages = self._max_turns * 2
            if len(history) > max_messages:
                self._sessions[session_id] = history[-max_messages:]

            self._last_access[session_id] = time.monotonic()

    def clear_session(self, session_id: str) -> None:
        """Delete all history for *session_id*."""
        with self._lock:
            self._sessions.pop(session_id, None)
            self._last_access.pop(session_id, None)

    def session_count(self) -> int:
        """Return the number of currently active sessions."""
        with self._lock:
            self._evict_expired()
            return len(self._sessions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_expired(self) -> None:
        """Remove sessions that have been idle longer than TTL.  Caller holds lock."""
        deadline = time.monotonic() - self._ttl
        expired  = [sid for sid, ts in self._last_access.items() if ts < deadline]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_access.pop(sid, None)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_memory = ConversationMemory()


def get_memory() -> ConversationMemory:
    """Return the process-wide ConversationMemory singleton."""
    return _memory
