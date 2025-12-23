# smart_food_bot/src/infrastructure/session_store.py
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

@dataclass
class SessionState:
    """
    WHY: Store minimal dialogue context to resolve references:
    - "món đó", "món vừa tìm", "cái số 2", "tăng khẩu phần", "thêm giỏ"
    """
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    last_user_text: str = ""
    last_intent: str = ""
    last_recipe_id: Optional[str] = None
    last_recipe_title: Optional[str] = None
    last_search_results: list[dict[str, Any]] = field(default_factory=list)

    user_servings: Optional[int] = None
    constraints: Dict[str, Any] = field(default_factory=dict)

    cart: Dict[str, Any] = field(default_factory=lambda: {"items": [], "pricing": {"subtotal": 0.0, "discount_total": 0.0, "total": 0.0}})

class InMemorySessionStore:
    """
    Fast TTL store; swapable to Redis later (DIP).
    """
    def __init__(self, ttl_seconds: int = 1800) -> None:
        self.ttl = ttl_seconds
        self._lock = threading.Lock()
        self._data: Dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        now = time.time()
        with self._lock:
            st = self._data.get(session_id)
            if st is None:
                st = SessionState(session_id=session_id)
                self._data[session_id] = st
            st.updated_at = now
            return st

    def save(self, st: SessionState) -> None:
        st.updated_at = time.time()
        with self._lock:
            self._data[st.session_id] = st

    def cleanup(self) -> None:
        now = time.time()
        with self._lock:
            expired = [sid for sid, st in self._data.items() if now - st.updated_at > self.ttl]
            for sid in expired:
                del self._data[sid]
