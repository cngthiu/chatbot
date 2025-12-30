# =========================
# FILE: smart_food_bot/src/infrastructure/session_store.py
# (ADD: awaiting field to keep dialogue state)
# =========================
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class SessionState:
    session_id: str
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    # dialogue context
    last_user_text: str = ""
    last_intent: str = ""
    last_search_results: List[Dict[str, Any]] = field(default_factory=list)

    last_recipe_id: Optional[str] = None
    last_recipe_title: Optional[str] = None

    user_servings: Optional[int] = None
    constraints: Dict[str, Any] = field(default_factory=dict)
    cart: Dict[str, Any] = field(default_factory=dict)

    awaiting: Optional[str] = None  # e.g. "pick_recipe", "servings"


class InMemorySessionStore:
    def __init__(self, ttl_seconds: int = 1800) -> None:
        self.ttl_seconds = ttl_seconds
        self._data: Dict[str, SessionState] = {}

    def get_or_create(self, session_id: str) -> SessionState:
        self._gc()
        st = self._data.get(session_id)
        if st is None:
            st = SessionState(session_id=session_id)
            self._data[session_id] = st
        st.updated_at = time.time()
        return st

    def save(self, st: SessionState) -> None:
        st.updated_at = time.time()
        self._data[st.session_id] = st

    def _gc(self) -> None:
        now = time.time()
        expired = [k for k, v in self._data.items() if now - v.updated_at > self.ttl_seconds]
        for k in expired:
            self._data.pop(k, None)