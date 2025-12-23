# smart_food_bot/src/application/rule_engine.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.infrastructure.session_store import SessionState

_RE_PICK_NUMBER = re.compile(r"(?:chọn|lấy|mua|đặt)\s*(?:món\s*)?(?:số|#)?\s*(\d+)", re.IGNORECASE)
_RE_SERVINGS = re.compile(r"(?:cho\s*)?(?:tăng|giảm|đổi|set)\s*(\d+)\s*(?:khẩu\s*phần|phần|người)", re.IGNORECASE)
_RE_PLAN_CART = re.compile(r"(?:lên\s*giỏ|tạo\s*giỏ|mua\s*nguyên\s*liệu|thêm\s*vào\s*giỏ)\b", re.IGNORECASE)
_RE_SHOW_CART = re.compile(r"(?:xem\s*giỏ|giỏ\s*hàng|cart)\b", re.IGNORECASE)
_RE_EXCLUDE = re.compile(r"(?:không|bỏ)\s+([a-zA-ZÀ-ỹ\s]+)$", re.IGNORECASE)

@dataclass(frozen=True)
class RuleResult:
    """
    action: one of {"pick_recipe", "set_servings", "plan_cart", "show_cart", "add_exclude"}
    payload: normalized info for DialogueManager
    """
    action: str
    payload: Dict[str, Any]

class RuleEngine:
    """
    WHY: Instant handling for high-frequency commands; avoids GPU/CPU inference latency.
    """
    def try_match(self, text: str, state: SessionState) -> Optional[RuleResult]:
        t = (text or "").strip()
        if not t:
            return None

        m = _RE_PICK_NUMBER.search(t)
        if m:
            idx = int(m.group(1))
            return RuleResult("pick_recipe", {"index_1based": idx})

        m = _RE_SERVINGS.search(t)
        if m:
            servings = int(m.group(1))
            return RuleResult("set_servings", {"user_servings": servings})

        if _RE_PLAN_CART.search(t):
            return RuleResult("plan_cart", {})

        if _RE_SHOW_CART.search(t):
            return RuleResult("show_cart", {})

        m = _RE_EXCLUDE.search(t)
        if m:
            ingredient = m.group(1).strip()
            if ingredient:
                return RuleResult("add_exclude", {"ingredient": ingredient})
        return None
