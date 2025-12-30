# =========================
# FILE: smart_food_bot/src/application/rule_engine.py
# (ADD: standalone number => servings when awaiting servings)
# =========================
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.infrastructure.session_store import SessionState

_RE_PICK_NUMBER = re.compile(r"(?:chọn|lấy|mua|đặt)\s*(?:món\s*)?(?:số|#)?\s*(\d+)", re.IGNORECASE)
_RE_SERVINGS = re.compile(r"(?:cho\s*)?(?:tăng|giảm|đổi|set)\s*(\d+)\s*(?:khẩu\s*phần|phần|người)", re.IGNORECASE)
_RE_PLAN_CART = re.compile(r"(?:lên\s*giỏ|tạo\s*giỏ|mua\s*nguyên\s*liệu|thêm\s*vào\s*giỏ)\b", re.IGNORECASE)
_RE_SHOW_CART = re.compile(r"(?:xem\s*giỏ|giỏ\s*hàng|cart)\b", re.IGNORECASE)
_RE_EXCLUDE = re.compile(r"(?:không|bỏ)\s+([a-zA-ZÀ-ỹ\s]+)$", re.IGNORECASE)

_RE_GREET = re.compile(r"\b(xin\s*chào|chào|hello|hi|hey)\b", re.IGNORECASE)
_RE_THANKS = re.compile(r"\b(cảm\s*ơn|cam\s*on|thanks|thx)\b", re.IGNORECASE)
_RE_BYE = re.compile(r"\b(tạm\s*biệt|bye|bai|chào\s*nhé|hẹn\s*gặp\s*lại)\b", re.IGNORECASE)
_RE_SORRY = re.compile(r"\b(xin\s*lỗi|sorry)\b", re.IGNORECASE)

_RE_STANDALONE_INT = re.compile(r"^\s*(\d{1,2})\s*(?:phần|khẩu\s*phần|người)?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class RuleResult:
    action: str
    payload: Dict[str, Any]


class RuleEngine:
    def try_match(self, text: str, state: SessionState) -> Optional[RuleResult]:
        t = (text or "").strip()
        if not t:
            return None

        # Social first
        if _RE_GREET.search(t):
            return RuleResult("greet", {})
        if _RE_THANKS.search(t):
            return RuleResult("thanks", {})
        if _RE_BYE.search(t):
            return RuleResult("bye", {})
        if _RE_SORRY.search(t):
            return RuleResult("apology", {})

        # ✅ If bot is awaiting servings, treat a bare number as servings
        if state.awaiting == "servings":
            m = _RE_STANDALONE_INT.match(t)
            if m:
                return RuleResult("set_servings", {"user_servings": int(m.group(1))})

        # Commands
        m = _RE_PICK_NUMBER.search(t)
        if m:
            return RuleResult("pick_recipe", {"index_1based": int(m.group(1))})

        m = _RE_SERVINGS.search(t)
        if m:
            return RuleResult("set_servings", {"user_servings": int(m.group(1))})

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