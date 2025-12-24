# smart_food_bot/src/application/dialogue_manager.py
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

import anyio

from src.infrastructure.session_store import InMemorySessionStore, SessionState
from src.application.rule_engine import RuleEngine, RuleResult
from src.services.nlu_engine import NLUEngine
from src.application.usecases import SearchRecipes
from src.application.cart_planner import CartPlanner

log = logging.getLogger("app.dialogue_manager")


class DialogueManager:
    """
    Smooth multi-turn orchestrator:
    - Rule-first (instant)
    - Concurrent NLU + Search (no waiting)
    - Context retention via SessionStore
    """

    def __init__(
        self,
        sessions: InMemorySessionStore,
        rule_engine: RuleEngine,
        nlu: NLUEngine,
        search_uc: SearchRecipes,
        cart_planner: CartPlanner,
    ) -> None:
        self.sessions = sessions
        self.rules = rule_engine
        self.nlu = nlu
        self.search_uc = search_uc
        self.cart_planner = cart_planner

    async def handle(self, session_id: str, text: str) -> Dict[str, Any]:
        st = self.sessions.get_or_create(session_id)
        st.last_user_text = text

        # 1) Rule-first: fast path for frequent patterns
        rule = self.rules.try_match(text, st)
        if rule:
            out = await self._handle_rule(rule, st)
            self.sessions.save(st)
            return out

        # 2) Run NLU + Search concurrently
        # anyio doesn't provide gather; use asyncio.gather
        nlu_coro = anyio.to_thread.run_sync(self.nlu.predict, text)
        search_coro = anyio.to_thread.run_sync(self.search_uc, text, 5)
        nlu_out, raw_results = await asyncio.gather(nlu_coro, search_coro)

        st.last_intent = nlu_out.get("intent", "")
        slots = nlu_out.get("slots", {}) or {}

        # 3) Route by intent
        if st.last_intent in ("search_recipe", "refine_search", "ask_recipe_detail"):
            refined_query = self._build_refined_query(text, slots)

            refined_results = await anyio.to_thread.run_sync(self.search_uc, refined_query, 5)
            st.last_search_results = refined_results

            out = {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": "Mình tìm được vài món phù hợp. Bạn chọn món số mấy để mình lên giỏ nguyên liệu?",
                "recipes": refined_results,
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out

        if st.last_intent in ("add_ingredients_to_cart", "ask_price_estimate"):
            # Prefer current selected recipe; otherwise fallback to last search result
            self._ensure_recipe_selected(st)

            if not st.last_recipe_id:
                st.last_search_results = raw_results
                out = {
                    "session_id": session_id,
                    "nlu": nlu_out,
                    "reply": "Bạn muốn mua nguyên liệu cho món nào? Hãy chọn 1 món trong danh sách nhé (ví dụ: 'chọn món số 2').",
                    "recipes": raw_results,
                    "context": self._context_view(st),
                }
                self.sessions.save(st)
                return out

            plan = await anyio.to_thread.run_sync(
                self.cart_planner.plan,
                st.last_recipe_id,
                st.user_servings,
                None,
                st.constraints,
            )
            st.cart = plan.get("cart", st.cart)

            out = {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": "Mình đã lên giỏ theo món hiện tại. Bạn muốn chỉnh khẩu phần hoặc bỏ nguyên liệu nào không?",
                "plan": plan,
                "cart": st.cart,
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out

        # fallback
        st.last_search_results = raw_results
        out = {
            "session_id": session_id,
            "nlu": nlu_out,
            "reply": "Bạn muốn nấu món gì? Mình có thể tìm công thức và lên giỏ nguyên liệu giúp bạn.",
            "recipes": raw_results,
            "context": self._context_view(st),
        }
        self.sessions.save(st)
        return out

    async def _handle_rule(self, rule: RuleResult, st: SessionState) -> Dict[str, Any]:
        action = rule.action
        payload = rule.payload

        if action == "pick_recipe":
            idx = int(payload["index_1based"])
            if not st.last_search_results or idx < 1 or idx > len(st.last_search_results):
                return {"reply": "Mình chưa có danh sách món để chọn. Bạn hãy tìm món trước nhé.", "context": self._context_view(st)}
            picked = st.last_search_results[idx - 1]
            st.last_recipe_id = picked.get("id")
            st.last_recipe_title = picked.get("title")
            return {
                "reply": f"OK, mình chọn: {st.last_recipe_title}. Bạn muốn bao nhiêu khẩu phần để mình lên giỏ?",
                "picked_recipe": picked,
                "context": self._context_view(st),
            }

        if action == "set_servings":
            st.user_servings = int(payload["user_servings"])
            return {
                "reply": f"Đã set {st.user_servings} khẩu phần. Bạn muốn mình lên giỏ nguyên liệu không? (gõ: 'lên giỏ')",
                "context": self._context_view(st),
            }

        if action == "add_exclude":
            ing = payload["ingredient"]
            st.constraints.setdefault("exclude_ingredients", [])
            st.constraints["exclude_ingredients"].append(ing)
            return {
                "reply": f"OK, mình sẽ bỏ {ing}. Bạn muốn lên giỏ lại theo món hiện tại không? (gõ: 'lên giỏ')",
                "context": self._context_view(st),
            }

        if action == "show_cart":
            return {"reply": "Đây là giỏ hiện tại của bạn.", "cart": st.cart, "context": self._context_view(st)}

        if action == "plan_cart":
            if not st.last_recipe_id:
                return {"reply": "Bạn chọn giúp mình 1 món (ví dụ: 'chọn món số 2') rồi mình lên giỏ nguyên liệu nhé.", "context": self._context_view(st)}

            plan = await anyio.to_thread.run_sync(
                self.cart_planner.plan,
                st.last_recipe_id,
                st.user_servings,
                None,
                st.constraints,
            )
            st.cart = plan.get("cart", st.cart)
            return {
                "reply": "Mình đã tạo giỏ nguyên liệu. Bạn muốn chỉnh gì nữa không?",
                "plan": plan,
                "cart": st.cart,
                "context": self._context_view(st),
            }

        return {"reply": "Mình chưa hiểu yêu cầu. Bạn nói rõ hơn giúp mình nhé.", "context": self._context_view(st)}

    def _build_refined_query(self, text: str, slots: Dict[str, List[str]]) -> str:
        parts = [text]
        for vals in slots.values():
            if isinstance(vals, list):
                parts.extend([v for v in vals if v])
        return " ".join(parts).strip()

    def _ensure_recipe_selected(self, st: SessionState) -> None:
        if st.last_recipe_id:
            return
        if st.last_search_results:
            st.last_recipe_id = st.last_search_results[0].get("id")
            st.last_recipe_title = st.last_search_results[0].get("title")

    def _context_view(self, st: SessionState) -> Dict[str, Any]:
        return {
            "last_recipe_id": st.last_recipe_id,
            "last_recipe_title": st.last_recipe_title,
            "user_servings": st.user_servings,
            "exclude_ingredients": st.constraints.get("exclude_ingredients", []),
            "cart_items": len(st.cart.get("items", [])) if isinstance(st.cart, dict) else 0,
            "last_results": len(st.last_search_results),
        }
