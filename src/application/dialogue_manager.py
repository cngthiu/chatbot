# =========================
# FILE: smart_food_bot/src/application/dialogue_manager.py
# (FIX: do NOT run NLU/search when awaiting servings; lock selection)
# =========================
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional

import anyio

from src.infrastructure.session_store import InMemorySessionStore, SessionState
from src.application.rule_engine import RuleEngine, RuleResult
from src.application.usecases import SearchRecipes, GetRecipeDetail
from src.application.cart_planner import CartPlanner
from src.services.nlu_engine import NLUEngine
from src.services.nlu_postprocess import normalize_slots
from src.application.response_composer import ResponseComposer

log = logging.getLogger("app.dialogue_manager")

_RE_ASK_RECIPE_DETAIL = re.compile(
    r"\b(công\s*thức|cách\s*làm|hướng\s*dẫn|nấu\s*thế\s*nào|cách\s*nấu)\b",
    re.IGNORECASE,
)


class DialogueManager:
    def __init__(
        self,
        sessions: InMemorySessionStore,
        rule_engine: RuleEngine,
        nlu: NLUEngine,
        search_uc: SearchRecipes,
        recipe_detail_uc: GetRecipeDetail,
        cart_planner: CartPlanner,
        min_intent_conf: float = 0.55,
        composer: Optional[ResponseComposer] = None,
    ) -> None:
        self.sessions = sessions
        self.rules = rule_engine
        self.nlu = nlu
        self.search_uc = search_uc
        self.recipe_detail_uc = recipe_detail_uc
        self.cart_planner = cart_planner
        self.min_intent_conf = min_intent_conf
        self.composer = composer or ResponseComposer()

    async def handle(self, session_id: str, text: str) -> Dict[str, Any]:
        st = self.sessions.get_or_create(session_id)
        st.last_user_text = text

        # ✅ Rule-first, but now rule can parse standalone "3" when awaiting servings
        rule = self.rules.try_match(text, st)
        if rule:
            out = await self._handle_rule(rule, st)
            self.sessions.save(st)
            return out

        # ✅ If awaiting something important, do NOT run speculative NLU/search.
        if st.awaiting == "servings":
            # User didn't give a number -> ask again, don't search new recipes
            out = {
                "session_id": session_id,
                "reply": self.composer.prompt_servings(recipe_title=st.last_recipe_title),
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out

        # Normal path: concurrent NLU + search
        nlu_coro = anyio.to_thread.run_sync(self.nlu.predict, text)
        search_coro = anyio.to_thread.run_sync(self.search_uc, text, 5)
        nlu_out, raw_results = await asyncio.gather(nlu_coro, search_coro)

        nlu_out = dict(nlu_out or {})
        nlu_out["slots"] = normalize_slots(nlu_out.get("slots", {}) or {})
        slots: Dict[str, List[str]] = nlu_out.get("slots", {}) or {}

        intent = (nlu_out.get("intent") or "fallback").strip()
        conf = float(nlu_out.get("intent_confidence", 0.0) or 0.0)
        if _RE_ASK_RECIPE_DETAIL.search(text):
            intent = "ask_recipe_detail"
        st.last_intent = intent

        if conf < self.min_intent_conf and intent != "ask_recipe_detail":
            st.last_search_results = raw_results
            out = {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": self.composer.clarify(),
                "recipes": raw_results,
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out

        if intent in ("search_recipe", "refine_search"):
            refined_query = self._build_refined_query(text, slots)
            refined_results = await anyio.to_thread.run_sync(self.search_uc, refined_query, 5)
            st.last_search_results = refined_results
            st.awaiting = "pick_recipe"  # ✅ waiting for user to choose
            out = {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": self.composer.prompt_pick_recipe(results_count=len(refined_results)),
                "recipes": refined_results,
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out

        if intent == "ask_recipe_detail":
            detail = await self._get_recipe_detail(st, slots, raw_results, text)
            if detail is None:
                st.last_search_results = raw_results
                st.awaiting = "pick_recipe"
                out = {
                    "session_id": session_id,
                    "nlu": nlu_out,
                    "reply": self.composer.prompt_pick_recipe(results_count=len(raw_results)),
                    "recipes": raw_results,
                    "context": self._context_view(st),
                }
                self.sessions.save(st)
                return out

            st.last_recipe_id = detail.get("id") or st.last_recipe_id
            st.last_recipe_title = detail.get("title") or st.last_recipe_title
            out = {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": self.composer.recipe_detail_intro(st.last_recipe_title or "món này"),
                "recipe_detail": detail,
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out

        if intent == "add_ingredients_to_cart":
            plan_out = await self._plan_cart_from_context(st, raw_results)
            self.sessions.save(st)
            return {"session_id": session_id, "nlu": nlu_out, **plan_out, "context": self._context_view(st)}

        if intent == "ask_price_estimate":
            estimate = await self._estimate_price(st, raw_results)
            out = {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": self.composer.price_estimate_intro(),
                "plan": estimate.get("plan"),
                "cart": estimate.get("cart"),
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out

        st.last_search_results = raw_results
        out = {
            "session_id": session_id,
            "nlu": nlu_out,
            "reply": self.composer.fallback(),
            "recipes": raw_results,
            "context": self._context_view(st),
        }
        self.sessions.save(st)
        return out

    async def _handle_rule(self, rule: RuleResult, st: SessionState) -> Dict[str, Any]:
        action = rule.action
        payload = rule.payload

        if action == "greet":
            return {"reply": self.composer.greet(), "context": self._context_view(st)}
        if action == "thanks":
            return {"reply": self.composer.thanks(), "context": self._context_view(st)}
        if action == "bye":
            return {"reply": self.composer.bye(), "context": self._context_view(st)}
        if action == "apology":
            return {"reply": self.composer.apology(), "context": self._context_view(st)}

        if action == "pick_recipe":
            idx = int(payload["index_1based"])
            if not st.last_search_results or idx < 1 or idx > len(st.last_search_results):
                st.awaiting = "pick_recipe"
                return {"reply": self.composer.prompt_pick_recipe(results_count=len(st.last_search_results)), "context": self._context_view(st)}

            picked = st.last_search_results[idx - 1]
            st.last_recipe_id = picked.get("id")
            st.last_recipe_title = picked.get("title")
            st.awaiting = "servings"  # ✅ now expecting servings, so "3" won't trigger search
            return {
                "reply": self.composer.prompt_servings(recipe_title=st.last_recipe_title),
                "picked_recipe": picked,
                "context": self._context_view(st),
            }

        if action == "set_servings":
            st.user_servings = int(payload["user_servings"])
            st.awaiting = None  # ✅ clear awaiting
            return {"reply": "Dạ OK. Bạn muốn mình 'lên giỏ' luôn không?", "context": self._context_view(st)}

        if action == "plan_cart":
            plan_out = await self._plan_cart_from_context(st, raw_results=None)
            return {**plan_out, "context": self._context_view(st)}

        if action == "show_cart":
            return {"reply": "Dạ đây là giỏ hiện tại của bạn.", "cart": st.cart, "context": self._context_view(st)}

        if action == "add_exclude":
            ing = payload["ingredient"]
            st.constraints.setdefault("exclude_ingredients", [])
            st.constraints["exclude_ingredients"].append(ing)
            return {"reply": f"Dạ được, mình sẽ bỏ {ing}. Bạn muốn mình 'lên giỏ' lại không?", "context": self._context_view(st)}

        return {"reply": self.composer.clarify(), "context": self._context_view(st)}

    def _build_refined_query(self, text: str, slots: Dict[str, List[str]]) -> str:
        parts = [text]
        for key in ("DISH", "INGREDIENT", "TASTE", "EXCLUDE"):
            parts.extend(slots.get(key, []))
        return " ".join([p for p in parts if p]).strip()

    async def _get_recipe_detail(
        self,
        st: SessionState,
        slots: Dict[str, List[str]],
        raw_results: List[Dict[str, Any]],
        text: str,
    ) -> Optional[Dict[str, Any]]:
        if st.last_recipe_id:
            return await anyio.to_thread.run_sync(self.recipe_detail_uc, st.last_recipe_id)

        dish = slots.get("DISH") or []
        if dish:
            top = await anyio.to_thread.run_sync(self.search_uc, dish[0], 1)
            if top:
                rid = top[0].get("id") or top[0].get("title")
                if rid:
                    return await anyio.to_thread.run_sync(self.recipe_detail_uc, rid)

        if raw_results:
            rid = raw_results[0].get("id") or raw_results[0].get("title")
            if rid:
                return await anyio.to_thread.run_sync(self.recipe_detail_uc, rid)

        top2 = await anyio.to_thread.run_sync(self.search_uc, text, 1)
        if top2:
            rid = top2[0].get("id") or top2[0].get("title")
            if rid:
                return await anyio.to_thread.run_sync(self.recipe_detail_uc, rid)

        return None

    async def _plan_cart_from_context(self, st: SessionState, raw_results: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        # lock recipe: DO NOT override if already chosen
        if not st.last_recipe_id:
            results = st.last_search_results or (raw_results or [])
            if results:
                st.last_recipe_id = results[0].get("id")
                st.last_recipe_title = results[0].get("title")
                st.last_search_results = results

        if not st.last_recipe_id:
            st.awaiting = "pick_recipe"
            return {"reply": self.composer.prompt_pick_recipe(results_count=len(st.last_search_results)), "recipes": st.last_search_results or []}

        if st.user_servings is None:
            st.awaiting = "servings"
            return {"reply": self.composer.prompt_servings(recipe_title=st.last_recipe_title)}

        plan = await anyio.to_thread.run_sync(
            self.cart_planner.plan,
            st.last_recipe_id,
            st.user_servings,
            None,
            st.constraints,
        )
        st.cart = plan.get("cart", st.cart)

        warnings = plan.get("warnings") or []
        cart_items = len(st.cart.get("items", [])) if isinstance(st.cart, dict) else 0
        st.awaiting = None

        return {"reply": self.composer.cart_done(cart_items=cart_items, warnings=warnings), "plan": plan, "cart": st.cart}

    async def _estimate_price(self, st: SessionState, raw_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not st.last_recipe_id:
            results = st.last_search_results or raw_results
            if results:
                st.last_recipe_id = results[0].get("id")
                st.last_recipe_title = results[0].get("title")
                st.last_search_results = results

        if not st.last_recipe_id:
            st.awaiting = "pick_recipe"
            return {"plan": None, "cart": None}

        if st.user_servings is None:
            # estimate can default to recipe servings, but if unknown ask
            try:
                detail = await anyio.to_thread.run_sync(self.recipe_detail_uc, st.last_recipe_id)
                st.user_servings = int(detail.get("servings") or 0) or 1
            except Exception:
                st.user_servings = 1

        plan = await anyio.to_thread.run_sync(
            self.cart_planner.plan,
            st.last_recipe_id,
            st.user_servings,
            None,
            st.constraints,
        )
        return {"plan": plan, "cart": plan.get("cart")}

    def _context_view(self, st: SessionState) -> Dict[str, Any]:
        return {
            "awaiting": st.awaiting,
            "last_recipe_id": st.last_recipe_id,
            "last_recipe_title": st.last_recipe_title,
            "user_servings": st.user_servings,
            "exclude_ingredients": st.constraints.get("exclude_ingredients", []),
            "cart_items": len(st.cart.get("items", [])) if isinstance(st.cart, dict) else 0,
            "last_results": len(st.last_search_results),
        }

    def _log_route(
        self,
        intent: str,
        conf: float,
        st: SessionState,
        slots: Dict[str, List[str]],
        handler: str,
    ) -> None:
        log.info(
            "route handler=%s intent=%s conf=%.2f recipe_id=%s servings=%s slots=%s",
            handler,
            intent,
            conf,
            st.last_recipe_id,
            st.user_servings,
            slots,
        )
