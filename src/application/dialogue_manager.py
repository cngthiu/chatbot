# =========================
# FILE: smart_food_bot/src/application/dialogue_manager.py
# (UPDATED: handle ask_recipe_detail properly, return full steps)
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
from src.application import response_templates as rt

log = logging.getLogger("app.dialogue_manager")

_RE_ASK_RECIPE_DETAIL = re.compile(r"\b(công\s*thức|cách\s*làm|hướng\s*dẫn|nấu\s*thế\s*nào)\b", re.IGNORECASE)


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
    ) -> None:
        self.sessions = sessions
        self.rules = rule_engine
        self.nlu = nlu
        self.search_uc = search_uc
        self.recipe_detail_uc = recipe_detail_uc
        self.cart_planner = cart_planner
        self.min_intent_conf = min_intent_conf

    async def handle(self, session_id: str, text: str) -> Dict[str, Any]:
        st = self.sessions.get_or_create(session_id)
        st.last_user_text = text

        # 1) Rule-first (greet/thanks/bye + commands)
        rule = self.rules.try_match(text, st)
        if rule:
            out = await self._handle_rule(rule, st)
            self.sessions.save(st)
            return out

        # 2) Concurrent NLU + Search
        nlu_coro = anyio.to_thread.run_sync(self.nlu.predict, text)
        search_coro = anyio.to_thread.run_sync(self.search_uc, text, 5)
        nlu_out, raw_results = await asyncio.gather(nlu_coro, search_coro)

        nlu_out = dict(nlu_out or {})
        nlu_out["slots"] = normalize_slots(nlu_out.get("slots", {}) or {})

        intent = (nlu_out.get("intent") or "fallback").strip()
        conf = float(nlu_out.get("intent_confidence", 0.0) or 0.0)
        slots: Dict[str, List[str]] = nlu_out.get("slots", {}) or {}

        # ✅ If user explicitly asks recipe instructions, treat as ask_recipe_detail
        if _RE_ASK_RECIPE_DETAIL.search(text):
            intent = "ask_recipe_detail"

        st.last_intent = intent

        if conf < self.min_intent_conf and intent not in ("ask_recipe_detail",):
            st.last_search_results = raw_results
            out = {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": rt.ask_clarify_reply(),
                "recipes": raw_results,
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out

        # 3) Routing
        if intent in ("search_recipe", "refine_search"):
            refined_query = self._build_refined_query(text, slots)
            refined_results = await anyio.to_thread.run_sync(self.search_uc, refined_query, 5)
            st.last_search_results = refined_results

            out = {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": rt.prompt_pick_recipe_reply(),
                "recipes": refined_results,
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out

        # ✅ FIX: ask_recipe_detail now returns full recipe (ingredients + steps)
        if intent == "ask_recipe_detail":
            recipe_key = self._resolve_recipe_key_for_detail(st, slots, raw_results)
            if not recipe_key:
                st.last_search_results = raw_results
                out = {
                    "session_id": session_id,
                    "nlu": nlu_out,
                    "reply": "Bạn muốn công thức món nào? Bạn chọn 1 món trong danh sách (ví dụ: 'chọn món số 2') nhé.",
                    "recipes": raw_results,
                    "context": self._context_view(st),
                }
                self.sessions.save(st)
                return out

            detail = await anyio.to_thread.run_sync(self.recipe_detail_uc, recipe_key)

            # update context
            st.last_recipe_id = detail.get("id") or st.last_recipe_id
            st.last_recipe_title = detail.get("title") or st.last_recipe_title

            out = {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": f"Dạ đây là công thức món {detail.get('title','')}. Bạn muốn mình lên giỏ nguyên liệu không? (gõ: 'lên giỏ')",
                "recipe_detail": detail,  # ✅ includes steps
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out

        if intent in ("add_ingredients_to_cart", "ask_price_estimate"):
            self._ensure_recipe_selected(st, raw_results)
            if not st.last_recipe_id:
                st.last_search_results = raw_results
                out = {
                    "session_id": session_id,
                    "nlu": nlu_out,
                    "reply": rt.prompt_pick_recipe_reply(),
                    "recipes": raw_results,
                    "context": self._context_view(st),
                }
                self.sessions.save(st)
                return out

            if st.user_servings is None:
                out = {
                    "session_id": session_id,
                    "nlu": nlu_out,
                    "reply": rt.prompt_servings_reply(),
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
                "reply": rt.cart_done_reply(),
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
            "reply": "Bạn muốn nấu món gì? Ví dụ: 'canh bí đỏ', 'thịt kho', 'phở bò'...",
            "recipes": raw_results,
            "context": self._context_view(st),
        }
        self.sessions.save(st)
        return out

    async def _handle_rule(self, rule: RuleResult, st: SessionState) -> Dict[str, Any]:
        # keep your current rule handler (greet/thanks/bye/pick/set_servings/plan_cart/show_cart/add_exclude)
        # NOTE: plan_cart uses cart_planner; pick_recipe updates st.last_recipe_id
        action = rule.action
        payload = rule.payload

        if action == "greet":
            return {"reply": rt.greet_reply(), "context": self._context_view(st)}
        if action == "thanks":
            return {"reply": rt.thanks_reply(), "context": self._context_view(st)}
        if action == "bye":
            return {"reply": rt.bye_reply(), "context": self._context_view(st)}
        if action == "apology":
            return {"reply": rt.apology_reply(), "context": self._context_view(st)}

        if action == "pick_recipe":
            idx = int(payload["index_1based"])
            if not st.last_search_results or idx < 1 or idx > len(st.last_search_results):
                return {"reply": "Mình chưa có danh sách món để chọn. Bạn hãy tìm món trước nhé.", "context": self._context_view(st)}
            picked = st.last_search_results[idx - 1]
            st.last_recipe_id = picked.get("id")
            st.last_recipe_title = picked.get("title")
            return {
                "reply": f"OK, mình chọn: {st.last_recipe_title}. {rt.prompt_servings_reply()}",
                "picked_recipe": picked,
                "context": self._context_view(st),
            }

        if action == "set_servings":
            st.user_servings = int(payload["user_servings"])
            return {"reply": f"Dạ OK, mình set {st.user_servings} khẩu phần. Bạn muốn 'lên giỏ' luôn không?", "context": self._context_view(st)}

        if action == "add_exclude":
            ing = payload["ingredient"]
            st.constraints.setdefault("exclude_ingredients", [])
            st.constraints["exclude_ingredients"].append(ing)
            return {"reply": f"Dạ được, mình sẽ bỏ {ing}. Bạn muốn 'lên giỏ' lại không?", "context": self._context_view(st)}

        if action == "show_cart":
            return {"reply": "Dạ đây là giỏ hiện tại của bạn.", "cart": st.cart, "context": self._context_view(st)}

        if action == "plan_cart":
            if not st.last_recipe_id:
                return {"reply": rt.prompt_pick_recipe_reply(), "context": self._context_view(st)}
            if st.user_servings is None:
                return {"reply": rt.prompt_servings_reply(), "context": self._context_view(st)}
            plan = await anyio.to_thread.run_sync(
                self.cart_planner.plan,
                st.last_recipe_id,
                st.user_servings,
                None,
                st.constraints,
            )
            st.cart = plan.get("cart", st.cart)
            return {"reply": rt.cart_done_reply(), "plan": plan, "cart": st.cart, "context": self._context_view(st)}

        return {"reply": "Mình chưa hiểu ý bạn. Bạn nói rõ hơn giúp mình nhé.", "context": self._context_view(st)}

    def _build_refined_query(self, text: str, slots: Dict[str, List[str]]) -> str:
        parts = [text]
        for key in ("DISH", "INGREDIENT", "TASTE", "EXCLUDE"):
            parts.extend(slots.get(key, []))
        return " ".join([p for p in parts if p]).strip()

    def _ensure_recipe_selected(self, st: SessionState, raw_results: List[Dict[str, Any]]) -> None:
        if st.last_recipe_id:
            return
        results = st.last_search_results or raw_results
        if results:
            st.last_recipe_id = results[0].get("id")
            st.last_recipe_title = results[0].get("title")
            st.last_search_results = results

    def _resolve_recipe_key_for_detail(
        self, st: SessionState, slots: Dict[str, List[str]], raw_results: List[Dict[str, Any]]
    ) -> Optional[str]:
        # 1) if user already selected a recipe
        if st.last_recipe_id:
            return st.last_recipe_id
        # 2) if NLU extracted dish name
        dish = (slots.get("DISH") or [])
        if dish:
            return dish[0]
        # 3) fallback to top search result
        results = st.last_search_results or raw_results
        if results:
            return results[0].get("id") or results[0].get("title")
        return None

    def _context_view(self, st: SessionState) -> Dict[str, Any]:
        return {
            "last_recipe_id": st.last_recipe_id,
            "last_recipe_title": st.last_recipe_title,
            "user_servings": st.user_servings,
            "exclude_ingredients": st.constraints.get("exclude_ingredients", []),
            "cart_items": len(st.cart.get("items", [])) if isinstance(st.cart, dict) else 0,
            "last_results": len(st.last_search_results),
        }
