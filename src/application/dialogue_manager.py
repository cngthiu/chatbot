# smart_food_bot/src/application/dialogue_manager.py
from __future__ import annotations

import anyio
import logging
from typing import Any, Dict, Optional, List

from src.infrastructure.session_store import InMemorySessionStore, SessionState
from src.application.rule_engine import RuleEngine
from src.services.nlu_engine import NLUEngine
from src.application.usecases import SearchRecipes
from src.application.cart_planner import CartPlanner  # bạn đã có file này
from src.services.search_engine import HybridSearchEngine

log = logging.getLogger("app.dialogue_manager")

class DialogueManager:
    """
    WHY: Single orchestrator for smooth multi-turn conversation:
    - Keep context per session_id
    - Rule-based fast path
    - ML fallback (PhoBERT)
    - Action routing (search -> pick -> plan cart -> pricing)
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

        # 1) RULE-FIRST: instant response for common commands
        rule = self.rules.try_match(text, st)
        if rule:
            out = await self._handle_rule(rule.action, rule.payload, st)
            self.sessions.save(st)
            return out

        # 2) LOW-LATENCY: run NLU + BM25/Hybrid search in parallel.
        #    Search on raw text is a good speculative candidate; later we may rerank using slots.
        nlu_task = anyio.to_thread.run_sync(self.nlu.predict, text)
        search_task = anyio.to_thread.run_sync(self.search_uc, text, 5)

        nlu_out, raw_results = await anyio.gather(nlu_task, search_task)

        st.last_intent = nlu_out.get("intent", "")
        slots = nlu_out.get("slots", {})

        # If search intent: refine query with extracted slots and rerank quickly
        if st.last_intent in ("search_recipe", "refine_search", "ask_recipe_detail"):
            q_parts = [text]
            for vals in slots.values():
                q_parts.extend(vals)
            refined_query = " ".join(q_parts).strip()

            # refine search (fast CPU) in thread
            refined_results = await anyio.to_thread.run_sync(self.search_uc, refined_query, 5)
            st.last_search_results = refined_results
            return {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": "Mình tìm được vài món phù hợp, bạn chọn món số mấy để lên giỏ nguyên liệu?",
                "recipes": refined_results,
                "context": self._context_view(st),
            }

        # If add to cart / price: try to use last selected recipe
        if st.last_intent in ("add_ingredients_to_cart", "ask_price_estimate"):
            if not st.last_recipe_id and st.last_search_results:
                # pick first as fallback
                st.last_recipe_id = st.last_search_results[0].get("id")
                st.last_recipe_title = st.last_search_results[0].get("title")

            if st.last_recipe_id:
                plan = await anyio.to_thread.run_sync(
                    self.cart_planner.plan,
                    st.last_recipe_id,
                    st.user_servings,
                    None,
                    st.constraints,
                )
                st.cart = plan.get("cart", st.cart)
                return {
                    "session_id": session_id,
                    "nlu": nlu_out,
                    "reply": "Mình đã lên giỏ theo món bạn chọn. Bạn muốn chỉnh khẩu phần hay bỏ nguyên liệu nào không?",
                    "plan": plan,
                    "context": self._context_view(st),
                }

            return {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": "Bạn muốn mua nguyên liệu cho món nào? Hãy chọn 1 món trong danh sách mình gợi ý nhé.",
                "recipes": st.last_search_results or raw_results,
                "context": self._context_view(st),
            }

        # fallback
        st.last_search_results = raw_results
        return {
            "session_id": session_id,
            "nlu": nlu_out,
            "reply": "Bạn muốn nấu món gì? Mình có thể tìm công thức và lên giỏ nguyên liệu giúp bạn.",
            "recipes": raw_results,
            "context": self._context_view(st),
        }

    async def _handle_rule(self, action: str, payload: Dict[str, Any], st: SessionState) -> Dict[str, Any]:
        if action == "pick_recipe":
            idx = int(payload["index_1based"])
            if not st.last_search_results or idx < 1 or idx > len(st.last_search_results):
                return {
                    "reply": "Mình chưa có danh sách món để chọn. Bạn hãy tìm món trước nhé.",
                    "context": self._context_view(st),
                }
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
                "reply": f"Đã set {st.user_servings} khẩu phần. Bạn muốn mình lên giỏ nguyên liệu không?",
                "context": self._context_view(st),
            }

        if action == "add_exclude":
            ing = payload["ingredient"]
            st.constraints.setdefault("exclude_ingredients", [])
            st.constraints["exclude_ingredients"].append(ing)
            return {
                "reply": f"OK, mình sẽ bỏ {ing}. Bạn muốn lên giỏ lại theo món hiện tại không?",
                "context": self._context_view(st),
            }

        if action == "show_cart":
            return {
                "reply": "Đây là giỏ hiện tại của bạn.",
                "cart": st.cart,
                "context": self._context_view(st),
            }

        if action == "plan_cart":
            if not st.last_recipe_id:
                return {
                    "reply": "Bạn chọn giúp mình 1 món (ví dụ: 'chọn món số 2') rồi mình lên giỏ nguyên liệu nhé.",
                    "context": self._context_view(st),
                }
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
                "context": self._context_view(st),
            }

        return {"reply": "Mình chưa hiểu yêu cầu. Bạn nói rõ hơn giúp mình nhé.", "context": self._context_view(st)}

    def _context_view(self, st: SessionState) -> Dict[str, Any]:
        return {
            "last_recipe_id": st.last_recipe_id,
            "last_recipe_title": st.last_recipe_title,
            "user_servings": st.user_servings,
            "exclude_ingredients": st.constraints.get("exclude_ingredients", []),
            "cart_items": len(st.cart.get("items", [])),
            "last_results": len(st.last_search_results),
        }
