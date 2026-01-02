# =========================
# FILE: smart_food_bot/src/application/dialogue_manager.py
# FIX: Smart Context Parsing (Auto detect 'servings') + Auto-Show Ingredients
# =========================
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import anyio

from src.infrastructure.session_store import InMemorySessionStore, SessionState
from src.application.rule_engine import RuleEngine, RuleResult
from src.application.usecases import SearchRecipes, GetRecipeDetail
from src.application.cart_planner import CartPlanner
from src.services.nlu_postprocess import normalize_slots
from src.application.response_composer import ResponseComposer

log = logging.getLogger("app.dialogue_manager")

# Regex bắt ý định hỏi cách làm
_RE_ASK_RECIPE_DETAIL = re.compile(
    r"\b(công\s*thức|cách\s*làm|hướng\s*dẫn|nấu\s*thế\s*nào|cách\s*nấu|chi\s*tiết)\b",
    re.IGNORECASE,
)

# Regex bắt số lượng người ăn (Ví dụ: 3 người, 2 suất, 5 phần)
_RE_SERVINGS_DETECT = re.compile(
    r"\b(\d+)\s*(?:người|suất|phần|bát|tô|chén|khẩu\s*phần)\b",
    re.IGNORECASE
)

class DialogueManager:
    def __init__(
        self,
        sessions: InMemorySessionStore,
        rule_engine: RuleEngine,
        nlu: Any,
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
        t = (text or "").strip()
        st = self.sessions.get_or_create(session_id)
        st.last_user_text = t

        if not t:
            out = {
                "session_id": session_id,
                "reply": self.composer.fallback(),
                "recipes": [],
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out

        # 1. INTELLIGENT PARSING (Bắt số người ăn NGAY LẬP TỨC)
        # Kể cả khi user đang search hay đang chat vu vơ, nếu có số liệu thì lượm ngay.
        servings_match = _RE_SERVINGS_DETECT.search(t)
        if servings_match:
            try:
                detected_servings = int(servings_match.group(1))
                if detected_servings > 0:
                    st.user_servings = detected_servings
                    # Nếu đang ở trạng thái chờ nhập số, thì giải phóng nó luôn
                    if st.awaiting == "servings":
                        st.awaiting = None
            except ValueError:
                pass

        # 2. Rule-Base Priority
        rule = self.rules.try_match(text, st)
        if rule:
            out = await self._handle_rule(rule, st)
            self.sessions.save(st)
            return out

        # 3. Check Blocking State (Chỉ block nếu chưa bắt được servings ở bước 1)
        if st.awaiting == "servings" and st.user_servings is None:
            # Vẫn chưa biết số người -> Tiếp tục hỏi
            out = {
                "session_id": session_id,
                "reply": self.composer.prompt_servings(recipe_title=st.last_recipe_title),
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out
        elif st.awaiting == "servings" and st.user_servings is not None:
            # Đã bắt được số người ở Bước 1 -> Tự động chuyển tiếp
            st.awaiting = None
            # Fallthrough xuống dưới để xử lý tiếp

        # 4. NLU Inference
        nlu_out = await anyio.to_thread.run_sync(self.nlu.predict, t)
        nlu_out = dict(nlu_out or {})
        nlu_out["slots"] = normalize_slots(nlu_out.get("slots", {}) or {})
        slots: Dict[str, List[str]] = nlu_out.get("slots", {}) or {}

        intent = (nlu_out.get("intent") or "fallback").strip()
        conf = float(nlu_out.get("intent_confidence", 0.0) or 0.0)

        # Regex override intents
        if _RE_ASK_RECIPE_DETAIL.search(text):
            intent = "ask_recipe_detail"
            conf = 1.0
        
        st.last_intent = intent
        
        # 5. Search Logic
        raw_results = []
        should_search = (intent in ("search_recipe", "refine_search")) or (conf < self.min_intent_conf)
        
        if should_search:
            query_for_search = t
            if intent in ("search_recipe", "refine_search"):
                query_for_search = self._build_refined_query(t, slots)
            
            raw_results = await anyio.to_thread.run_sync(self.search_uc, query_for_search, 5)
            st.last_search_results = raw_results

        # 6. ROUTING LOGIC

        # Case: Search Recipe -> Trả về list món
        if intent in ("search_recipe", "refine_search") or (conf < self.min_intent_conf and raw_results):
            st.awaiting = "pick_recipe"
            reply_msg = self.composer.prompt_pick_recipe(results_count=len(raw_results))
            
            # Nếu đã biết servings, nhắc nhẹ user
            if st.user_servings:
                reply_msg += f" (Đang tính cho {st.user_servings} người ăn)"

            out = {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": reply_msg,
                "recipes": raw_results,
                "context": self._context_view(st),
            }
            self.sessions.save(st)
            return out

        # Case: Ask Detail (Hỏi chi tiết) -> chỉ trả công thức, không mở popup nguyên liệu
        if intent == "ask_recipe_detail":
            return await self._show_detail_flow(st, nlu_out, slots, raw_results, text, session_id)

        # Case: Add to Cart (Chốt đơn)
        if intent == "add_ingredients_to_cart":
            plan_out = await self._plan_cart_from_context(st, raw_results)
            self.sessions.save(st)
            return {"session_id": session_id, "nlu": nlu_out, **plan_out, "context": self._context_view(st)}

        # Fallback
        out = {
            "session_id": session_id,
            "nlu": nlu_out,
            "reply": self.composer.fallback(),
            "recipes": [],
            "context": self._context_view(st),
        }
        self.sessions.save(st)
        return out
    
    async def _handle_rule(self, rule: RuleResult, st: SessionState) -> Dict[str, Any]:
        action = rule.action
        payload = rule.payload
        
        if action == "greet": return {"reply": self.composer.greet(), "context": self._context_view(st)}
        if action == "thanks": return {"reply": self.composer.thanks(), "context": self._context_view(st)}
        if action == "bye": return {"reply": self.composer.bye(), "context": self._context_view(st)}
        if action == "apology": return {"reply": self.composer.apology(), "context": self._context_view(st)}
        
        if action == "pick_recipe":
            # User chọn số thứ tự (ví dụ: "chọn món 1")
            idx = int(payload["index_1based"])
            if not st.last_search_results or idx < 1 or idx > len(st.last_search_results):
                st.awaiting = "pick_recipe"
                return {"reply": "Số thứ tự không hợp lệ, bạn chọn lại giúp mình nhé.", "context": self._context_view(st)}
            
            picked = st.last_search_results[idx - 1]
            st.last_recipe_id = picked.get("id")
            st.last_recipe_title = picked.get("title")
            
            # === AUTO FLOW: Đã chọn món -> Show luôn chi tiết & Nguyên liệu ===
            return await self._auto_show_detail_and_plan(st, picked)
            
        if action == "set_servings":
            st.user_servings = int(payload["user_servings"])
            st.awaiting = None
            # Đã có số người -> Nếu đã có món -> Lên đơn luôn
            if st.last_recipe_id:
                return await self._plan_cart_from_context(st, None)
            return {"reply": f"Okie, nấu cho {st.user_servings} người. Bạn muốn tìm món gì?", "context": self._context_view(st)}
            
        if action == "plan_cart":
            plan_out = await self._plan_cart_from_context(st, raw_results=None)
            return {**plan_out, "context": self._context_view(st)}
            
        if action == "show_cart":
            return {"reply": "Giỏ hàng hiện tại của bạn đây ạ.", "cart": st.cart, "context": self._context_view(st)}

        return {"reply": self.composer.clarify(), "context": self._context_view(st)}

    # === NEW: Logic tự động hiển thị chi tiết và chuẩn bị lên đơn ===
    async def _auto_show_detail_and_plan(self, st: SessionState, recipe_basic: Dict[str, Any]) -> Dict[str, Any]:
        """
        Khi user chọn món:
        1. Hiển thị thông tin chi tiết (nguyên liệu).
        2. Tính toán luôn giỏ hàng (dùng servings mặc định nếu user chưa nhập).
        """
        # Lấy chi tiết món ăn (ingredients, steps)
        detail = await anyio.to_thread.run_sync(self.recipe_detail_uc, st.last_recipe_id)
        
        reply = f"Đã chọn **{st.last_recipe_title}**. Mình mở danh sách nguyên liệu phù hợp nhé."

        # Nếu chưa có số người ăn -> dùng servings mặc định của món hoặc 1
        if not st.user_servings:
            st.user_servings = int(detail.get("servings") or 1)
            st.awaiting = None

        plan = await anyio.to_thread.run_sync(
            self.cart_planner.plan,
            st.last_recipe_id,
            st.user_servings,
            None,
            st.constraints,
        )

        return {
            "reply": reply,
            "recipe_detail": detail,
            "plan": plan,
            "cart": plan.get("cart"),
            "open_plan": True,
            "context": self._context_view(st),
        }

    async def _show_detail_flow(self, st, nlu_out, slots, raw_results, text, session_id):
        detail = await self._get_recipe_detail(st, slots, raw_results, text)
        if detail is None:
            # Fallback search
            search_backup = await anyio.to_thread.run_sync(self.search_uc, text, 5)
            st.last_search_results = search_backup
            st.awaiting = "pick_recipe"
            return {
                "session_id": session_id,
                "nlu": nlu_out,
                "reply": "Mình không tìm thấy chi tiết món đó. Bạn thử chọn trong danh sách này nhé:",
                "recipes": search_backup,
                "context": self._context_view(st),
            }
        
        st.last_recipe_id = detail.get("id") or st.last_recipe_id
        st.last_recipe_title = detail.get("title") or st.last_recipe_title

        reply = self.composer.recipe_detail_intro(st.last_recipe_title)
        return {
            "session_id": session_id,
            "nlu": nlu_out,
            "reply": reply,
            "recipe_detail": detail,
            "context": self._context_view(st),
        }

    def _build_refined_query(self, text: str, slots: Dict[str, List[str]]) -> str:
        parts = [text]
        for key in ("DISH", "INGREDIENT", "TASTE", "EXCLUDE"):
            parts.extend(slots.get(key, []))
        return " ".join([p for p in parts if p]).strip()

    async def _get_recipe_detail(self, st: SessionState, slots: Dict[str, List[str]], raw_results: List[Dict[str, Any]], text: str) -> Optional[Dict[str, Any]]:
        # Logic cũ vẫn tốt
        if st.last_recipe_id: return await anyio.to_thread.run_sync(self.recipe_detail_uc, st.last_recipe_id)
        
        dish = slots.get("DISH") or []
        if dish:
            top = await anyio.to_thread.run_sync(self.search_uc, dish[0], 1)
            if top:
                rid = top[0].get("id") or top[0].get("title")
                if rid: return await anyio.to_thread.run_sync(self.recipe_detail_uc, rid)

        if raw_results:
             rid = raw_results[0].get("id") or raw_results[0].get("title")
             if rid: return await anyio.to_thread.run_sync(self.recipe_detail_uc, rid)

        top2 = await anyio.to_thread.run_sync(self.search_uc, text, 1)
        if top2:
            rid = top2[0].get("id") or top2[0].get("title")
            if rid: return await anyio.to_thread.run_sync(self.recipe_detail_uc, rid)
            
        return None

    async def _plan_cart_from_context(self, st: SessionState, raw_results: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        if not st.last_recipe_id:
            results = st.last_search_results or (raw_results or [])
            if results:
                st.last_recipe_id = results[0].get("id")
                st.last_recipe_title = results[0].get("title")
                st.last_search_results = results
        
        if not st.last_recipe_id:
             st.awaiting = "pick_recipe"
             return {"reply": self.composer.prompt_pick_recipe(results_count=0), "recipes": []}
             
        if st.user_servings is None:
             # dùng servings mặc định của món để tránh hỏi nhiều
             detail = await anyio.to_thread.run_sync(self.recipe_detail_uc, st.last_recipe_id)
             st.user_servings = int(detail.get("servings") or 1)
             st.awaiting = None
             
        # Thực hiện Plan thật sự và lưu vào session
        plan = await anyio.to_thread.run_sync(self.cart_planner.plan, st.last_recipe_id, st.user_servings, None, st.constraints)
        
        # Merge vào giỏ hàng hiện tại (Logic đơn giản: Cộng dồn)
        # Ở đây ta giả định thay thế hoặc append logic trong SessionStore, 
        # nhưng đơn giản nhất là cập nhật st.cart
        st.cart = plan.get("cart", st.cart)
        
        warnings = plan.get("warnings") or []
        msg = self.composer.cart_done(cart_items=len(st.cart.get("items", [])), warnings=warnings)
        
        return {"reply": msg, "plan": plan, "cart": st.cart}

    async def _estimate_price(self, st: SessionState, raw_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # (Giữ nguyên logic cũ)
        if not st.last_recipe_id and raw_results:
            st.last_recipe_id = raw_results[0].get("id")
        
        if not st.last_recipe_id: return {"plan": None, "cart": None}
        
        if st.user_servings is None: st.user_servings = 1
                
        plan = await anyio.to_thread.run_sync(self.cart_planner.plan, st.last_recipe_id, st.user_servings, None, st.constraints)
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
