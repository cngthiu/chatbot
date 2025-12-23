# smart_food_bot/src/api/routes.py
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.schemas import ChatRequest, ChatResponse, PlanCartRequest, PlanCartResponse

log = logging.getLogger("api.routes")
router = APIRouter()


# -------------------------
# Dependencies via app.state (Clean DI, tránh import vòng)
# -------------------------
def get_dialogue_manager(request: Request):
    dm = getattr(request.app.state, "dialogue_manager", None)
    if dm is None:
        raise RuntimeError("dialogue_manager not initialized. Check app startup wiring.")
    return dm


def get_cart_planner(request: Request):
    planner = getattr(request.app.state, "cart_planner", None)
    if planner is None:
        raise RuntimeError("cart_planner not initialized. Check app startup wiring.")
    return planner


# -------------------------
# /chat (async, giữ ngữ cảnh + rule-base + không đợi)
# -------------------------
@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, dm=Depends(get_dialogue_manager)) -> Any:
    if not req.session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    try:
        # dm.handle() trả về dict theo contract của ChatResponse
        out = await dm.handle(req.session_id, req.text)
        out.setdefault("session_id", req.session_id)
        out.setdefault("reply", "OK")
        return out
    except Exception as e:
        log.exception("Processing /chat error")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------------
# /plan_cart (giữ endpoint cũ, dùng CartPlanner)
# -------------------------
@router.post("/plan_cart", response_model=PlanCartResponse)
def plan_cart(req: PlanCartRequest, planner=Depends(get_cart_planner)) -> Any:
    recipe_key = (req.recipe_id or req.recipe_title or "").strip()
    if not recipe_key:
        raise HTTPException(status_code=400, detail="recipe_id or recipe_title is required")

    try:
        return planner.plan(
            recipe_id_or_title=recipe_key,
            user_servings=req.user_servings,
            budget=req.budget,
            constraints=req.constraints,
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Processing /plan_cart error")
        raise HTTPException(status_code=500, detail=str(e))
