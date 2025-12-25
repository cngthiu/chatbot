# =========================
# FILE: smart_food_bot/src/api/schemas.py
# (UPDATED: add recipe_detail field into ChatResponse)
# =========================
from __future__ import annotations

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class NLUResult(BaseModel):
    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    slots: Dict[str, List[str]]


class Recipe(BaseModel):
    id: str
    title: str
    summary: Optional[str] = None
    ingredients: List[str] = Field(default_factory=list)
    cook_time: Optional[int] = None
    servings: Optional[int] = None
    image: Optional[str] = None
    score: float = 0.0


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Client session id to keep context")
    text: str = Field(..., example="Cho mình công thức canh bí đỏ thịt bằm")


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    nlu: Optional[Dict[str, Any]] = None
    recipes: Optional[List[Dict[str, Any]]] = None
    picked_recipe: Optional[Dict[str, Any]] = None

    recipe_detail: Optional[Dict[str, Any]] = None

    plan: Optional[Dict[str, Any]] = None
    cart: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class PlanCartRequest(BaseModel):
    recipe_id: Optional[str] = None
    recipe_title: Optional[str] = None
    user_servings: Optional[int] = Field(default=None, ge=1)
    budget: Optional[float] = Field(default=None, ge=0)
    constraints: Optional[Dict[str, Any]] = None


class PlanCartResponse(BaseModel):
    recipe: Dict[str, Any]
    ingredient_plan: List[Dict[str, Any]]
    unmatched_ingredients: List[Dict[str, Any]]
    cart: Dict[str, Any]
    warnings: List[str]
