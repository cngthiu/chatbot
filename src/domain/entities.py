# smart_food_bot/src/domain/entities.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

@dataclass(frozen=True)
class Ingredient:
    name: str
    qty: float | int | None
    unit: str | None
    type: str | None

@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    summary: str
    ingredients: List[Ingredient]
    steps: List[str]
    cook_time: int | None
    servings: int | None
    tags: List[str]
    diet: List[str]
    image: str | None
    search_keywords: List[str]

@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price: float  # base price before discount
    unit: str
    net_weight: float
    measure_unit: str
    stock: int
    discount: float
    image: str | None = None
