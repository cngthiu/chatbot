# smart_food_bot/src/application/usecases.py
from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass
from src.domain.repositories import RecipeReadRepo, ProductReadRepo
from src.services.search_engine import HybridSearchEngine
from src.domain.entities import Recipe, Product

@dataclass
class SearchRecipes:
    search: HybridSearchEngine
    def __call__(self, query: str, top_k: int = 5) -> List[Dict]:
        return self.search.search(query=query, top_k=top_k)

@dataclass
class EstimatePrice:
    products: ProductReadRepo
    def __call__(self, ingredient_names: List[str]) -> float:
        # Sum cheapest matching product per ingredient (very naive baseline)
        total = 0.0
        for name in ingredient_names:
            candidates = self.products.find_by_name(name, top_k=1)
            if candidates:
                total += max(0.0, candidates[0].price)
        return round(total, 2)

@dataclass
class BuildCart:
    products: ProductReadRepo
    def __call__(self, ingredient_names: List[str]) -> List[Dict]:
        cart: List[Dict] = []
        for name in ingredient_names:
            match = self.products.find_by_name(name, top_k=1)
            if match:
                p = match[0]
                cart.append({"sku": p.sku, "name": p.name, "price": p.price, "qty": 1, "unit": p.unit})
        return cart
