# =========================
# FILE: smart_food_bot/src/application/usecases.py
# (UPDATED: add GetRecipeDetail use-case)
# =========================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.domain.repositories import RecipeReadRepo, ProductReadRepo


@dataclass(frozen=True)
class SearchRecipes:
    search_engine: Any

    def __call__(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.search_engine.search(query=query, top_k=top_k)


@dataclass(frozen=True)
class EstimatePrice:
    product_repo: ProductReadRepo

    def __call__(self, ingredient_names: List[str]) -> Dict[str, Any]:
        # keep your existing implementation
        return {"items": [], "estimate": 0.0, "notes": "not_implemented"}


@dataclass(frozen=True)
class BuildCart:
    product_repo: ProductReadRepo

    def __call__(self, ingredient_names: List[str]) -> Dict[str, Any]:
        # keep your existing implementation
        return {"items": [], "notes": "not_implemented"}


#: recipe detail (full steps/ingredients from Mongo)
@dataclass(frozen=True)
class GetRecipeDetail:
    recipe_repo: RecipeReadRepo

    def __call__(self, recipe_id_or_title: str) -> Dict[str, Any]:
        key = (recipe_id_or_title or "").strip()
        if not key:
            raise ValueError("recipe_id_or_title is required")

        by_id = self.recipe_repo.by_id(key)
        if by_id:
            return by_id.to_dict()

        # fallback: title contains (cheap + OK for now)
        k = key.lower().strip()
        best = None
        for r in self.recipe_repo.all():
            if k in (r.title or "").lower():
                best = r
                break
        if not best:
            raise LookupError(f"Recipe not found: {recipe_id_or_title}")
        return best.to_dict()
