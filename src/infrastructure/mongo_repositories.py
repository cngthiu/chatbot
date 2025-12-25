# smart_food_bot/src/infrastructure/mongo_repositories.py
from __future__ import annotations
from typing import List, Dict, Any
import logging
from pymongo.collection import Collection
from bson import ObjectId
from src.domain.entities import Recipe, Product, Ingredient
from src.domain.repositories import RecipeReadRepo, ProductReadRepo

log = logging.getLogger("infra.mongo_repo")

def _as_str_id(v: Any) -> str:
    if isinstance(v, ObjectId):
        return str(v)
    return str(v)

class MongoRecipeRepository(RecipeReadRepo):
    """
    Read-only recipe repository backed by MongoDB.
    Loads all recipes once at startup to power BM25/TF-IDF indexing.
    """
    def __init__(self, col: Collection) -> None:
        self._col = col
        self._items: List[Recipe] = [self._parse_recipe(doc) for doc in col.find({})]
        self._by_id = {r.id: r for r in self._items}
        if not self._items:
            log.warning("MongoRecipeRepository: recipes collection is empty")
        else:
            log.info("MongoRecipeRepository loaded %d recipes", len(self._items))

    def _parse_recipe(self, doc: Dict[str, Any]) -> Recipe:
        try:
            ingredients = [
                Ingredient(
                    name=(i.get("name") or "").strip(),
                    qty=i.get("qty"),
                    unit=i.get("unit"),
                    type=i.get("type"),
                )
                for i in (doc.get("ingredients") or [])
            ]
            return Recipe(
                id=_as_str_id(doc.get("id") or doc.get("_id")),
                title=(doc.get("title") or "").strip(),
                summary=(doc.get("summary") or "").strip(),
                ingredients=ingredients,
                steps=list(doc.get("steps") or []),
                cook_time=doc.get("cook_time"),
                servings=doc.get("servings"),
                tags=list(doc.get("tags") or []),
                diet=list(doc.get("diet") or []),
                image=doc.get("image"),
                search_keywords=list(doc.get("search_keywords") or []),
            )
        except Exception as e:
            log.exception("Invalid recipe document: %s", doc)
            raise ValueError(f"Invalid recipe document: {e}") from e

    def all(self) -> List[Recipe]:
        return self._items

    def by_id(self, recipe_id: str) -> Recipe | None:
        return self._by_id.get(str(recipe_id))

class MongoProductRepository(ProductReadRepo):
    """
    Read-only product repository backed by MongoDB.
    Simple name-based lookup for building cart/price estimates.
    """
    def __init__(self, col: Collection) -> None:
        self._col = col
        self._items: List[Product] = [self._parse_product(doc) for doc in col.find({})]
        if not self._items:
            log.warning("MongoProductRepository: products collection is empty")
        else:
            log.info("MongoProductRepository loaded %d products", len(self._items))

    def _parse_product(self, x: Dict[str, Any]) -> Product:
        try:
            price = float(x.get("price", 0))
            discount = float(x.get("discount", 0))
            image = None
            img = x.get("image")
            if isinstance(img, list) and img:
                image = str(img[0])
            elif isinstance(img, str):
                image = img
            return Product(
                sku=str(x.get("sku") or x.get("_id") or ""),
                name=str(x.get("name") or "").strip(),
                price=max(0.0, price),
                unit=str(x.get("unit") or "Unit"),
                net_weight=float(x.get("net_weight") or 0),
                measure_unit=str(x.get("measure_unit") or "g"),
                stock=int(x.get("stock") or 0),
                discount=discount,
                image=image,
            )
        except Exception as e:
            log.exception("Invalid product document: %s", x)
            raise ValueError(f"Invalid product document: {e}") from e

    def all(self) -> List[Product]:
        return self._items

    def find_by_name(self, name: str, top_k: int = 5) -> List[Product]:
        q = (name or "").lower().strip()
        if not q:
            return []
        # naive overlap score; stable tiebreak by price asc then stock desc
        def _score(p: Product) -> int:
            p_name = p.name.lower()
            if q in p_name:
                return 3
            qs = set(q.split())
            ps = set(p_name.split())
            inter = len(qs & ps)
            return 1 if inter > 0 else 0

        candidates = [p for p in self._items if _score(p) > 0]
        candidates.sort(key=lambda p: (p.price, -p.stock))
        return candidates[:top_k]
