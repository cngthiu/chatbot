# =========================
# FILE: smart_food_bot/src/infrastructure/mongo_repositories.py
# FIX: No Load-All-To-RAM (Direct Query)
# =========================
from __future__ import annotations
from typing import List, Dict, Any, Optional
import logging
from pymongo.collection import Collection
from bson import ObjectId
from src.domain.entities import Recipe, Product, Ingredient
from src.domain.repositories import RecipeReadRepo, ProductReadRepo

log = logging.getLogger("infra.mongo_repo")

def _as_str_id(v: Any) -> str:
    return str(v)

class MongoRecipeRepository(RecipeReadRepo):
    """
    Hybrid Repo: 
    - .all() vẫn load để build Index cho Search Engine (chấp nhận tốn RAM 1 lần đầu).
    - .by_id() query trực tiếp để nhanh và tiết kiệm khi truy cập chi tiết.
    """
    def __init__(self, col: Collection) -> None:
        self._col = col
        # Cache nhẹ để build index, nhưng có thể tối ưu sau nếu data quá lớn
        self._items = [self._parse_recipe(doc) for doc in col.find({})]
        if self._items:
            log.info("MongoRecipeRepository loaded %d recipes for Indexing", len(self._items))
        else:
            log.warning("MongoRecipeRepository: recipes collection is empty")

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
        except Exception:
            return Recipe(id="err", title="Error", summary="", ingredients=[])

    def all(self) -> List[Recipe]:
        return self._items

    def by_id(self, recipe_id: str) -> Recipe | None:
        # Query trực tiếp DB thay vì loop trong list để luôn có data mới nhất
        try:
            doc = self._col.find_one({"id": recipe_id})
            if not doc:
                # Thử tìm bằng _id object nếu id string ko thấy
                try:
                    doc = self._col.find_one({"_id": ObjectId(recipe_id)})
                except:
                    pass
            if doc:
                return self._parse_recipe(doc)
            return None
        except Exception:
            return None

class MongoProductRepository(ProductReadRepo):
    """
    Direct-Query Only:
    Không bao giờ load toàn bộ sản phẩm vào RAM.
    """
    def __init__(self, col: Collection) -> None:
        self._col = col
        # KHÔNG load _items

    def _parse_product(self, x: Dict[str, Any]) -> Product:
        try:
            price = float(x.get("price", 0))
            discount = float(x.get("discount", 0))
            image = x.get("image")
            if isinstance(image, list) and image: image = str(image[0])
            elif isinstance(image, str): image = image
            
            return Product(
                sku=str(x.get("sku") or x.get("_id") or ""),
                name=str(x.get("name") or "").strip(),
                price=max(0.0, price),
                unit=str(x.get("unit") or "Unit"),
                net_weight=float(x.get("net_weight") or 0),
                measure_unit=str(x.get("measure_unit") or "g"),
                stock=int(x.get("stock") or 0),
                discount=discount,
                image=str(image) if image else None,
            )
        except Exception:
            return Product(sku="err", name="Error", price=0)

    def all(self) -> List[Product]:
        log.warning("Calling .all() on ProductRepo is expensive!")
        return [self._parse_product(doc) for doc in self._col.find({})]

    def find_by_name(self, name: str, top_k: int = 5) -> List[Product]:
        # Dùng MongoDB Regex Search
        q = (name or "").strip()
        if not q: return []
        
        cursor = self._col.find({"name": {"$regex": q, "$options": "i"}}).limit(top_k)
        results = [self._parse_product(doc) for doc in cursor]
        results.sort(key=lambda p: p.price)
        return results