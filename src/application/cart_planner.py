# smart_food_bot/src/application/cart_planner.py
from __future__ import annotations

import logging
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.domain.entities import Ingredient, Product, Recipe
from src.domain.repositories import ProductReadRepo, RecipeReadRepo

log = logging.getLogger("app.cart_planner")

_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")


# ----------------------------
# Normalization & tokenization
# ----------------------------
def normalize_vi(text: str) -> str:
    """Normalize Vietnamese strings for robust matching."""
    text = (text or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = _NON_ALNUM.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_norm(text: str) -> List[str]:
    n = normalize_vi(text)
    return [t for t in n.split() if t]


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _final_price(p: Product) -> float:
    return max(0.0, float(p.price) * (1.0 - float(p.discount) / 100.0))


# ----------------------------
# Unit conversion
# ----------------------------
_MASS_UNITS_TO_G: Dict[str, float] = {
    "g": 1.0,
    "gram": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "lạng": 100.0,
    "lang": 100.0,
}
_VOL_UNITS_TO_ML: Dict[str, float] = {
    "ml": 1.0,
    "l": 1000.0,
    "lit": 1000.0,
    "liter": 1000.0,
}


def _to_base_amount(qty: float, unit: str) -> Tuple[Optional[float], Optional[str]]:
    """Convert (qty, unit) -> (base_amount, base_unit) where base_unit in {'g','ml'}."""
    u = normalize_vi(unit)
    if u in _MASS_UNITS_TO_G:
        return qty * _MASS_UNITS_TO_G[u], "g"
    if u in _VOL_UNITS_TO_ML:
        return qty * _VOL_UNITS_TO_ML[u], "ml"
    return None, None


def _product_net_to_base(net_weight: float, measure_unit: str) -> Tuple[Optional[float], Optional[str]]:
    if not net_weight or net_weight <= 0:
        return None, None
    return _to_base_amount(float(net_weight), measure_unit)


def _ceil_div(a: float, b: float) -> int:
    return int(math.ceil(a / b))


def _scale_qty(qty: Any, factor: float) -> Optional[float]:
    try:
        return float(qty) * factor
    except Exception:
        return None


# ----------------------------
# Constraints
# ----------------------------
@dataclass(frozen=True)
class CartConstraints:
    diet: Optional[str] = None
    exclude_ingredients: List[str] = None  # type: ignore[assignment]
    preferred_brand: Optional[str] = None
    max_items: Optional[int] = None
    preferred_store: Optional[str] = None


def _excluded(ingredient_name: str, excludes: Iterable[str]) -> bool:
    """
    Token/substr exclusion:
    - exclude="hành" should exclude "hành ngò"
    """
    ing_norm = normalize_vi(ingredient_name)
    ing_toks = set(tokenize_norm(ingredient_name))
    for ex in excludes:
        ex_norm = normalize_vi(ex)
        if not ex_norm:
            continue
        if ex_norm in ing_norm:
            return True
        ex_toks = set(tokenize_norm(ex))
        if ex_toks and ex_toks.issubset(ing_toks):
            return True
    return False


# ----------------------------
# Matching engine
# ----------------------------
@dataclass(frozen=True)
class MatchCandidate:
    product: Product
    score: float
    reason: str


class ProductMatcher:
    """
    Explainable ingredient->product matching.
    Signals:
      - name exact/contains/token overlap
      - stock preference (in-stock preferred but still keep OOS for suggestions)
      - price penalty (avoid too expensive dominating)
      - fallback: char-ngrams TFIDF similarity on product names
    """

    def __init__(self, products: List[Product]) -> None:
        self.products = products
        self._by_sku: Dict[str, Product] = {p.sku: p for p in products}
        self._norm_name: Dict[str, str] = {p.sku: normalize_vi(p.name) for p in products}
        self._tokens: Dict[str, List[str]] = {p.sku: tokenize_norm(p.name) for p in products}
        self._inv: Dict[str, List[str]] = {}
        for p in products:
            sku = p.sku
            for t in set(self._tokens[sku]):
                self._inv.setdefault(t, []).append(sku)

        # TFIDF fallback for suggestions (CPU)
        names_norm = [self._norm_name[p.sku] for p in products]
        self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        self._X = self._tfidf.fit_transform(names_norm) if products else None

    def candidates(self, ingredient_name: str, top_k: int = 10) -> List[MatchCandidate]:
        ing_norm = normalize_vi(ingredient_name)
        ing_toks = tokenize_norm(ingredient_name)
        if not ing_norm:
            return []

        sku_hits: List[str] = []
        for t in set(ing_toks):
            sku_hits.extend(self._inv.get(t, []))

        # substring fallback (fast)
        if not sku_hits:
            sku_hits = [p.sku for p in self.products if ing_norm in self._norm_name[p.sku]]

        # TFIDF fallback (robust)
        if not sku_hits and self._X is not None:
            sku_hits = self._tfidf_top_skus(ing_norm, k=min(50, len(self.products)))

        # dedupe
        sku_hits = list(dict.fromkeys(sku_hits))

        scored: List[MatchCandidate] = []
        for sku in sku_hits:
            p = self._by_sku[sku]
            p_norm = self._norm_name[sku]
            p_toks = self._tokens[sku]
            score, reason = self._score(ing_norm, ing_toks, p, p_norm, p_toks)
            scored.append(MatchCandidate(product=p, score=score, reason=reason))

        # prefer higher score, tie-break by cheaper final price
        scored.sort(key=lambda x: (-x.score, _final_price(x.product)))
        return scored[:top_k]

    def best(self, ingredient_name: str, constraints: CartConstraints, top_k: int = 10) -> Tuple[Optional[MatchCandidate], List[MatchCandidate]]:
        cands = self.candidates(ingredient_name, top_k=top_k)
        if not cands:
            return None, []

        pref_brand = normalize_vi(constraints.preferred_brand or "")
        boosted: List[MatchCandidate] = []
        for c in cands:
            if pref_brand and pref_brand not in normalize_vi(c.product.name):
                boosted.append(MatchCandidate(c.product, c.score - 0.25, c.reason + "+brand_miss"))
            else:
                boosted.append(c)

        # prefer in-stock candidate if exists
        in_stock = [c for c in boosted if c.product.stock > 0]
        in_stock.sort(key=lambda x: (-x.score, _final_price(x.product)))
        boosted.sort(key=lambda x: (-x.score, _final_price(x.product)))

        return (in_stock[0] if in_stock else boosted[0]), boosted

    def suggestions(self, ingredient_name: str, top_k: int = 3) -> List[str]:
        cands = self.candidates(ingredient_name, top_k=10)
        return [c.product.name for c in cands[:top_k]]

    def _score(self, ing_norm: str, ing_toks: List[str], p: Product, p_norm: str, p_toks: List[str]) -> Tuple[float, str]:
        # name similarity
        if ing_norm == p_norm:
            name_score, reason = 5.0, "exact"
        elif ing_norm and ing_norm in p_norm:
            name_score, reason = 4.0, "contains"
        else:
            inter = len(set(ing_toks) & set(p_toks))
            if inter == 0:
                # allow very low score; keep for TFIDF fallback and suggestions
                return -2.5, "weak_match"
            ratio = inter / max(1, len(set(ing_toks)))
            name_score = 2.0 + 2.0 * ratio
            reason = f"token_overlap_{inter}"

        # stock signal (do not discard OOS entirely)
        if p.stock <= 0:
            stock_adj = -1.5
            reason += "+out_of_stock"
        elif p.stock < 3:
            stock_adj = 0.2
            reason += "+low_stock"
        else:
            stock_adj = 0.5
            reason += "+in_stock"

        # price penalty
        fp = _final_price(p)
        price_penalty = min(1.5, fp / 200_000.0)
        score = name_score + stock_adj - price_penalty
        return score, reason

    def _tfidf_top_skus(self, ing_norm: str, k: int = 50) -> List[str]:
        if self._X is None:
            return []
        q = self._tfidf.transform([ing_norm])
        sims = (self._X @ q.T).toarray().ravel()  # cosine-like on normalized vectors
        idxs = np.argsort(-sims)[:k]
        skus = [self.products[int(i)].sku for i in idxs if sims[int(i)] > 0]
        return skus


# ----------------------------
# Cart planning
# ----------------------------
class CartPlanner:
    """Cart planning: recipe -> ingredient_plan -> cart -> pricing -> warnings/unmatched."""

    def __init__(self, recipe_repo: RecipeReadRepo, product_repo: ProductReadRepo) -> None:
        self.recipe_repo = recipe_repo
        self.product_repo = product_repo
        self.matcher = ProductMatcher(product_repo.all())

    def _find_recipe(self, recipe_id_or_title: str) -> Recipe:
        key = (recipe_id_or_title or "").strip()
        if not key:
            raise ValueError("recipe_id_or_title is required")

        by_id = self.recipe_repo.by_id(key)
        if by_id:
            return by_id

        k_norm = normalize_vi(key)
        k_toks = tokenize_norm(key)

        best: Optional[Recipe] = None
        best_score = -1e9
        for r in self.recipe_repo.all():
            t_norm = normalize_vi(r.title)
            if not t_norm:
                continue
            score = 0.0
            if k_norm and k_norm == t_norm:
                score += 5.0
            elif k_norm and k_norm in t_norm:
                score += 3.0
            score += 2.0 * _jaccard(k_toks, tokenize_norm(r.title))
            if score > best_score:
                best_score, best = score, r

        if not best or best_score < 1.0:
            raise LookupError(f"Recipe not found: {recipe_id_or_title}")
        return best

    def plan(
        self,
        recipe_id_or_title: str,
        user_servings: Optional[int] = None,
        budget: Optional[float] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cons = constraints or {}
        c = CartConstraints(
            diet=cons.get("diet"),
            exclude_ingredients=cons.get("exclude_ingredients") or [],
            preferred_brand=cons.get("preferred_brand"),
            max_items=cons.get("max_items"),
            preferred_store=cons.get("preferred_store"),
        )

        recipe = self._find_recipe(recipe_id_or_title)
        base_servings = int(recipe.servings or 1)
        usr_servings = int(user_servings or base_servings)
        factor = usr_servings / max(1, base_servings)

        ingredient_plan: List[Dict[str, Any]] = []
        unmatched: List[Dict[str, Any]] = []
        warnings: List[str] = []

        cart_items: List[Dict[str, Any]] = []
        subtotal = 0.0
        discount_total = 0.0

        planned_count = 0

        for ing in recipe.ingredients:
            ing_name = (ing.name or "").strip()
            if not ing_name:
                continue

            if _excluded(ing_name, c.exclude_ingredients or []):
                continue

            if c.max_items is not None and planned_count >= c.max_items:
                warnings.append(f"max_items reached; skipped ingredient: {ing_name}")
                continue

            qty_needed = _scale_qty(ing.qty, factor) if ing.qty is not None else None
            unit = (ing.unit or "").strip()
            ing_type = ing.type

            best, cands = self.matcher.best(ing_name, constraints=c, top_k=10)
            if not best:
                unmatched.append(
                    {
                        "name": ing_name,
                        "qty_needed": qty_needed,
                        "unit": unit or None,
                        "type": ing_type,
                        "suggestions": self.matcher.suggestions(ing_name),
                    }
                )
                continue

            p = best.product
            fp = _final_price(p)

            qty_to_buy, notes = self._calc_quantity_to_buy(qty_needed=qty_needed, unit=unit, product=p)
            qty_to_buy = int(max(1, qty_to_buy))

            # stock checks
            if p.stock <= 0:
                warnings.append(f"out_of_stock: {p.sku} ({p.name})")
                # if we selected OOS but have an in-stock alternative, pick it
                alt = self._pick_in_stock_alternative(cands, exclude_sku=p.sku)
                if alt:
                    p = alt.product
                    fp = _final_price(p)
                    best = alt
                    notes = (notes + " | " if notes else "") + "switched_to_in_stock_alternative"

            if p.stock < qty_to_buy:
                warnings.append(f"low_stock: {p.sku} needs {qty_to_buy} but stock={p.stock}")
                alt_skus = self._suggest_alternatives(cands, exclude_sku=p.sku, needed=qty_to_buy)
                if alt_skus:
                    notes = (notes + " | " if notes else "") + f"suggest_alternatives={','.join(alt_skus)}"

            line_subtotal = float(p.price) * qty_to_buy
            line_discount = (float(p.price) - fp) * qty_to_buy
            line_total = fp * qty_to_buy

            subtotal += line_subtotal
            discount_total += line_discount

            cart_items.append(
                {
                    "sku": p.sku,
                    "name": p.name,
                    "qty": qty_to_buy,
                    "unit_price": round(fp, 2),
                    "line_total": round(line_total, 2),
                }
            )

            ingredient_plan.append(
                {
                    "ingredient": {"name": ing_name, "qty_needed": qty_needed, "unit": unit or None, "type": ing_type},
                    "matched_product": {
                        "sku": p.sku,
                        "name": p.name,
                        "unit": p.unit,
                        "price": float(p.price),
                        "discount": float(p.discount),
                        "final_price": round(fp, 2),
                        "net_weight": float(p.net_weight),
                        "measure_unit": p.measure_unit,
                        "stock": int(p.stock),
                        "image": p.image,
                    },
                    "quantity_to_buy": qty_to_buy,
                    "match_reason": best.reason,
                    "notes": notes or "",
                }
            )

            planned_count += 1

        total = subtotal - discount_total
        if budget is not None and total > float(budget):
            warnings.append(f"budget_exceeded: total={round(total,2)} > budget={float(budget)}")

        return {
            "recipe": {"id": recipe.id, "title": recipe.title, "servings": base_servings, "user_servings": usr_servings},
            "ingredient_plan": ingredient_plan,
            "unmatched_ingredients": unmatched,
            "cart": {
                "items": cart_items,
                "pricing": {
                    "subtotal": round(subtotal, 2),
                    "discount_total": round(discount_total, 2),
                    "total": round(total, 2),
                },
            },
            "warnings": warnings,
        }

    def _calc_quantity_to_buy(self, qty_needed: Optional[float], unit: str, product: Product) -> Tuple[int, str]:
        if qty_needed is None or not unit:
            return 1, "estimated_qty_no_recipe_amount"

        needed_base, needed_unit = _to_base_amount(float(qty_needed), unit)
        pack_base, pack_unit = _product_net_to_base(float(product.net_weight), product.measure_unit)

        if needed_base is None or pack_base is None:
            return 1, "estimated_qty_unit_not_convertible_or_missing_net_weight"

        if needed_unit != pack_unit:
            return 1, "estimated_qty_unit_mismatch"

        packs = max(1, _ceil_div(needed_base, pack_base))
        if packs > 20:
            return 1, "estimated_qty_excessive_pack_count"
        return packs, ""

    def _suggest_alternatives(self, candidates: List[MatchCandidate], exclude_sku: str, needed: int) -> List[str]:
        alts: List[str] = []
        for c in candidates:
            if c.product.sku == exclude_sku:
                continue
            if c.product.stock >= needed:
                alts.append(c.product.sku)
            if len(alts) >= 3:
                break
        return alts

    def _pick_in_stock_alternative(self, candidates: List[MatchCandidate], exclude_sku: str) -> Optional[MatchCandidate]:
        for c in candidates:
            if c.product.sku == exclude_sku:
                continue
            if c.product.stock > 0:
                return c
        return None
