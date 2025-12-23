# smart_food_bot/src/application/cart_planner.py
from __future__ import annotations

import math
import re
import unicodedata
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.domain.entities import Recipe, Product, Ingredient
from src.domain.repositories import RecipeReadRepo, ProductReadRepo

log = logging.getLogger("app.cart_planner")

# ----------------------------
# Normalization & tokenization
# ----------------------------
_NON_ALNUM = re.compile(r"[^a-z0-9\s]+")

def normalize_vi(text: str) -> str:
    """
    WHY: Normalize Vietnamese strings for robust matching.
    - lowercase
    - remove diacritics
    - remove special chars
    - collapse spaces
    """
    text = (text or "").strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")  # strip accents
    text = _NON_ALNUM.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_norm(text: str) -> List[str]:
    n = normalize_vi(text)
    return [t for t in n.split() if t]


# ----------------------------
# Unit conversion
# ----------------------------
_MASS_UNITS_TO_G = {
    "g": 1.0,
    "gram": 1.0,
    "kg": 1000.0,
    "kilogram": 1000.0,
    "lạng": 100.0,
    "lang": 100.0,
}
_VOL_UNITS_TO_ML = {
    "ml": 1.0,
    "l": 1000.0,
    "lit": 1000.0,
    "liter": 1000.0,
}

def _to_base_amount(qty: float, unit: str) -> Tuple[Optional[float], Optional[str]]:
    """
    Convert (qty, unit) -> (amount_in_base, base_unit) where base_unit in {"g","ml"}.
    Returns (None, None) if cannot convert.
    """
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


# ----------------------------
# Constraints
# ----------------------------
@dataclass(frozen=True)
class CartConstraints:
    diet: Optional[str] = None
    exclude_ingredients: Optional[List[str]] = None
    preferred_brand: Optional[str] = None
    max_items: Optional[int] = None
    # placeholder for store/channel filtering if needed
    preferred_store: Optional[str] = None


# ----------------------------
# Matching engine (in-memory)
# ----------------------------
@dataclass(frozen=True)
class MatchCandidate:
    product: Product
    score: float
    reason: str

class ProductMatcher:
    """
    WHY: Fast, deterministic, explainable ingredient->product matching.
    Uses token inverted index over normalized product names.
    """
    def __init__(self, products: List[Product]) -> None:
        self.products = products
        self._norm_name: Dict[str, str] = {}
        self._tokens: Dict[str, List[str]] = {}
        self._inv: Dict[str, List[str]] = {}  # token -> [sku]

        for p in products:
            n = normalize_vi(p.name)
            self._norm_name[p.sku] = n
            toks = tokenize_norm(p.name)
            self._tokens[p.sku] = toks
            for t in set(toks):
                self._inv.setdefault(t, []).append(p.sku)

        self._by_sku = {p.sku: p for p in products}

    def _final_price(self, p: Product) -> float:
        return max(0.0, float(p.price) * (1.0 - float(p.discount) / 100.0))

    def candidates(self, ingredient_name: str, top_k: int = 10) -> List[MatchCandidate]:
        ing_norm = normalize_vi(ingredient_name)
        ing_toks = tokenize_norm(ingredient_name)
        if not ing_norm:
            return []

        # retrieve by token overlap
        sku_hits: List[str] = []
        for t in set(ing_toks):
            sku_hits.extend(self._inv.get(t, []))
        if not sku_hits:
            # fallback: scan small subset by substring on normalized name
            sku_hits = [p.sku for p in self.products if ing_norm and ing_norm in self._norm_name[p.sku]]

        # dedupe
        sku_hits = list(dict.fromkeys(sku_hits))
        scored: List[MatchCandidate] = []
        for sku in sku_hits:
            p = self._by_sku[sku]
            p_norm = self._norm_name[sku]
            p_toks = self._tokens[sku]
            score, reason = self._score(ing_norm, ing_toks, p, p_norm, p_toks)
            if score > -1e8:
                scored.append(MatchCandidate(product=p, score=score, reason=reason))

        scored.sort(key=lambda x: (-x.score, self._final_price(x.product)))
        return scored[:top_k]

    def best(
        self,
        ingredient_name: str,
        constraints: CartConstraints,
        top_k: int = 10,
    ) -> Tuple[Optional[MatchCandidate], List[MatchCandidate]]:
        cands = self.candidates(ingredient_name, top_k=top_k)
        if not cands:
            return None, []

        pref_brand = normalize_vi(constraints.preferred_brand or "")
        filtered: List[MatchCandidate] = []
        for c in cands:
            p = c.product
            if pref_brand and pref_brand not in normalize_vi(p.name):
                # keep but lower priority
                filtered.append(MatchCandidate(p, c.score - 0.25, c.reason + "+brand_miss"))
            else:
                filtered.append(c)

        filtered.sort(key=lambda x: (-x.score, self._final_price(x.product)))
        best = filtered[0] if filtered else None
        return best, filtered

    def _score(
        self,
        ing_norm: str,
        ing_toks: List[str],
        p: Product,
        p_norm: str,
        p_toks: List[str],
    ) -> Tuple[float, str]:
        if not p_norm:
            return -1e9, "invalid_name"

        # Name similarity
        if ing_norm == p_norm:
            name_score, reason = 5.0, "exact"
        elif ing_norm and ing_norm in p_norm:
            name_score, reason = 4.0, "contains"
        else:
            inter = len(set(ing_toks) & set(p_toks))
            if inter == 0:
                return -1e9, "no_overlap"
            ratio = inter / max(1, len(set(ing_toks)))
            name_score = 2.0 + 2.0 * ratio  # [2..4]
            reason = f"token_overlap_{inter}"

        # Stock signal
        if p.stock <= 0:
            return -1e9, reason + "+out_of_stock"
        stock_bonus = 0.5 if p.stock >= 3 else 0.2

        # Price signal (prefer cheaper final price but avoid dominating)
        final_price = max(0.0, float(p.price) * (1.0 - float(p.discount) / 100.0))
        price_penalty = min(1.5, final_price / 200_000.0)  # normalize
        score = name_score + stock_bonus - price_penalty
        return score, reason + "+in_stock"


# ----------------------------
# Cart planning
# ----------------------------
def _scale_qty(qty: Any, factor: float) -> Optional[float]:
    try:
        return float(qty) * factor
    except Exception:
        return None

def _ceil_div(a: float, b: float) -> int:
    return int(math.ceil(a / b))

class CartPlanner:
    """
    Input:
      - recipe_id or recipe_title
      - user_servings (default=recipe.servings)
      - budget (optional)
      - constraints (optional)

    Output: dict exactly matching required JSON contract.
    """
    def __init__(self, recipe_repo: RecipeReadRepo, product_repo: ProductReadRepo) -> None:
        self.recipe_repo = recipe_repo
        self.product_repo = product_repo
        self.matcher = ProductMatcher(product_repo.all())

    def _find_recipe(self, recipe_id_or_title: str) -> Recipe:
        key = (recipe_id_or_title or "").strip()
        if not key:
            raise ValueError("recipe_id_or_title is required")

        # 1) try id
        by_id = self.recipe_repo.by_id(key)
        if by_id:
            return by_id

        # 2) fallback by normalized title contains
        k_norm = normalize_vi(key)
        best: Optional[Recipe] = None
        for r in self.recipe_repo.all():
            if k_norm and k_norm in normalize_vi(r.title):
                best = r
                break
        if not best:
            raise LookupError(f"Recipe not found: {recipe_id_or_title}")
        return best

    def plan(
        self,
        recipe_id_or_title: str,
        user_servings: Optional[int] = None,
        budget: Optional[float] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        c = CartConstraints(
            diet=(constraints or {}).get("diet"),
            exclude_ingredients=(constraints or {}).get("exclude_ingredients") or [],
            preferred_brand=(constraints or {}).get("preferred_brand"),
            max_items=(constraints or {}).get("max_items"),
            preferred_store=(constraints or {}).get("preferred_store"),
        )

        recipe = self._find_recipe(recipe_id_or_title)
        base_servings = int(recipe.servings or 1)
        usr_servings = int(user_servings or base_servings)
        factor = usr_servings / max(1, base_servings)

        exclude_norm = {normalize_vi(x) for x in (c.exclude_ingredients or []) if x}

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
            if normalize_vi(ing_name) in exclude_norm:
                continue

            if c.max_items is not None and planned_count >= c.max_items:
                warnings.append(f"max_items reached; skipped ingredient: {ing_name}")
                continue

            qty_needed = _scale_qty(ing.qty, factor) if ing.qty is not None else None
            unit = (ing.unit or "").strip()
            ing_type = ing.type

            # ---- product matching
            best, cands = self.matcher.best(ing_name, constraints=c, top_k=10)
            if not best:
                unmatched.append({
                    "name": ing_name,
                    "qty_needed": qty_needed,
                    "unit": unit or None,
                    "type": ing_type,
                    "suggestions": [],
                })
                continue

            p = best.product
            final_price = max(0.0, float(p.price) * (1.0 - float(p.discount) / 100.0))

            # ---- quantity_to_buy calculation
            qty_to_buy, notes = self._calc_quantity_to_buy(
                qty_needed=qty_needed,
                unit=unit,
                product=p,
                user_servings=usr_servings,
                base_servings=base_servings,
            )

            # stock check + alternatives
            if p.stock < qty_to_buy:
                warnings.append(
                    f"out_of_stock_risk: {p.sku} needs {qty_to_buy} but stock={p.stock}"
                )
                alt = self._suggest_alternatives(cands, exclude_sku=p.sku, needed=qty_to_buy)
                if alt:
                    notes = (notes + " | " if notes else "") + f"suggest_alternatives={','.join(alt)}"
                # clamp purchasable qty (still return requested qty_to_buy per requirement? keep request and warn)
                # Keep qty_to_buy as requested; user can decide.

            # ---- build cart/pricing
            line_subtotal = float(p.price) * qty_to_buy
            line_discount = (float(p.price) - final_price) * qty_to_buy
            line_total = final_price * qty_to_buy

            subtotal += line_subtotal
            discount_total += line_discount
            cart_items.append({
                "sku": p.sku,
                "name": p.name,
                "qty": qty_to_buy,
                "unit_price": round(final_price, 2),
                "line_total": round(line_total, 2),
            })

            ingredient_plan.append({
                "ingredient": {
                    "name": ing_name,
                    "qty_needed": qty_needed,
                    "unit": unit or None,
                    "type": ing_type,
                },
                "matched_product": {
                    "sku": p.sku,
                    "name": p.name,
                    "unit": p.unit,
                    "price": float(p.price),
                    "discount": float(p.discount),
                    "final_price": round(final_price, 2),
                    "net_weight": float(p.net_weight),
                    "measure_unit": p.measure_unit,
                    "stock": int(p.stock),
                },
                "quantity_to_buy": int(qty_to_buy),
                "match_reason": best.reason,
                "notes": notes or "",
            })
            planned_count += 1

        total = subtotal - discount_total
        if budget is not None and total > float(budget):
            warnings.append(f"budget_exceeded: total={round(total,2)} > budget={float(budget)}")

        # Suggestions for unmatched
        for u in unmatched:
            u["suggestions"] = self._unmatched_suggestions(u["name"])

        return {
            "recipe": {
                "id": recipe.id,
                "title": recipe.title,
                "servings": base_servings,
                "user_servings": usr_servings,
            },
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

    def _calc_quantity_to_buy(
        self,
        qty_needed: Optional[float],
        unit: str,
        product: Product,
        user_servings: int,
        base_servings: int,
    ) -> Tuple[int, str]:
        """
        Rule:
        - If product has net_weight+measure_unit and ingredient qty+unit convertible -> ceil(needed/base_per_pack)
        - Else qty_to_buy=1 and notes includes estimation.
        """
        if qty_needed is None or not unit:
            return 1, "estimated_qty_no_recipe_amount"

        needed_base, needed_unit = _to_base_amount(float(qty_needed), unit)
        pack_base, pack_unit = _product_net_to_base(float(product.net_weight), product.measure_unit)

        if needed_base is None or pack_base is None:
            return 1, "estimated_qty_unit_not_convertible_or_missing_net_weight"

        # unit mismatch (g vs ml) -> estimate
        if needed_unit != pack_unit:
            return 1, "estimated_qty_unit_mismatch"

        packs = _ceil_div(needed_base, pack_base)
        packs = max(1, packs)
        # avoid too much dư (cap excessive packs if weird data)
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

    def _unmatched_suggestions(self, ingredient_name: str) -> List[str]:
        ing_toks = tokenize_norm(ingredient_name)
        if not ing_toks:
            return []
        # suggest top products that share tokens (cheap & in stock)
        cands = self.matcher.candidates(ingredient_name, top_k=5)
        return [c.product.name for c in cands[:3]]