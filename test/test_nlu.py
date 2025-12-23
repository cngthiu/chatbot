# smart_food_bot/test.py
"""
Quick API smoke tests for:
- POST /chat
- POST /plan_cart

Usage:
  1) Start API:
       uvicorn main:app --host 0.0.0.0 --port 8000
  2) Run tests:
       python test.py --base-url http://127.0.0.1:8000

Notes:
  - Exits with non-zero code if any test fails.
  - Prints pretty JSON responses for debugging.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, Tuple

import requests


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _post_json(base_url: str, path: str, payload: Dict[str, Any], timeout: float = 30.0) -> Tuple[int, Dict[str, Any]]:
    url = base_url.rstrip("/") + path
    r = requests.post(url, json=payload, timeout=timeout)
    try:
        data = r.json()
    except Exception:
        data = {"_raw": r.text}
    return r.status_code, data


def wait_ready(base_url: str, retries: int = 20, sleep_s: float = 0.5) -> None:
    """
    Best-effort: if you don't have /healthz, we just try hitting /chat with a tiny payload.
    """
    last_err = None
    for _ in range(retries):
        try:
            code, _ = _post_json(base_url, "/chat", {"text": "xin chào"}, timeout=5.0)
            if code in (200, 400, 422):
                return
        except Exception as e:
            last_err = e
        time.sleep(sleep_s)
    raise RuntimeError(f"API not reachable at {base_url}. Last error: {last_err}")


def test_chat(base_url: str) -> None:
    payload = {
        "text": "Tìm công thức canh bí đỏ thịt bằm thanh nhẹ"
    }
    code, data = _post_json(base_url, "/chat", payload)

    _assert(code == 200, f"/chat expected 200, got {code}: {_pretty(data)}")
    _assert("nlu" in data, f"/chat missing 'nlu': {_pretty(data)}")
    _assert("intent" in data["nlu"], f"/chat missing nlu.intent: {_pretty(data)}")
    # recipes may or may not exist depending on intent routing, but for search it should.
    _assert("recipes" in data and data["recipes"] is not None, f"/chat expected recipes: {_pretty(data)}")

    print("\n✅ /chat OK")
    print(_pretty(data))


def test_plan_cart(base_url: str, recipe_id: str) -> None:
    payload = {
        "recipe_id": recipe_id,
        "user_servings": 4,
        "budget": None,
        "constraints": {
            "exclude_ingredients": [],
            "preferred_brand": None,
            "max_items": 30
        }
    }
    code, data = _post_json(base_url, "/plan_cart", payload)

    _assert(code == 200, f"/plan_cart expected 200, got {code}: {_pretty(data)}")
    _assert("recipe" in data, f"/plan_cart missing 'recipe': {_pretty(data)}")
    _assert("ingredient_plan" in data, f"/plan_cart missing 'ingredient_plan': {_pretty(data)}")
    _assert("cart" in data and "pricing" in data["cart"], f"/plan_cart missing cart.pricing: {_pretty(data)}")
    _assert("warnings" in data, f"/plan_cart missing 'warnings': {_pretty(data)}")

    pricing = data["cart"]["pricing"]
    for k in ("subtotal", "discount_total", "total"):
        _assert(k in pricing, f"/plan_cart missing pricing.{k}: {_pretty(data)}")

    print("\n✅ /plan_cart OK")
    print(_pretty(data))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000", help="FastAPI base URL")
    parser.add_argument("--recipe-id", default="canh-bi-do-thit-bam-20p", help="Recipe id to test /plan_cart")
    args = parser.parse_args()

    try:
        wait_ready(args.base_url)
        test_chat(args.base_url)
        test_plan_cart(args.base_url, args.recipe_id)
        print("\n🎉 All tests passed.")
        return 0
    except Exception as e:
        print("\n❌ Test failed:", str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
