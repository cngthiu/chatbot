# smart_food_bot/src/application/response_composer.py
from __future__ import annotations

import random
from typing import List, Optional

from src.application import response_templates as rt


def _pick(xs: List[str]) -> str:
    return random.choice(xs) if xs else ""


class ResponseComposer:
    """
    WHY: Make replies less "hardcoded" by composing from facts + light variations.
    This keeps deterministic flow, but makes wording less repetitive.
    """

    def greet(self) -> str:
        return rt.greet_reply()

    def thanks(self) -> str:
        return rt.thanks_reply()

    def bye(self) -> str:
        return rt.bye_reply()

    def apology(self) -> str:
        return rt.apology_reply()

    def clarify(self) -> str:
        return rt.ask_clarify_reply()

    def prompt_pick_recipe(self, results_count: int) -> str:
        if results_count <= 0:
            return _pick(
                [
                    "Mình chưa thấy món phù hợp ngay. Bạn nói rõ tên món/khẩu vị hoặc nguyên liệu giúp mình nhé.",
                    "Bạn mô tả rõ hơn giúp mình: muốn món gì, vị gì, có nguyên liệu nào bắt buộc không?",
                ]
            )
        head = _pick(
            [
                f"Mình tìm được {results_count} món phù hợp.",
                f"Có {results_count} gợi ý cho bạn đây.",
                "Mình lọc được vài món hợp yêu cầu của bạn.",
            ]
        )
        tail = _pick(
            [
                "Bạn chọn món số mấy để mình đưa công thức hoặc lên giỏ nguyên liệu?",
                "Bạn chọn giúp mình 1 món trong danh sách (ví dụ: 'chọn món số 2') nhé.",
            ]
        )
        return f"{head} {tail}".strip()

    def prompt_servings(self, recipe_title: Optional[str] = None) -> str:
        if recipe_title:
            return _pick(
                [
                    f"Bạn muốn nấu {recipe_title} cho mấy khẩu phần để mình tính nguyên liệu?",
                    f"Bạn định nấu {recipe_title} cho mấy người để mình lên giỏ đúng số lượng?",
                ]
            )
        return rt.prompt_servings_reply()

    def recipe_detail_intro(self, recipe_title: str) -> str:
        recipe_title = (recipe_title or "món này").strip()
        return _pick(
            [
                f"Dạ đây là công thức món {recipe_title}. Bạn muốn mình lên giỏ nguyên liệu luôn không? (gõ: 'lên giỏ')",
                f"Mình gửi bạn cách làm món {recipe_title} nhé. Muốn mình tạo giỏ nguyên liệu luôn không?",
            ]
        )

    def cart_done(self, cart_items: int, warnings: Optional[List[str]] = None) -> str:
        base = _pick(
            [
                f"Mình đã tạo giỏ nguyên liệu ({cart_items} sản phẩm).",
                f"Giỏ nguyên liệu đã sẵn sàng ({cart_items} món).",
            ]
        )
        if warnings:
            return f"{base} Lưu ý: {warnings[0]}"
        return f"{base} Bạn có thể bỏ chọn hoặc chỉnh số lượng trong danh sách nguyên liệu."

    def price_estimate_intro(self) -> str:
        return _pick(
            [
                "Mình ước tính chi phí như sau (có thể chênh lệch tuỳ khuyến mãi).",
                "Dưới đây là ước tính chi phí. Bạn muốn tối ưu theo ngân sách không?",
            ]
        )

    def fallback(self) -> str:
        return _pick(
            [
                "Bạn muốn nấu món gì? Ví dụ: 'canh bí đỏ', 'thịt kho', 'phở bò'...",
                "Bạn mô tả món/khẩu vị, mình gợi ý công thức nhé.",
            ]
        )
