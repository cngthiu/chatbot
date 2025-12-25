# =========================
# FILE: smart_food_bot/src/application/response_templates.py
# =========================
from __future__ import annotations

import random
from typing import Optional

def _pick(options: list[str]) -> str:
    return random.choice(options)

def greet_reply(name: Optional[str] = None) -> str:
    base = [
        "Xin chào! Mình giúp bạn tìm món và lên giỏ nguyên liệu nhé.",
        "Chào bạn 👋 Bạn muốn nấu món gì hôm nay?",
        "Xin chào! Bạn mô tả món bạn muốn nấu, mình tìm công thức ngay.",
    ]
    return _pick(base)

def thanks_reply() -> str:
    return _pick([
        "Dạ cảm ơn bạn! Bạn cần mình hỗ trợ thêm gì nữa không?",
        "Cảm ơn bạn nhé 😊 Bạn muốn mình lên giỏ nguyên liệu cho món nào?",
    ])

def bye_reply() -> str:
    return _pick([
        "Tạm biệt! Khi nào cần gợi ý món ngon cứ quay lại nhé 👋",
        "Chào bạn! Chúc bạn nấu ăn ngon miệng 😊",
    ])

def apology_reply() -> str:
    return _pick([
        "Xin lỗi bạn nha. Bạn nói lại giúp mình cụ thể hơn được không?",
        "Mình xin lỗi vì chưa hiểu ý bạn. Bạn muốn tìm món hay lên giỏ nguyên liệu?",
    ])

def ask_clarify_reply() -> str:
    return _pick([
        "Mình chưa chắc bạn muốn tìm món hay lên giỏ. Bạn nói rõ hơn giúp mình nhé.",
        "Bạn muốn nấu món gì (tên món) hay muốn mua nguyên liệu (liệt kê nguyên liệu)?",
    ])

def prompt_pick_recipe_reply() -> str:
    return _pick([
        "Mình tìm được vài món phù hợp. Bạn chọn món số mấy để mình lên giỏ nguyên liệu?",
        "Bạn chọn giúp mình 1 món trong danh sách (ví dụ: 'chọn món số 2') nhé.",
    ])

def prompt_servings_reply() -> str:
    return _pick([
        "Bạn muốn bao nhiêu khẩu phần để mình tính lại nguyên liệu?",
        "Bạn định nấu cho mấy người để mình lên giỏ đúng số lượng?",
    ])

def cart_done_reply() -> str:
    return _pick([
        "Mình đã lên giỏ nguyên liệu rồi. Bạn muốn chỉnh khẩu phần hay bỏ nguyên liệu nào không?",
        "Giỏ nguyên liệu đã sẵn sàng. Bạn muốn thay đổi gì trước khi đặt mua không?",
    ])
