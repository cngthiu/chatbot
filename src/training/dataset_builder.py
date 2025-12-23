# src/training/dataset_builder.py
import json
import random
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

# Định nghĩa các intent
INTENTS = [
    "search_recipe",
    "ask_recipe_detail",
    "refine_search",
    "add_ingredients_to_cart",
    "ask_price_estimate",
    "fallback"
]
SLOT_TYPES = ["DISH", "INGREDIENT", "QUANTITY", "UNIT", "TASTE", "EXCLUDE"]
# Load data from files (assuming files are in data/raw or adjust path)
def load_data():
    with open('/media/congthieu/ubuntu_data/LTTM/MM/serverAI/data/recipes/recipies.json', 'r', encoding='utf-8') as f:
        recipes = json.load(f)
    with open('/media/congthieu/ubuntu_data/LTTM/MM/serverAI/data/products_seed.json', 'r', encoding='utf-8') as f:
        products = json.load(f)
    return recipes, products

recipes, products = load_data()

# Extract useful lists
DISHES = [recipe['title'] for recipe in recipes]

# Ingredients from recipes
recipe_ingredients = set()
for recipe in recipes:
    for ing in recipe['ingredients']:
        recipe_ingredients.add(ing['name'])
RECIPE_INGREDIENTS = list(recipe_ingredients)

# Trích xuất nguyên liệu/sản phẩm từ products (tên sản phẩm thường là nguyên liệu)
product_names = set()
for prod in products:
    name = prod['name']
    # Loại bỏ phần quảng cáo như "giá tốt tại Bách hoá XANH", "hộp", "túi"...
    clean_name = name.split(' giá tốt')[0].split(' tại ')[0].split(' hộp ')[0] \
                   .split(' túi ')[0].split(' khay ')[0].split(' chai ')[0].strip().lower()
    if clean_name:
        product_names.add(clean_name)
PRODUCT_INGREDIENTS = list(product_names)

# Gộp tất cả nguyên liệu
INGREDIENTS = sorted(list(set(RECIPE_INGREDIENTS + PRODUCT_INGREDIENTS)))

QUANTITIES = ["1", "2", "3", "4", "5", "một", "hai", "ba", "nửa", "một nửa", "một ít", "nửa kg", "hai lạng"]

UNITS = ["kg", "lạng", "gam", "cái", "bó", "quả", "tép", "muỗng", "thìa", "lít", "ml", "gói", "hộp", "chai"]

TASTES = ["chua cay", "ngọt", "mặn", "chua ngọt", "giòn", "thơm", "đậm đà", "ngọt mát", "cay nồng", "thanh đạm"]

EXCLUDES = ["không hành", "không ngò", "không tỏi", "không cay", "không giá", "ít dầu", "không đường", "không mắm"]

# Templates for each intent (with placeholders for slots)
TEMPLATES = {
    "search_recipe": [
        "Tìm công thức làm {DISH}",
        "Cách nấu {DISH} ngon",
        "Cho mình công thức {DISH} với",
        "Hướng dẫn làm {DISH}",
        "Recipe {DISH} đi",
        "Làm {DISH} thế nào",
        "Công thức món {DISH}",
        "Cách chế biến {DISH}",
        "Món {DISH} làm ra sao"
    ],
    "ask_recipe_detail": [
        "Nguyên liệu để nấu {DISH} gồm những gì",
        "Chi tiết cách làm {DISH}",
        "Các bước nấu {DISH} ra sao",
        "Làm {DISH} cần chuẩn bị gì",
        "Hướng dẫn chi tiết {DISH}",
        "Thành phần của {DISH}",
        "Thời gian nấu {DISH} bao lâu",
        "Món {DISH} ăn với gì ngon"
    ],
    "refine_search": [
        "{DISH} {TASTE}",
        "{DISH} {EXCLUDE}",
        "{DISH} kiểu miền Bắc",
        "Công thức {DISH} với {INGREDIENT}",
        "Làm {DISH} mà {EXCLUDE}",
        "{DISH} thêm {INGREDIENT}",
        "{DISH} phiên bản {TASTE}",
        "Tìm {DISH} không dùng {EXCLUDE}",
        "{DISH} cho người ăn kiêng"
    ],
    "add_ingredients_to_cart": [
        "Thêm {QUANTITY} {UNIT} {INGREDIENT} vào giỏ hàng",
        "Mua {QUANTITY} {UNIT} {INGREDIENT}",
        "Cho {QUANTITY} {UNIT} {INGREDIENT} vào giỏ",
        "Thêm {INGREDIENT} số lượng {QUANTITY} {UNIT}",
        "Giỏ hàng thêm {QUANTITY} {UNIT} {INGREDIENT}",
        "Mua thêm {QUANTITY} {UNIT} {INGREDIENT} cho {DISH}",
        "Thêm nguyên liệu {INGREDIENT} {QUANTITY} {UNIT}"
    ],
    "ask_price_estimate": [
        "Giá khoảng bao nhiêu cho {QUANTITY} {UNIT} {INGREDIENT}",
        "Ước tính giá {QUANTITY} {UNIT} {INGREDIENT}",
        "Bao nhiêu tiền một {UNIT} {INGREDIENT}",
        "Giá {INGREDIENT} hiện tại thế nào",
        "Chi phí nguyên liệu cho {DISH} khoảng bao nhiêu",
        "Ước lượng giá mua {QUANTITY} {INGREDIENT}",
        "Giá của {INGREDIENT} là bao nhiêu"
    ],
    "fallback": [
        "Xin chào",
        "Chào bạn",
        "Cảm ơn",
        "Bye",
        "Hẹn gặp lại",
        "Ok",
        "Không hiểu",
        "Giúp mình với",
        "Hello",
        "Thanks",
        "What?",
        "Help"
    ],
}

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_sample(intent: str) -> Dict:
    template = random.choice(TEMPLATES[intent])
    text = template

    # Chọn entity ngẫu nhiên (có xác suất bỏ qua để đa dạng)
    dish = random.choice(DISHES) if "{DISH}" in template and random.random() > 0.1 else ""
    ingredient = random.choice(INGREDIENTS) if "{INGREDIENT}" in template and random.random() > 0.1 else ""
    quantity = random.choice(QUANTITIES) if "{QUANTITY}" in template and random.random() > 0.1 else ""
    unit = random.choice(UNITS) if "{UNIT}" in template and random.random() > 0.1 else ""
    taste = random.choice(TASTES) if "{TASTE}" in template and random.random() > 0.1 else ""
    exclude = random.choice(EXCLUDES) if "{EXCLUDE}" in template and random.random() > 0.1 else ""

    entity_map = {
        "DISH": (dish, "DISH"),
        "INGREDIENT": (ingredient, "INGREDIENT"),
        "QUANTITY": (quantity, "QUANTITY"),
        "UNIT": (unit, "UNIT"),
        "TASTE": (taste, "TASTE"),
        "EXCLUDE": (exclude, "EXCLUDE"),
    }

    spans = []  # Lưu (start, end, type)
    for placeholder, (value, etype) in entity_map.items():
        if value and f"{{{placeholder}}}" in text:
            text = text.replace(f"{{{placeholder}}}", value, 1)
            value_tokens = value.split()
            temp_tokens = text.split()
            # Tìm vị trí chính xác
            for i in range(len(temp_tokens) - len(value_tokens) + 1):
                if temp_tokens[i:i+len(value_tokens)] == value_tokens:
                    spans.append((i, i + len(value_tokens) - 1, etype))
                    break

    tokens = text.split()
    bio_labels = ["O"] * len(tokens)
    for start, end, etype in spans:
        bio_labels[start] = f"B-{etype}"
        for i in range(start + 1, end + 1):
            bio_labels[i] = f"I-{etype}"

    return {
        "text": text.strip(),
        "tokens": tokens,
        "intent": intent,
        "bio_labels": bio_labels
    }

def main():
    samples_per_intent = 400  # 400 mẫu mỗi intent → tổng 2400 mẫu
    all_data = defaultdict(list)

    print("Đang sinh dữ liệu synthetic từ recipes và products thực tế...")
    for intent in INTENTS:
        print(f"  Sinh {samples_per_intent} mẫu cho intent: {intent}")
        for _ in range(samples_per_intent):
            sample = generate_sample(intent)
            if sample["text"].strip():  # Loại bỏ mẫu rỗng
                all_data[intent].append(sample)

    # Chia train / valid / test (80% - 10% - 10%) theo từng intent để cân bằng
    train_data, valid_data, test_data = [], [], []

    for intent, samples in all_data.items():
        random.shuffle(samples)
        n = len(samples)
        train_data.extend(samples[:int(0.8 * n)])
        valid_data.extend(samples[int(0.8 * n):int(0.9 * n)])
        test_data.extend(samples[int(0.9 * n):])

    # Lưu từng tập
    splits = {
        "train.json": train_data,
        "valid.json": valid_data,
        "test.json": test_data,
    }

    for filename, data in splits.items():
        path = DATA_DIR / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Đã lưu {len(data)} mẫu vào {path}")

    print("\nHoàn tất! Đã tạo 3 tập dữ liệu riêng biệt:")
    print(f"  - Train: {len(train_data)} mẫu")
    print(f"  - Valid: {len(valid_data)} mẫu")
    print(f"  - Test:  {len(test_data)} mẫu")

if __name__ == "__main__":
    main()  