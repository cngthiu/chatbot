# Smart Food Bot (VN) — Vietnamese Food Recommendation & Shopping Chatbot

PhoBERT-powered NLU, Hybrid Search (BM25 + Faiss TF-IDF), and a FastAPI backend. Optimized for GTX 1650 Max-Q (4GB VRAM) with **AMP FP16** training/inference and **faiss-cpu** for vector search. Clean Architecture: Domain ↔ Application ↔ Infrastructure ↔ Interfaces (API).

---

## 🚀 Giới thiệu

**Smart Food Bot** là chatbot tư vấn món Việt & mua sắm nguyên liệu:

- **NLU**: PhoBERT (`vinai/phobert-base`) fine-tuned cho **Intent** + **Slot (BIO)**.
- **Search**: **Hybrid Search** = BM25 (từ khóa) + TF-IDF + **Faiss L2** (CPU).
- **Template-based Response (Phản hồi theo mẫu).**
- **Backend**: **FastAPI** + Pydantic, cấu trúc **Clean Architecture** theo SOLID.
- **Dữ liệu**: **MongoDB** (collections: `recipes`, `products`).
- **Hiệu năng**: GTX 1650 Max-Q (4GB VRAM)
  - **AMP FP16** (`torch.cuda.amp`)
  - Batch nhỏ (8–16)
  - **faiss-cpu** giữ VRAM cho model

### NLU Domain

- **Intents**: `search_recipe`, `ask_recipe_detail`, `refine_search`, `add_ingredients_to_cart`, `ask_price_estimate`, `fallback`
- **Slots (BIO)**: `B/I-DISH`, `B/I-INGREDIENT`, `B-QUANTITY`, `B-UNIT`, `B-TASTE`, `B-EXCLUDE`, `O`

---

## 🧰 Cài đặt (Installation)

### 1) Yêu cầu hệ thống

- Python **3.9+**
- CUDA (khuyến nghị, nếu dùng GPU)
- MongoDB (local hoặc hosted)

### 2) Clone & tạo môi trường

```bash
git clone https://github.com/cngthiu/smart_food_bot.git
cd smart_food_bot
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Cấu hình biến môi trường

Tạo file `.env` (hoặc export trực tiếp):

```bash
export MONGO_URI="mongodb://localhost:27017"
export MONGO_DB="smart_food"
export MONGO_RECIPES_COL="recipes"
export MONGO_PRODUCTS_COL="products"

# NLU / Training
export MODEL_NAME="vinai/phobert-base"
export BATCH_SIZE="8"        # 1650 Max-Q
export MAX_LEN="128"
export EPOCHS="3"
export LR="3e-5"
export SEED="42"
```

> Lưu ý: Bạn đã seed dữ liệu vào MongoDB (`recipes`, `products`). Ứng dụng đọc trực tiếp từ đó, **không dùng mock**.

---

## ▶️ Cách sử dụng (Usage)

### 1) Train NLU (tùy chọn – nếu bạn muốn fine-tune)

Sinh synthetic dataset và train nhanh (giữ batch nhỏ, AMP FP16):

```bash
# Tạo dữ liệu huấn luyện mẫu (template-based)
python -m src.training.dataset_builder

# Train PhoBERT joint intent+slot (AMP FP16, faiss-cpu, batch nhỏ)
python -m src.training.trainer
```

Artifacts sẽ được lưu tại `models/phobert_joint_nlu/`.

### 2) Khởi chạy API

```bash
python -m src.main
# hoặc
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

### 3) Gọi API `/chat`

**Request**

```bash
curl -X POST http://localhost:8000/chat   -H "Content-Type: application/json"   -d '{"text":"Tìm công thức canh bí đỏ thịt bằm thanh nhẹ"}'
```

**Response (ví dụ)**

```json
{
  "nlu": {
    "intent": "search_recipe",
    "intent_confidence": 0.97,
    "slots": {
      "DISH": ["canh bí đỏ", "thịt bằm"],
      "TASTE": ["thanh nhẹ"]
    }
  },
  "recipes": [
    {
      "id": "canh-bi-do-thit-bam-20p",
      "title": "Canh bí đỏ thịt bằm",
      "summary": "Món canh ngọt mát, bổ dưỡng, bí đỏ dẻo bùi kết hợp với thịt bằm ngọt nước.",
      "ingredients": [
        "bí đỏ",
        "thịt heo xay",
        "hành ngò",
        "hạt nêm knor",
        "tiêu đen"
      ],
      "cook_time": 20,
      "servings": 4,
      "image": "https://...",
      "score": 0.82
    }
  ]
}
```

---

## 🧱 Cấu trúc dự án

```
smart_food_bot/
├── data/
│   ├── raw/                  # Synthetic dataset (JSON) do dataset_builder tạo
│   └── processed/            # vocab/mapping (intent2id, slot_label2id)
├── models/
│   └── phobert_joint_nlu/    # Model đã train (pytorch_model.bin, tokenizer, mappings)
├── src/
│   ├── core/
│   │   └── config.py         # Config (AMP/Batch/Mongo/Paths), logging
│   ├── domain/
│   │   ├── entities.py       # Recipe, Product, Ingredient (dataclass)
│   │   └── repositories.py   # Repository Protocols (SOLID)
│   ├── application/
│   │   └── usecases.py       # SearchRecipes, EstimatePrice, BuildCart (Use Cases)
│   ├── infrastructure/
│   │   ├── json_repositories.py    # Repo đọc JSON (tùy chọn/offline)
│   │   └── mongo_repositories.py   # Repo đọc MongoDB (recipes, products)
│   ├── model/
│   │   ├── architecture.py   # PhoBERTJointNLU (intent + slot), loss tổng hợp
│   │   └── tokenizer_utils.py# align_labels cho BPE (-100 cho subword/special)
│   ├── training/
│   │   ├── dataset_builder.py # Tạo synthetic data (≥200 mẫu/intent)
│   │   └── trainer.py        # Loop huấn luyện (AMP FP16, GradScaler)
│   ├── services/
│   │   ├── nlu_engine.py     # NLU inference (AMP), merge BIO → entities
│   │   └── search_engine.py  # Hybrid Search: BM25 + TF-IDF + faiss-cpu
│   └── api/
│       ├── schemas.py        # Pydantic models (Req/Resp)
│       └── routes.py         # FastAPI endpoints (/chat)
├── requirements.txt
└── main.py                   # App entry point (DI, startup, routes)
```

### Kiến trúc & Nguyên tắc

- **Clean Architecture / SOLID**
  - **Domain**: Entities (Recipe, Product), Repository Protocols.
  - **Application**: Use Cases (không phụ thuộc framework/infra).
  - **Infrastructure**: Implement Repos (Mongo/JSON), I/O.
  - **Interfaces**: FastAPI (schemas, routes).
- **NLU**
  - `PhoBERTJointNLU(RobertaPreTrainedModel)` chia sẻ encoder → `intent_head`, `slot_head`.
  - Loss: `total_loss = intent_loss + slot_loss`.
  - **Alignment**: `align_labels` gán label cho **token đầu** của mỗi word; subword/special = **-100**.
- **Search**
  - **BM25** (rank-bm25) + **TF-IDF** (sklearn) + **Faiss L2 (CPU)**, rerank kết hợp (0.6/0.4).
- **Hiệu năng 1650 Max-Q**
  - **AMP FP16**: `torch.cuda.amp.autocast` + `GradScaler`.
  - **Batch nhỏ** (mặc định 8), có gradient clipping & scheduler.
  - **faiss-cpu** giữ VRAM cho PhoBERT.

---

## 🔧 Ghi chú & Mẹo vận hành

- **MongoDB**: đảm bảo 2 collections `recipes`, `products` đã được seed.  
  Trường `discount` là % → giá cuối = `price * (1 - discount/100)`.
- **Khởi động**: startup sẽ
  1. load NLU (từ `models/phobert_joint_nlu/`)
  2. đọc toàn bộ `recipes`/`products` từ Mongo
  3. build chỉ mục BM25/TF-IDF/Faiss
- **Triển khai**: có thể chạy `uvicorn` tiêu chuẩn; nếu traffic lớn, cân nhắc caching kết quả search, hoặc chuyển sang `motor` (async) cho Mongo.

---

## 🧪 Kiểm thử (gợi ý)

- Unit tests cho:
  - `align_labels` (BPE alignment)
  - Merge BIO → entities
  - HybridSearchEngine (BM25/TF-IDF/fusion)
  - Mongo Repos (parse document, discount)
- Integration test `/chat` với các intents chính.

---

## 📜 License

MIT (hoặc cập nhật theo nhu cầu của bạn).

---

## 🙌 Credits

- **PhoBERT**: vinai/phobert-base
- **Libraries**: transformers, torch, scikit-learn, rank_bm25, faiss-cpu, FastAPI, Pydantic, PyMongo
