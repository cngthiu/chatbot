# =========================
# FILE: smart_food_bot/src/services/search_engine.py
# REFACTORED FOR PERFORMANCE & MINIMALISM
# Strategy: Lazy Loading + NumPy Backend + Int8 Quantization
# =========================
from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from rank_bm25 import BM25Okapi
# Import class nhưng KHÔNG load model ngay lập tức để Startup nhanh
from sentence_transformers import SentenceTransformer

from src.domain.repositories import RecipeReadRepo
from src.core.config import DEVICE

log = logging.getLogger("services.search_engine")

# --- Utilities (Siêu tối giản & Nhanh) ---
def _normalize_vi(text: str) -> str:
    """Chuẩn hóa chuỗi nhẹ nhàng cho BM25"""
    if not text: return ""
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9\s]", " ", t).strip()

def _tokenize_vi(text: str) -> List[str]:
    return _normalize_vi(text).split()

class HybridSearchEngine:
    """
    Search Engine hiệu năng cao:
    - Lazy Loading: Startup tốn 0 giây.
    - NumPy Backend: Không cài FAISS, không lỗi dependency.
    - Int8 Quantization: Giảm 4 lần RAM tiêu thụ.
    """

    def __init__(
        self,
        recipe_repo: RecipeReadRepo,
        model_name: str = "keepitreal/vietnamese-sbert",
    ) -> None:
        self.repo = recipe_repo
        self.recipes = recipe_repo.all()
        self.model_name = model_name
        
        # 1. Khởi tạo Index nhẹ (BM25) - Tốn < 10MB RAM, mất < 0.1s
        if self.recipes:
            log.info("Initializing BM25 index for %d recipes...", len(self.recipes))
            self.bm25_corpus = [
                _tokenize_vi(f"{r.title} {' '.join(i.name for i in r.ingredients)}") 
                for r in self.recipes
            ]
            self.bm25 = BM25Okapi(self.bm25_corpus)
            self.titles_norm = [_normalize_vi(r.title) for r in self.recipes]
        else:
            log.warning("No recipes found to index!")
            self.bm25 = None
            self.titles_norm = []

        # 2. Tài nguyên NẶNG (Để dành, chưa load)
        self._model: Optional[SentenceTransformer] = None
        self._vectors: Optional[np.ndarray] = None 

    def _ensure_model_loaded(self):
        """Chốt chặn Lazy Loading: Chỉ chạy khi User thực sự tìm kiếm"""
        if self._model is not None:
            return
        if not self.recipes:
            # Không có dữ liệu để index, bỏ qua để tránh tải model vô ích
            return

        log.info("❄️ COLD START: Loading Semantic Model (First Request Only)...")
        # Load Model
        model = SentenceTransformer(self.model_name, device=DEVICE)
        
        # Tối ưu RAM: Nén Model xuống Int8 nếu chạy CPU
        if DEVICE == "cpu":
            log.info("🚀 Applying Dynamic Quantization (Float32 -> Int8)...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        self._model = model

        # Index Vector một lần duy nhất (Cache vào RAM)
        if self.recipes:
            log.info("Indexing vectors...")
            texts = [
                f"{r.title}. {r.summary or ''}. {' '.join((r.steps or [])[:3])}" 
                for r in self.recipes
            ]
            # Encode batch
            embeddings = self._model.encode(texts, batch_size=32, convert_to_numpy=True, show_progress_bar=False)
            
            # Normalize L2 để dùng Dot Product thay cho Cosine Similarity
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self._vectors = embeddings / (norms + 1e-9) 
            
        log.info("Search Engine Ready. RAM optimized.")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query.strip() or not self.recipes:
            return []

        # Bước 1: BM25 Score (Nhanh, dựa trên từ khóa)
        q_tokens = _tokenize_vi(query)
        if self.bm25:
            bm25_scores = np.array(self.bm25.get_scores(q_tokens), dtype=np.float32)
            # Chuẩn hóa Min-Max
            _min, _max = bm25_scores.min(), bm25_scores.max()
            if _max > _min:
                bm25_scores = (bm25_scores - _min) / (_max - _min)
        else:
            bm25_scores = np.zeros(len(self.recipes), dtype=np.float32)

        # Bước 2: Semantic Score (Kích hoạt Lazy Load tại đây)
        self._ensure_model_loaded()
        
        # Encode Query
        q_vec = self._model.encode([query], convert_to_numpy=True)
        q_norm = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-9)
        
        # Tính toán: Ma trận nhân Vector (Dot Product)
        # Cực nhanh với NumPy nhờ SIMD/AVX2
        if self._vectors is not None:
            semantic_scores = np.dot(self._vectors, q_norm.T).flatten()
        else:
            semantic_scores = np.zeros(len(self.recipes), dtype=np.float32)

        # Bước 3: Kết hợp (0.3 BM25 + 0.7 Semantic)
        final_scores = (0.3 * bm25_scores) + (0.7 * semantic_scores)

        # Bước 4: Rerank (Ưu tiên khớp tiêu đề)
        q_norm_str = _normalize_vi(query)
        for i, t_norm in enumerate(self.titles_norm):
            if q_norm_str in t_norm: final_scores[i] += 0.2

        # Bước 5: Lấy Top K (Dùng argpartition để sort nhanh hơn)
        if len(final_scores) > top_k:
            ind = np.argpartition(final_scores, -top_k)[-top_k:]
            ind = ind[np.argsort(final_scores[ind])[::-1]]
        else:
            ind = np.argsort(final_scores)[::-1]

        results = []
        for idx in ind:
            score = float(final_scores[idx])
            if score < 0.25: continue # Lọc kết quả rác
            r = self.recipes[idx]
            results.append({
                "id": r.id,
                "title": r.title,
                "score": score,
                "ingredients": [i.name for i in r.ingredients],
                "image": r.image,
                "summary": r.summary
            })
        
        return results
