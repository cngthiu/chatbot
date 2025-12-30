# =========================
# FILE: smart_food_bot/src/services/search_engine.py
# (REWRITE: normalize VN + word+char TFIDF + rerank boosts + cache TTL)
# =========================
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import logging
import time
import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
import scipy.sparse as sp  # scikit-learn dependency
from src.domain.repositories import RecipeReadRepo

log = logging.getLogger("services.search_engine")

_WORD_RE = re.compile(r"[0-9]+|[A-Za-zÀ-ỹ]+(?:[-'][A-Za-zÀ-ỹ]+)*", re.UNICODE)


def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    return s


def _normalize_vi(text: str) -> str:
    t = (text or "").lower().strip()
    t = _strip_accents(t)
    t = re.sub(r"[^0-9a-z\s\-']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize_vi(text: str) -> List[str]:
    return _WORD_RE.findall(_normalize_vi(text))


def _to_faiss_matrix(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(x)


class _TTLCache:
    def __init__(self, ttl_s: int = 60, max_items: int = 512) -> None:
        self.ttl_s = ttl_s
        self.max_items = max_items
        self._data: Dict[str, Tuple[float, Any]] = {}

    def get(self, key: str):
        now = time.time()
        v = self._data.get(key)
        if not v:
            return None
        ts, payload = v
        if now - ts > self.ttl_s:
            self._data.pop(key, None)
            return None
        return payload

    def set(self, key: str, payload: Any) -> None:
        if len(self._data) >= self.max_items:
            # drop oldest
            oldest = sorted(self._data.items(), key=lambda kv: kv[1][0])[: max(1, self.max_items // 10)]
            for k, _ in oldest:
                self._data.pop(k, None)
        self._data[key] = (time.time(), payload)


class HybridSearchEngine:
    """
    Hybrid search (BM25 + TF-IDF via faiss-cpu) on CPU.

    Upgrades:
      - VN normalize (remove accents)
      - BM25 tokenize VN
      - TF-IDF: word ngrams + char_wb ngrams (robust for misspell/spacing)
      - title exact/contains boost for better top1
      - TTL cache to avoid recompute for repeated queries
    """

    def __init__(
        self,
        recipe_repo: RecipeReadRepo,
        topk_candidate_mul: int = 3,
        max_features_word: int = 20000,
        max_features_char: int = 30000,
        cache_ttl_s: int = 60,
    ) -> None:
        self.recipe_repo = recipe_repo
        self.recipes = recipe_repo.all()
        if not self.recipes:
            raise RuntimeError("No recipes available from repository.")
        self.topk_candidate_mul = topk_candidate_mul
        self.cache = _TTLCache(ttl_s=cache_ttl_s)
        self.max_features_word = max_features_word
        self.max_features_char = max_features_char
        self._build_indices()

    def _build_indices(self) -> None:
        corpus: List[str] = []
        title_norm: List[str] = []

        for r in self.recipes:
            joined = " ".join(
                [
                    r.title or "",
                    r.summary or "",
                    " ".join([i.name for i in (r.ingredients or [])]),
                    " ".join(r.tags or []),
                    " ".join(r.diet or []),
                    " ".join(r.search_keywords or []),
                    # Optional: a tiny hint from steps improves “công thức” queries
                    " ".join((r.steps or [])[:2]) if getattr(r, "steps", None) else "",
                ]
            ).strip()
            corpus.append(joined)
            title_norm.append(_normalize_vi(r.title or ""))

        self.corpus = corpus
        self._title_norm = title_norm

        # BM25 on normalized tokens
        self.bm25 = BM25Okapi([_tokenize_vi(doc) for doc in self.corpus])

        # TF-IDF word
        self.vectorizer_word = TfidfVectorizer(
            analyzer="word",
            tokenizer=_tokenize_vi,
            preprocessor=lambda x: x,  # already normalized in tokenizer
            lowercase=False,
            ngram_range=(1, 2),
            min_df=1,
            max_features=self.max_features_word,
        )

        # TF-IDF char (robust)
        self.vectorizer_char = TfidfVectorizer(
            analyzer="char_wb",
            preprocessor=_normalize_vi,
            ngram_range=(3, 5),
            min_df=1,
            max_features=self.max_features_char,
        )

        doc_word = self.vectorizer_word.fit_transform(self.corpus)
        doc_char = self.vectorizer_char.fit_transform(self.corpus)
        doc_sparse = sp.hstack([doc_word, doc_char]).tocsr()

        doc_mat = _to_faiss_matrix(doc_sparse)
        faiss.normalize_L2(doc_mat)

        dim = doc_mat.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(doc_mat)

        log.info("HybridSearchEngine indexed %d recipes | dim=%d", len(self.recipes), dim)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        cache_key = f"{_normalize_vi(query)}|{top_k}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        q_norm = _normalize_vi(query)
        q_tokens = _tokenize_vi(query)

        # BM25
        bm25_scores = self.bm25.get_scores(q_tokens).astype(np.float32)
        denom = (bm25_scores.max() - bm25_scores.min()) + 1e-6
        bm25_norm = (bm25_scores - bm25_scores.min()) / denom

        # TF-IDF query vector
        q_word = self.vectorizer_word.transform([query])
        q_char = self.vectorizer_char.transform([query])
        q_sparse = sp.hstack([q_word, q_char]).tocsr()

        q_vec = _to_faiss_matrix(q_sparse)
        faiss.normalize_L2(q_vec)

        k = min(max(1, top_k * self.topk_candidate_mul), len(self.recipes))
        D, I = self.index.search(q_vec, k)
        faiss_scores = (1.0 / (1.0 + D[0])).astype(np.float32)

        faiss_full = np.zeros(len(self.recipes), dtype=np.float32)
        faiss_full[I[0]] = faiss_scores

        combo = 0.55 * bm25_norm + 0.45 * faiss_full

        # rerank boost: title match
        for idx in I[0]:
            tn = self._title_norm[int(idx)]
            if not tn:
                continue
            if tn == q_norm:
                combo[int(idx)] += 0.25
            elif tn and tn in q_norm or q_norm in tn:
                combo[int(idx)] += 0.10

        idxs = np.argsort(-combo)[: min(top_k, len(self.recipes))]

        results: List[Dict[str, Any]] = []
        for idx in idxs:
            r = self.recipes[int(idx)]
            results.append(
                {
                    "id": r.id,
                    "title": r.title,
                    "summary": r.summary,
                    "ingredients": [i.name for i in r.ingredients],
                    "cook_time": r.cook_time,
                    "servings": r.servings,
                    "image": r.image,
                    "score": float(combo[idx]),
                }
            )

        self.cache.set(cache_key, results)
        return results