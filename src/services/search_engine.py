# smart_food_bot/src/services/search_engine.py
from __future__ import annotations
from typing import List, Dict, Any
import logging
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from src.domain.repositories import RecipeReadRepo

log = logging.getLogger("services.search_engine")

def _tokenize_vi(text: str) -> List[str]:
    return text.lower().strip().split()

def _to_faiss_matrix(x) -> np.ndarray:
    """
    Convert TF-IDF output (usually scipy sparse) to a FAISS-ready numpy array:
    - np.ndarray
    - float32
    - C-contiguous
    """
    # scipy sparse has .toarray()
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x, dtype=np.float32)
    return np.ascontiguousarray(x)

class HybridSearchEngine:
    """
    Hybrid search (BM25 + TF-IDF via faiss-cpu).
    - Keep FAISS on CPU.
    """
    def __init__(self, recipe_repo: RecipeReadRepo) -> None:
        self.recipe_repo = recipe_repo
        self.recipes = recipe_repo.all()
        if not self.recipes:
            raise RuntimeError("No recipes available from repository.")
        self._build_indices()

    def _build_indices(self) -> None:
        corpus: List[str] = []
        for r in self.recipes:
            joined = " ".join([
                r.title or "",
                r.summary or "",
                " ".join([i.name for i in (r.ingredients or [])]),
                " ".join(r.tags or []),
                " ".join(r.diet or []),
                " ".join(r.search_keywords or []),
            ]).strip()
            corpus.append(joined)

        self.corpus = corpus

        # BM25
        self.bm25 = BM25Okapi([_tokenize_vi(doc) for doc in self.corpus])

        # TF-IDF (sparse)
        self.vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
        doc_sparse = self.vectorizer.fit_transform(self.corpus)  # scipy sparse

        # Convert to FAISS matrix (dense float32 contiguous)
        doc_mat = _to_faiss_matrix(doc_sparse)

        # Normalize for cosine-like similarity with L2 (optional but consistent)
        faiss.normalize_L2(doc_mat)

        # Build FAISS index
        dim = doc_mat.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(doc_mat)

        # Keep for debugging if needed (optional)
        self.doc_mat = doc_mat

        log.info("HybridSearchEngine indexed %d recipes | dim=%d", len(self.recipes), dim)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        q_tokens = _tokenize_vi(query)

        # BM25
        bm25_scores = self.bm25.get_scores(q_tokens).astype(np.float32)
        denom = (bm25_scores.max() - bm25_scores.min()) + 1e-6
        bm25_norm = (bm25_scores - bm25_scores.min()) / denom

        # TF-IDF query vector (sparse -> dense)
        q_sparse = self.vectorizer.transform([query])
        q_vec = _to_faiss_matrix(q_sparse)
        faiss.normalize_L2(q_vec)

        # FAISS search
        k = min(top_k * 3, len(self.recipes))
        D, I = self.index.search(q_vec, k)  # D: distances (lower better)

        # Convert distance to a score (higher better)
        faiss_scores = (1.0 / (1.0 + D[0])).astype(np.float32)

        faiss_full = np.zeros(len(self.recipes), dtype=np.float32)
        faiss_full[I[0]] = faiss_scores

        combo = 0.6 * bm25_norm + 0.4 * faiss_full
        idxs = np.argsort(-combo)[: min(top_k, len(self.recipes))]

        results: List[Dict[str, Any]] = []
        for idx in idxs:
            r = self.recipes[int(idx)]
            results.append({
                "id": r.id,
                "title": r.title,
                "summary": r.summary,
                "ingredients": [i.name for i in r.ingredients],
                "cook_time": r.cook_time,
                "servings": r.servings,
                "image": r.image,
                "score": float(combo[idx]),
            })
        return results
