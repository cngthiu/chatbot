# =========================
# FILE: smart_food_bot/src/services/nlu_engine.py
# (REWRITE: fix slot alignment + better VN tokenization)
# =========================
from __future__ import annotations

import os
import logging
import re
import unicodedata
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import ujson as json
import onnxruntime as ort
from transformers import AutoTokenizer

from src.core.config import Paths, MAX_LEN

log = logging.getLogger("services.nlu_onnx")

_WORD_RE = re.compile(r"[0-9]+|[A-Za-zÀ-ỹ]+(?:[-'][A-Za-zÀ-ỹ]+)*", re.UNICODE)


def _normalize_space(s: str) -> str:
    return " ".join((s or "").strip().split())


def _load_mappings(model_dir: str) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, str], Dict[int, str]]:
    with open(os.path.join(model_dir, "intent2id.json"), "r", encoding="utf-8") as f:
        intent2id = json.load(f)
    with open(os.path.join(model_dir, "slot_label2id.json"), "r", encoding="utf-8") as f:
        slot2id = json.load(f)
    id2intent = {v: k for k, v in intent2id.items()}
    id2slot = {v: k for k, v in slot2id.items()}
    return intent2id, slot2id, id2intent, id2slot


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-9)


def _tokenize_words_vi(text: str) -> List[str]:
    """
    WHY: text.split() breaks VN punctuation and causes slot drift.
    """
    t = _normalize_space(text)
    return _WORD_RE.findall(t)


def _merge_bio(words: List[str], tags: List[str]) -> Dict[str, List[str]]:
    """
    Merge BIO tags into entity spans.
    Fixes:
      - Orphan I-xxx => treat as B-xxx
      - Jumping entity types => flush previous
      - Dedup + clean
    """
    entities: Dict[str, List[str]] = {}
    cur_type: Optional[str] = None
    cur_tokens: List[str] = []

    def flush() -> None:
        nonlocal cur_type, cur_tokens
        if cur_type and cur_tokens:
            val = _normalize_space(" ".join(cur_tokens))
            if val:
                entities.setdefault(cur_type, []).append(val)
        cur_type, cur_tokens = None, []

    for w, tag in zip(words, tags):
        tag = tag or "O"
        if tag == "O":
            flush()
            continue

        if "-" not in tag:
            flush()
            continue

        pref, typ = tag.split("-", 1)
        typ = typ.strip().upper()

        if pref == "I" and cur_type is None:
            pref = "B"

        if pref == "B" or (cur_type is not None and typ != cur_type):
            flush()
            cur_type = typ
            cur_tokens = [w]
        else:
            cur_tokens.append(w)

    flush()

    # deduplicate per type
    for k, vals in list(entities.items()):
        seen = set()
        dedup: List[str] = []
        for v in vals:
            key = v.lower()
            if key not in seen:
                seen.add(key)
                dedup.append(v)
        entities[k] = dedup

    return entities


class _WordEncoder:
    """
    Robust word->token encoding for ONNX inference.
    FIX: return mapping of (word_index -> first_subtoken_pos) to avoid drift.
    """

    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        if self.cls_id is None or self.sep_id is None or self.pad_id is None:
            raise ValueError("Tokenizer missing special token ids (cls/sep/pad).")

    def encode(self, words: List[str]) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int]]]:
        input_ids: List[int] = [self.cls_id]
        mapping: List[Tuple[int, int]] = []  # (word_idx, token_pos)

        for wi, w in enumerate(words):
            ww = w if wi == 0 else " " + w
            piece_ids = self.tok.encode(ww, add_special_tokens=False)

            if not piece_ids:
                continue

            pos = len(input_ids)
            mapping.append((wi, pos))
            input_ids.extend(piece_ids)

            if len(input_ids) >= self.max_len - 1:
                break

        input_ids = input_ids[: self.max_len - 1]
        input_ids.append(self.sep_id)

        attn = [1] * len(input_ids)
        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids.extend([self.pad_id] * pad_len)
            attn.extend([0] * pad_len)

        # clip mapping
        mapping = [(wi, p) for (wi, p) in mapping if p < self.max_len]

        return (
            np.array([input_ids], dtype=np.int64),
            np.array([attn], dtype=np.int64),
            mapping,
        )


class NLUEngineONNX:
    """
    ONNX inference engine for PhoBERT Joint Intent+Slot.
    Upgrades:
      - VN word tokenizer (regex)
      - correct word->token mapping (prevents slot drift)
      - BIO merge fixes
    """

    def __init__(
        self,
        model_dir: str = Paths.MODEL_OUT_DIR,
        onnx_relpath: str = os.path.join("onnx", "phobert_joint_nlu.onnx"),
        max_len: int = MAX_LEN,
        providers: Optional[List[str]] = None,
    ):
        self.model_dir = model_dir
        self.onnx_path = os.path.join(model_dir, onnx_relpath)
        self.max_len = max_len

        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.encoder = _WordEncoder(self.tokenizer, max_len=max_len)

        _, _, self.id2intent, self.id2slot = _load_mappings(model_dir)

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)

        log.info("NLUEngineONNX loaded: %s | providers=%s", self.onnx_path, providers)

    def predict(self, text: str) -> Dict[str, Any]:
        text = _normalize_space(text)
        if not text:
            return {"intent": "fallback", "intent_confidence": 0.0, "slots": {}}

        words = _tokenize_words_vi(text)
        if not words:
            return {"intent": "fallback", "intent_confidence": 0.0, "slots": {}}

        input_ids, attention_mask, mapping = self.encoder.encode(words)

        ort_out = self.session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        intent_logits = ort_out[0]  # [1, C_intents]
        slot_logits = ort_out[1]    # [1, T, C_slots]

        intent_probs = _softmax(intent_logits, axis=-1)[0]
        intent_id = int(np.argmax(intent_probs))
        intent = self.id2intent.get(intent_id, "fallback")
        intent_conf = float(intent_probs[intent_id])

        slot_pred_ids = np.argmax(slot_logits[0], axis=-1).tolist()  # [T]

        # build word-level tags using mapping
        word_tags = ["O"] * len(words)
        for wi, pos in mapping:
            if 0 <= wi < len(words) and 0 <= pos < len(slot_pred_ids):
                word_tags[wi] = self.id2slot.get(int(slot_pred_ids[pos]), "O")

        entities = _merge_bio(words, word_tags)

        return {
            "intent": intent,
            "intent_confidence": round(intent_conf, 4),
            "slots": entities,
        }
