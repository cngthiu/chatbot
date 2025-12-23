# smart_food_bot/src/services/nlu_engine.py
from __future__ import annotations

import os
import logging
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import ujson as json
import onnxruntime as ort
from transformers import AutoTokenizer

from src.core.config import Paths, MAX_LEN

log = logging.getLogger("services.nlu_onnx")


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


def _merge_bio(words: List[str], slots: List[str]) -> Dict[str, List[str]]:
    """
    Merge BIO tags into entity spans.
    Output keys are entity types: DISH/INGREDIENT/...
    """
    entities: Dict[str, List[str]] = {}
    cur_type: Optional[str] = None
    cur_tokens: List[str] = []

    for w, tag in zip(words, slots):
        if tag == "O" or tag is None:
            if cur_type and cur_tokens:
                entities.setdefault(cur_type, []).append(" ".join(cur_tokens))
            cur_type, cur_tokens = None, []
            continue

        if tag.startswith("B-"):
            if cur_type and cur_tokens:
                entities.setdefault(cur_type, []).append(" ".join(cur_tokens))
            cur_type = tag[2:]
            cur_tokens = [w]
        elif tag.startswith("I-") and cur_type == tag[2:]:
            cur_tokens.append(w)
        else:
            # Tag lỗi/nhảy type → đóng span cũ
            if cur_type and cur_tokens:
                entities.setdefault(cur_type, []).append(" ".join(cur_tokens))
            cur_type, cur_tokens = None, []

    if cur_type and cur_tokens:
        entities.setdefault(cur_type, []).append(" ".join(cur_tokens))
    return entities


class _WordEncoder:
    """
    Robust word->token encoding for ONNX inference, hỗ trợ slow tokenizer.
    WHY: PhoBERT tokenizer nhiều khi không phải 'fast', nên không có word_ids().
    Strategy:
      - Tokenize theo từng word với tiền tố khoảng trắng (roberta-style) để gần khớp encode thực tế.
      - Tự xây input_ids, attention_mask, và positions của 'first sub-token' mỗi word.
    """

    def __init__(self, tokenizer, max_len: int):
        self.tok = tokenizer
        self.max_len = max_len

        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id

        if self.cls_id is None or self.sep_id is None or self.pad_id is None:
            raise ValueError("Tokenizer missing special token ids (cls/sep/pad).")

    def encode(self, words: List[str]) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        # Build token ids manually: [CLS] + words(subtokens) + [SEP]
        input_ids: List[int] = [self.cls_id]
        first_token_pos: List[int] = []

        for i, w in enumerate(words):
            # RoBERTa-family uses space-sensitive BPE; prefix space except first word
            ww = w if i == 0 else " " + w
            piece_ids = self.tok.encode(ww, add_special_tokens=False)
            if not piece_ids:
                # if tokenizer can't encode, skip mapping (will be ignored)
                continue
            # record first sub-token position (current length)
            first_token_pos.append(len(input_ids))
            input_ids.extend(piece_ids)

            # early truncate (keep room for [SEP])
            if len(input_ids) >= self.max_len - 1:
                break

        # Add SEP
        input_ids = input_ids[: self.max_len - 1]
        input_ids.append(self.sep_id)

        # Pad to max_len
        attn = [1] * len(input_ids)
        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids.extend([self.pad_id] * pad_len)
            attn.extend([0] * pad_len)

        # Clip first_token_pos if beyond max_len
        first_token_pos = [p for p in first_token_pos if p < self.max_len]

        # ONNX needs int64
        input_ids_np = np.array([input_ids], dtype=np.int64)     # [1, T]
        attn_np = np.array([attn], dtype=np.int64)               # [1, T]
        return input_ids_np, attn_np, first_token_pos


class NLUEngineONNX:
    """
    ONNX inference engine for PhoBERT Joint Intent+Slot.
    - Uses onnxruntime
    - Returns intent + merged BIO entities
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

        # Load tokenizer from trained directory (best)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        self.encoder = _WordEncoder(self.tokenizer, max_len=max_len)

        _, _, self.id2intent, self.id2slot = _load_mappings(model_dir)

        # Providers
        if providers is None:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)

        log.info("NLUEngineONNX loaded: %s | providers=%s", self.onnx_path, providers)

    def predict(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            return {"intent": "fallback", "intent_confidence": 0.0, "slots": {}}

        words = text.split()
        input_ids, attention_mask, first_pos = self.encoder.encode(words)

        # Run ONNX
        ort_out = self.session.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        # output order: ["intent_logits", "slot_logits"]
        intent_logits = ort_out[0]  # [1, C_intents]
        slot_logits = ort_out[1]    # [1, T, C_slots]

        intent_probs = _softmax(intent_logits, axis=-1)[0]
        intent_id = int(np.argmax(intent_probs))
        intent = self.id2intent.get(intent_id, "fallback")
        intent_conf = float(intent_probs[intent_id])

        # Slot: lấy label tại first sub-token mỗi word
        slot_pred_ids = np.argmax(slot_logits[0], axis=-1).tolist()  # [T]
        word_level_tags: List[str] = []
        used_words = 0
        for p in first_pos:
            if used_words >= len(words):
                break
            tag = self.id2slot.get(int(slot_pred_ids[p]), "O")
            word_level_tags.append(tag)
            used_words += 1

        # If for some reason mapping shorter than words, pad with O
        if len(word_level_tags) < len(words):
            word_level_tags.extend(["O"] * (len(words) - len(word_level_tags)))

        entities = _merge_bio(words, word_level_tags)

        return {
            "intent": intent,
            "intent_confidence": round(intent_conf, 4),
            "slots": entities,
        }
