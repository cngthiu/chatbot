# smart_food_bot/src/services/nlu_engine.py
from __future__ import annotations
from typing import Dict, List, Any
import os
import logging
import ujson as json
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer
from src.model.architecture import PhoBERTJointNLU
from src.core.config import Paths, DEVICE, MAX_LEN

log = logging.getLogger("services.nlu")

def _load_mappings(model_dir: str):
    with open(os.path.join(model_dir, "intent2id.json"), "r", encoding="utf-8") as f:
        intent2id = json.load(f)
    with open(os.path.join(model_dir, "slot_label2id.json"), "r", encoding="utf-8") as f:
        slot2id = json.load(f)
    id2intent = {v: k for k, v in intent2id.items()}
    id2slot = {v: k for k, v in slot2id.items()}
    return intent2id, slot2id, id2intent, id2slot

def _merge_bio(words: List[str], slots: List[str]) -> Dict[str, List[str]]:
    entities: Dict[str, List[str]] = {}
    cur_type, cur_tokens = None, []
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
            if cur_type and cur_tokens:
                entities.setdefault(cur_type, []).append(" ".join(cur_tokens))
            cur_type, cur_tokens = None, []
    if cur_type and cur_tokens:
        entities.setdefault(cur_type, []).append(" ".join(cur_tokens))
    return entities

class NLUEngine:
    """1650 Max-Q Optimization: AMP autocast for inference."""
    def __init__(self, model_dir: str = Paths.MODEL_OUT_DIR):
        self.model_dir = model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        cfg = AutoConfig.from_pretrained(model_dir)

        _, _, self.id2intent, self.id2slot = _load_mappings(model_dir)
        num_intents = len(self.id2intent)
        num_slots = len(self.id2slot)

        # gọi classmethod và pass kwargs vào __init__
        self.model = PhoBERTJointNLU.from_pretrained(
            model_dir,
            config=cfg,
            num_intents=num_intents,
            num_slots=num_slots,
        )

        self.model.to(DEVICE)
        self.model.eval()
        log.info("NLU loaded from %s", model_dir)

    @torch.inference_mode()
    def predict(self, text: str) -> Dict[str, Any]:
        words = text.split()
        enc = self.tokenizer(
            words, is_split_into_words=True, return_tensors="pt",
            truncation=True, padding="max_length", max_length=MAX_LEN
        )
        enc = {k: v.to(DEVICE) for k, v in enc.items()}
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            out = self.model(**enc)
            intent_id = int(torch.argmax(F.softmax(out["intent_logits"], dim=-1), dim=-1).item())
            slot_ids = torch.argmax(out["slot_logits"][0], dim=-1).tolist()

        # Re-map token→word by first-token rule
        word_ids = self.tokenizer(words, is_split_into_words=True, truncation=True,
                                  padding="max_length", max_length=MAX_LEN).word_ids()
        word_slots: List[str] = []
        seen = set()
        for tidx, wid in enumerate(word_ids):
            if wid is None or wid in seen:
                continue
            seen.add(wid)
            tag = self.id2slot.get(slot_ids[tidx], "O")
            word_slots.append(tag)
        words = words[:len(word_slots)]
        entities = _merge_bio(words, word_slots)
        return {
            "intent": self.id2intent[intent_id],
            "intent_confidence": 1.0,  # keep concise; use prob if needed
            "slots": entities,
        }
