# =========================
# FILE: smart_food_bot/src/services/nlu_postprocess.py
# =========================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

_ALLOWED_SLOT_KEYS = {"DISH", "INGREDIENT", "QUANTITY", "UNIT", "TASTE", "EXCLUDE"}

def _entity_key_from_bio(label: str) -> str | None:
    # label like B-DISH / I-INGREDIENT
    if not label or label == "O":
        return None
    if "-" not in label:
        return None
    _, ent = label.split("-", 1)
    ent = ent.strip().upper()
    return ent if ent in _ALLOWED_SLOT_KEYS else ent

def merge_bio_spans(tokens: List[str], bio_labels: List[str]) -> Dict[str, List[str]]:
    """
    WHY: Fix slot quality:
    - Merge B-xxx + I-xxx into full phrase
    - Convert orphan I-xxx into B-xxx
    - Deduplicate values
    """
    slots: Dict[str, List[str]] = {}
    cur_type: str | None = None
    cur_buf: List[str] = []

    def flush():
        nonlocal cur_type, cur_buf
        if cur_type and cur_buf:
            val = " ".join(cur_buf).strip()
            if val:
                slots.setdefault(cur_type, []).append(val)
        cur_type, cur_buf = None, []

    for tok, lab in zip(tokens, bio_labels):
        lab = lab or "O"
        if lab == "O" or lab == "-100":
            flush()
            continue

        ent = _entity_key_from_bio(lab)
        if ent is None:
            flush()
            continue

        prefix = lab.split("-", 1)[0]
        if prefix == "I" and cur_type is None:
            # orphan I -> treat as B
            prefix = "B"

        if prefix == "B" or (cur_type is not None and ent != cur_type):
            flush()
            cur_type = ent
            cur_buf = [tok]
        else:
            # I- same entity
            cur_buf.append(tok)

    flush()

    # dedupe per key
    for k, vals in slots.items():
        dedup = []
        seen = set()
        for v in vals:
            vv = v.strip().lower()
            if vv and vv not in seen:
                seen.add(vv)
                dedup.append(v.strip())
        slots[k] = dedup

    return slots
def normalize_slots(slots: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    WHY: Ensure keys are stable and values cleaned.
    """
    out: Dict[str, List[str]] = {}
    for k, vals in (slots or {}).items():
        kk = (k or "").strip().upper()
        if kk in ("DISH", "MON", "FOOD"):
            kk = "DISH"
        elif kk in ("ING", "INGREDIENTS", "NGUYENLIEU"):
            kk = "INGREDIENT"
        elif kk in ("QTY", "AMOUNT"):
            kk = "QUANTITY"
        elif kk in ("U", "UNITs"):
            kk = "UNIT"
        elif kk in ("FLAVOR", "TASTES"):
            kk = "TASTE"
        elif kk in ("EXC", "EXCLUDE_INGREDIENT"):
            kk = "EXCLUDE"

        cleaned = []
        for v in vals or []:
            vv = " ".join(str(v).split()).strip()
            if vv:
                cleaned.append(vv)
        if cleaned:
            out[kk] = cleaned
    return out
