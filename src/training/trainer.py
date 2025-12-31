# smart_food_bot/src/training/trainer.py
from __future__ import annotations
import os
import math
import random
from typing import Dict, List, Tuple
import ujson as json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from src.core.config import Paths, DEVICE, BATCH_SIZE, ACCUM_STEPS, MAX_LEN, EPOCHS, LR, SEED, MODEL_NAME
from src.model.architecture import PhoBERTJointNLU
from src.model.tokenizer_utils import align_labels

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class JointDataset(Dataset):
    def __init__(self, records: List[Dict], tokenizer, label2id: Dict[str, int], intent2id: Dict[str, int], max_len: int):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.intent2id = intent2id
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        words = rec["tokens"]
        bio = rec["bio_labels"]
        tokenized, slot_labels = align_labels(words, bio, self.tokenizer, self.label2id, self.max_len)
        intent_id = self.intent2id[rec["intent"]]
        item = {
            "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(tokenized["attention_mask"], dtype=torch.long),
            "slot_labels": torch.tensor(slot_labels, dtype=torch.long),
            "intent_labels": torch.tensor(intent_id, dtype=torch.long),
        }
        return item

def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out = {k: torch.stack([b[k] for b in batch]) for k in keys}
    return out

def train():
    set_seed(SEED)
    
    # 1. Load Data
    with open(Paths.DATA_RAW, "r", encoding="utf-8") as f:
        data = json.load(f)
    random.shuffle(data) 
    with open(os.path.join(Paths.DATA_PROCESSED_DIR, "slot_label2id.json"), "r", encoding="utf-8") as f:
        slot_label2id = json.load(f)
    with open(os.path.join(Paths.DATA_PROCESSED_DIR, "intent2id.json"), "r", encoding="utf-8") as f:
        intent2id = json.load(f)

    # Split 90/10
    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    config = AutoConfig.from_pretrained(MODEL_NAME)
    model = PhoBERTJointNLU(config, num_intents=len(intent2id), num_slots=len(slot_label2id))
    model.to(DEVICE)

    train_ds = JointDataset(train_data, tokenizer, slot_label2id, intent2id, MAX_LEN)
    val_ds = JointDataset(val_data, tokenizer, slot_label2id, intent2id, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate, num_workers=0)
    
    optim = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS // max(1, ACCUM_STEPS)
    warmup = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optim, warmup, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))
    
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                out = model(**batch)
                loss = out["loss"] / ACCUM_STEPS
            scaler.scale(loss).backward()
            if step % ACCUM_STEPS == 0:
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
                scheduler.step()
            running_loss += loss.item()

        # === EVALUATION LOOP (UPDATED) ===
        model.eval()
        val_loss = 0.0
        correct_intents = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                    out = model(**batch)
                    val_loss += out["loss"].item()
                    
                    # Tính Intent Accuracy sơ bộ
                    logits = out["intent_logits"] # [B, num_intents]
                    preds = torch.argmax(logits, dim=1)
                    labels = batch["intent_labels"]
                    correct_intents += (preds == labels).sum().item()
                    total_samples += labels.size(0)

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct_intents / total_samples if total_samples > 0 else 0.0

        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Intent Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_all(model, tokenizer, intent2id, slot_label2id)

def save_all(model, tokenizer, intent2id: Dict[str, int], slot_label2id: Dict[str, int]):
    os.makedirs(Paths.MODEL_OUT_DIR, exist_ok=True)
    model.save_pretrained(Paths.MODEL_OUT_DIR)
    tokenizer.save_pretrained(Paths.MODEL_OUT_DIR)

    # Save mappings
    with open(os.path.join(Paths.MODEL_OUT_DIR, "intent2id.json"), "w", encoding="utf-8") as f:
        json.dump(intent2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(Paths.MODEL_OUT_DIR, "slot_label2id.json"), "w", encoding="utf-8") as f:
        json.dump(slot_label2id, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    train()