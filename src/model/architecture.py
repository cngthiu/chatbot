# smart_food_bot/src/model/architecture.py
from __future__ import annotations
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
from transformers import RobertaPreTrainedModel, RobertaModel

class PhoBERTJointNLU(RobertaPreTrainedModel):
    """
    PhoBERT-based joint Intent Classification + Slot Filling.
    1650 Max-Q Optimization: single shared encoder, minimal heads, dropout.
    """
    base_model_prefix = "backbone"

    def __init__(self, config, num_intents: int, num_slots: int):
        super().__init__(config)
        self.num_intents = num_intents
        self.num_slots = num_slots

        # Use `backbone` to stay compatible with the checkpoint that was trained with this attribute name.
        self.backbone = RobertaModel(config)
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))

        hidden = config.hidden_size
        self.intent_classifier = nn.Linear(hidden, num_intents)
        self.slot_classifier = nn.Linear(hidden, num_slots)

        self.intent_loss_fct = nn.CrossEntropyLoss()
        self.slot_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        intent_labels: Optional[torch.Tensor] = None,
        slot_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = self.dropout(outputs.last_hidden_state)
        pooled_output = self.dropout(outputs.last_hidden_state[:, 0])  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)  # [B, num_intents]
        slot_logits = self.slot_classifier(sequence_output)    # [B, T, num_slots]

        total_loss = None
        intent_loss = None
        slot_loss = None
        if intent_labels is not None:
            intent_loss = self.intent_loss_fct(intent_logits, intent_labels)
        if slot_labels is not None:
            slot_loss = self.slot_loss_fct(slot_logits.view(-1, self.num_slots), slot_labels.view(-1))
        if (intent_loss is not None) and (slot_loss is not None):
            total_loss = intent_loss + slot_loss
        elif intent_loss is not None:
            total_loss = intent_loss
        elif slot_loss is not None:
            total_loss = slot_loss

        return {
            "loss": total_loss,
            "intent_logits": intent_logits,
            "slot_logits": slot_logits,
            "intent_loss": intent_loss,
            "slot_loss": slot_loss,
        }
