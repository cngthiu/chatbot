# smart_food_bot/src/model/tokenizer_utils.py 
from __future__ import annotations
from typing import List, Dict, Tuple
from transformers import PreTrainedTokenizerBase

def align_labels(
    words: List[str],
    bio_labels: List[str],
    tokenizer: PreTrainedTokenizerBase,
    label2id: Dict[str, int],
    max_len: int,
) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    Align word-level BIO labels to subword tokens.
    Only first sub-token of each word gets the label; others & specials get -100.
    Reason: Prevent learning on meaningless BPE suffixes (e.g., @@_bò).

    Returns:
        tokenized (dict of input_ids, attention_mask)
        aligned_slot_labels (List[int])
    """
    assert len(words) == len(bio_labels), "words and bio_labels must be same length"

    tokenized = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors=None,
        return_attention_mask=True,
    )
    word_ids = tokenized.word_ids()
    aligned = []
    prev_word_id = None
    for idx, wid in enumerate(word_ids):
        if wid is None:
            aligned.append(-100)
            continue
        if wid != prev_word_id:  # first token of the word
            label = bio_labels[wid]
            aligned.append(label2id.get(label, label2id["O"]))
        else:
            aligned.append(-100)
        prev_word_id = wid
    # pad/truncate safety
    if len(aligned) < max_len:
        aligned += [-100] * (max_len - len(aligned))
    else:
        aligned = aligned[:max_len]
    return tokenized, aligned