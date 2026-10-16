from __future__ import annotations

import re
from collections import Counter


def normalize_answer(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def exact_match(golds: list[str], prediction: str) -> float:
    pred = normalize_answer(prediction)
    return float(any(normalize_answer(gold) == pred for gold in golds)) if golds else 0.0


def token_f1(golds: list[str], prediction: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    if not golds or not pred_tokens:
        return 0.0
    best = 0.0
    pred_counter = Counter(pred_tokens)
    for gold in golds:
        gold_tokens = normalize_answer(gold).split()
        if not gold_tokens:
            continue
        common = pred_counter & Counter(gold_tokens)
        same = sum(common.values())
        if same == 0:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def recall_at_k(gold_doc_ids: list[int], predicted_doc_ids: list[int], k: int) -> float:
    gold = list(dict.fromkeys(int(item) for item in gold_doc_ids))
    if not gold:
        return 0.0
    pred = set(int(item) for item in predicted_doc_ids[:k])
    return sum(1 for item in gold if item in pred) / len(gold)
