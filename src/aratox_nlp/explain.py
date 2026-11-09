"""Word-occlusion explanations.

Remove one word at a time and record how much each label score drops.
The same method works for TF-IDF, QARiB and the ensemble, so highlights can be
compared across models. It shows sensitivity, not the model's reasoning.
"""

from __future__ import annotations

import numpy as np


def word_occlusion(text, score_fn, max_words=60):
    """Return (words, base_scores, deltas) where deltas[i, j] = base[j] - score without word i."""
    words = text.split()
    if not words:
        return [], None, np.zeros((0, 0))
    words = words[:max_words]
    variants = [" ".join(words)] + [" ".join(words[:i] + words[i + 1 :]) for i in range(len(words))]
    scores = np.asarray(score_fn(variants), dtype=float)
    base = scores[0]
    return words, base, base - scores[1:]


def top_words(words, deltas, label_index, k=5, min_delta=0.005):
    order = np.argsort(-deltas[:, label_index])
    return [(words[i], float(deltas[i, label_index])) for i in order[:k] if deltas[i, label_index] > min_delta]
