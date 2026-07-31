import numpy as np
from src.metrics.base_metric import BaseMetric


class BinaryAccuracy(BaseMetric):
    """Share of correctly predicted classes in one batch"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, logits, labels, **kwargs):
        return (logits.argmax(dim=-1) == labels).float().mean().item()


def compute_eer(scores, labels):
    """Compute EER in %"""
    s, l = np.asarray(scores), np.asarray(labels)
    bon_sc, spoof_sc = s[l == 1], s[l == 0]
    all_scores = np.concatenate((bon_sc, spoof_sc))
    
    targets = np.concatenate((np.ones(bon_sc.size), np.zeros(spoof_sc.size)))
    targets = targets[np.argsort(all_scores, kind="mergesort")]

    bon_sums = np.cumsum(targets) # Bonafide samples rejected at each threshold
    spoof_sums = spoof_sc.size - (np.arange(1, all_scores.size + 1) - bon_sums)

    false_rej = np.concatenate((np.atleast_1d(0), bon_sums / bon_sc.size))
    false_acc = np.concatenate((np.atleast_1d(1), spoof_sums / spoof_sc.size))

    eer_index = np.argmin(np.abs(false_rej - false_acc))
    return float(np.mean((false_rej[eer_index], false_acc[eer_index])) * 100)
