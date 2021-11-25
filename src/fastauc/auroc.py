import numpy as np
from numba import njit, prange


class AUROC:
    def __init__(self, min_num_pos=1):
        self.min_num_pos = min_num_pos

    def __call__(self, y_true, y_pred):
        return _binary_auroc_score(y_true, y_pred, self.min_num_pos)


@njit(nogil=True, cache=True)
def _binary_auroc_score(y_true, y_pred, min_num_pos):
    n = y_true.shape[0]
    y_true_sorted = y_true[y_pred.argsort()[::-1]] == 1
    n_pos = y_true_sorted.sum()
    n_neg = n - n_pos

    if n_pos < min_num_pos:
        return np.nan

    n_tp = n_fp = tpr = fpr = 0

    auroc_score = 0
    for i in range(n):
        prev_tpr, prev_fpr = tpr, fpr

        if y_true_sorted[i]:
            n_tp += 1
        else:
            n_fp += 1

        tpr = n_tp / n_pos
        fpr = n_fp / n_neg

        interval = fpr - prev_fpr
        if interval > 0:
            auroc_score += tpr * interval

    return auroc_score
