import numpy as np
from numba import njit, prange


class AUROC:
    def __init__(self, min_num_pos=1):
        self.min_num_pos = min_num_pos

    def __call__(self, y_true, y_pred):
        if len(y_true.shape) == 1:
            score_func = _binary_auroc_score
        else:
            score_func = _multi_auroc_score

        return score_func(y_true, y_pred, self.min_num_pos)


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


@njit(nogil=True)
def _classidx_to_onehot(y_true, n_classes):
    """Convert multiclass label array to onehot encoded label format."""
    y_true_onehot = np.zeros((y_true.size, n_classes), dtype=bool_)
    for i, j in enumerate(y_true):
        y_true_onehot[i, j] = True
    return y_true_onehot


@njit(nogil=True)
def _multi_auroc_score(y_true, y_pred, min_num_pos):
    """NUMBA implementation for computing AUROC scores in multilabel settings.

    Iterate over individual class and compute the corresponding AUROC by
    1. Sorting label vector based on the corresponding predictions.
    2. Scaning through the sorted label vector and compute the true positive
        rate (``tpr``) and false positive rate (``fpr``).
    3. Integrating over the ROC curve (``tpr`` vs ``fpr``) via right Riemann
        sum.

    Note:
        If a class does not have any available positive example, then set the
            corresponding AUROC score to ``np.nan``
    Args:
        y_true (numpy.ndarray): n-classes-d one-hot encoded label array.
        y_pred (numpy.ndarray): n_classes-d prediction score array.
        min_mum_pos (int): minimum number of positive required for evaluation.

    Return:
        1-d array of size n_classes containing auroc scores for the
            corresponding classes (nan if no positive example for a class).

    """
    n, n_classes = y_true.shape
    auroc_scores = np.zeros(n_classes)

    # Iterate over classes and compute AUROC individually
    for j in range(n_classes):
        n_pos = np.count_nonzero(y_true[:, j])
        n_neg = n - n_pos

        # Skip computation if no positive example is available
        if n_pos < min_num_pos:
            auroc_scores[j] = np.nan

        else:
            # Initializing ROC and sort the labels by predictions
            auroc_score = n_tp = n_fp = tpr = fpr = 0
            y_true_sorted = y_true[y_pred[:, j].argsort()[::-1], j]

            # Iterate over the sorted prediction array and integrate ROC
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
                    auroc_score += tpr * interval  # right Riemann sum

            auroc_scores[j] = auroc_score

    return auroc_scores
