import time

import numpy as np
from sklearn.metrics import roc_auc_score

from fastauc.auroc import AUROC


def main():
    n = 1000000
    m = 100000
    repeat = 10

    skl_t_list, fac_t_list, diff_list = [], [], []

    for i in range(repeat):
        y_pred = np.sort(np.random.random(n))[::-1]
        y_true = np.random.random(n) > np.random.random()
        y_true[:m] = True

        t = time.perf_counter()
        skl_auroc = roc_auc_score(y_true, y_pred)
        skl_t_list.append(time.perf_counter() - t)

        t = time.perf_counter()
        fac_auroc = AUROC()(y_true, y_pred)
        fac_t_list.append(time.perf_counter() - t)

        diff_list.append(np.abs(skl_auroc - fac_auroc))

        print(f"Scikit-learn: {skl_auroc:07.4f}, FastAUR: {fac_auroc:07.4f}")

    print("\nFinal evaluation")
    print(f"Run time statistics for Scikit-learn: "
          f"avg = {np.mean(skl_t_list):.2e}, std = {np.std(skl_t_list):.2e}")
    print(f"Run time statistics for FastAUC     : "
          f"avg = {np.mean(fac_t_list):.2e}, std = {np.std(fac_t_list):.2e}")
    print(f"Diff: avg={np.mean(diff_list):.2e}, std={np.std(diff_list):.2e}")


if __name__ == '__main__':
    main()
