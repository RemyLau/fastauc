import time

import numpy as np
from sklearn.metrics import roc_auc_score

from fastauc.auroc import AUROC


def test_runtime(ary_size, num_pos, repeat):
    skl_t_list, fac_t_list, diff_list = [], [], []

    for i in range(repeat):
        y_pred = np.sort(np.random.random(ary_size))[::-1]
        y_true = np.random.random(ary_size) > np.random.random()
        y_true[:num_pos] = True

        t = time.perf_counter()
        skl_auroc = roc_auc_score(y_true, y_pred)
        skl_t_list.append(time.perf_counter() - t)

        t = time.perf_counter()
        fac_auroc = AUROC()(y_true, y_pred)
        fac_t_list.append(time.perf_counter() - t)

        diff_list.append(np.abs(skl_auroc - fac_auroc))

        print(f"Scikit-learn: {skl_auroc:06.4f}, FastAUR: {fac_auroc:06.4f}")

    print("\nFinal evaluation")
    print(f"Run time statistics for Scikit-learn: "
          f"avg = {np.mean(skl_t_list):.2e}, std = {np.std(skl_t_list):.2e}")
    print(f"Run time statistics for FastAUC     : "
          f"avg = {np.mean(fac_t_list):.2e}, std = {np.std(fac_t_list):.2e}")
    print(f"Diff: avg={np.mean(diff_list):.2e}, std={np.std(diff_list):.2e}")


def test_min_num_pos():
    ary_size = 100

    y_pred = np.random.random(ary_size)
    y_true = np.zeros(ary_size, dtype=bool)
    y_true[:5] = True

    assert not np.isnan(AUROC()(y_true, y_pred))
    assert np.isnan(AUROC(min_num_pos=10)(y_true, y_pred))


def main():
    test_runtime(ary_size=1_000_000, num_pos=100_000, repeat=10)
    test_min_num_pos()


if __name__ == '__main__':
    main()
