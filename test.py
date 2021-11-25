import time

import numpy as np
from sklearn.metrics import roc_auc_score

from fastauc.auroc import AUROC


class section:

    def __init__(self, name):
        self.name = name

    def __call__(self, func):

        def wrapped_func(*args, **kwargs):
            bar = '=' * 20
            print(f"{bar}Start testing {self.name}{bar}")
            func(*args, **kwargs)
            print(f"{bar}Done testing {self.name}{bar}")

        return wrapped_func


def runtime_summary(skl_t_list, fac_t_list, diff_list):
    print(
        f"\nSummary stats\n"
        f"Scikit-learn: "
        f"avg = {np.mean(skl_t_list):.2e}, std = {np.std(skl_t_list):.2e}\n"
        f"FastAUC     : "
        f"avg = {np.mean(fac_t_list):.2e}, std = {np.std(fac_t_list):.2e}\n"
        f"Diff: avg={np.mean(diff_list):.2e}, std={np.std(diff_list):.2e}"
    )


@section("binray")
def test_runtime_binary(num_samples, num_pos, repeat):
    skl_t_list, fac_t_list, diff_list = [], [], []

    for i in range(repeat):
        y_pred = np.sort(np.random.random(num_samples))[::-1]
        y_true = np.random.random(num_samples) > np.random.random()
        y_true[:num_pos] = True

        t = time.perf_counter()
        skl_auroc = roc_auc_score(y_true, y_pred)
        skl_t_list.append(time.perf_counter() - t)

        t = time.perf_counter()
        fac_auroc = AUROC()(y_true, y_pred)
        fac_t_list.append(time.perf_counter() - t)

        diff_list.append(np.abs(skl_auroc - fac_auroc))

        print(f"Scikit-learn: {skl_auroc:06.4f}, FastAUR: {fac_auroc:06.4f}")

    runtime_summary(skl_t_list, fac_t_list, diff_list)


@section("multi")
def test_runtime_multi(num_classes, num_samples, num_pos, repeat):
    skl_t_list, fac_t_list, diff_list = [], [], []

    for i in range(repeat):
        y_pred = np.zeros((num_samples, num_classes))
        y_true = np.zeros((num_samples, num_classes), dtype=bool)

        for j in range(num_classes):
            y_pred[:, j] = np.sort(np.random.random(num_samples))[::-1]
            y_true[:, j] = np.random.random(num_samples) > np.random.random()
            y_true[:num_pos, j] = True

        t = time.perf_counter()
        skl_aurocs = [
            roc_auc_score(y_true[:, i], y_pred[:, i])
            for i in range(num_classes)
        ]
        skl_auroc = np.mean(skl_aurocs)
        skl_t_list.append(time.perf_counter() - t)

        t = time.perf_counter()
        fac_auroc = np.mean(AUROC()(y_true, y_pred))
        fac_t_list.append(time.perf_counter() - t)

        diff_list.append(np.abs(skl_auroc - fac_auroc))

        print(f"Scikit-learn: {skl_auroc:06.4f}, FastAUR: {fac_auroc:06.4f}")

    runtime_summary(skl_t_list, fac_t_list, diff_list)


def test_min_num_pos():
    num_samples = 100

    y_pred = np.random.random(num_samples)
    y_true = np.zeros(num_samples, dtype=bool)
    y_true[:5] = True

    assert not np.isnan(AUROC()(y_true, y_pred))
    assert np.isnan(AUROC(min_num_pos=10)(y_true, y_pred))


def main():
    test_runtime_binary(1_000_000, 100_000, 10)
    test_runtime_multi(10, 100_000, 10_000, 10)
    test_min_num_pos()


if __name__ == '__main__':
    main()
