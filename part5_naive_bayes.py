"""Part 5: Flat Gaussian Naive Bayes.

Run:  python part5_naive_bayes.py
"""
import numpy as np

from common import get_split, accuracy, derived_family_accuracy, pct


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y)
        K, d = len(self.classes_), X.shape[1]
        # epsilon = 1e-9 * largest per-feature variance of the training data
        self.epsilon_ = self.var_smoothing * np.max(np.var(X, axis=0))
        self.prior_ = np.zeros(K)
        self.mean_ = np.zeros((K, d))
        self.var_ = np.zeros((K, d))
        for k, c in enumerate(self.classes_):         # loop over classes only
            Xc = X[y == c]
            self.prior_[k] = len(Xc) / len(X)          # pi_c = n_c / N
            self.mean_[k] = Xc.mean(axis=0)
            self.var_[k] = Xc.var(axis=0) + self.epsilon_   # 1/n_c variance + epsilon
        return self

    def joint_log_likelihood(self, X):
        """(n, K) matrix of log pi_c + sum_j log N(x_j | mu_cj, sigma2_cj)."""
        X = np.asarray(X, dtype=float)
        diff = X[:, None, :] - self.mean_[None, :, :]                      # (n, K, d)
        ll = -0.5 * np.sum(np.log(2 * np.pi * self.var_)[None]
                           + diff ** 2 / self.var_[None], axis=2)          # (n, K)
        return np.log(self.prior_)[None, :] + ll

    def predict(self, X):
        return self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)]  # ties -> smallest


def sanity_checks():
    print("=== Sanity check ===")
    X = np.array([[0.0], [2.0], [10.0], [12.0]])
    y = np.array([0, 0, 1, 1])
    nb = GaussianNB().fit(X, y)
    assert np.allclose(nb.mean_.ravel(), [1, 11])
    assert np.allclose(nb.var_.ravel(), [1, 1])
    assert np.isclose(nb.var_[0, 0] - 1, nb.epsilon_)
    pred = nb.predict(np.array([[3.0], [9.0]]))
    assert np.array_equal(pred, [0, 1])
    print(f"means {nb.mean_.ravel()}, variances 1 + eps (eps = {nb.epsilon_:.1e}); "
          f"x=3 -> {pred[0]}, x=9 -> {pred[1]}  OK\n")


def main():
    sanity_checks()
    d = get_split()
    Xtr, Xte, ytr, yte = d["X_train"], d["X_test"], d["ys_train"], d["ys_test"]

    nb = GaussianNB(1e-9).fit(Xtr, ytr)
    ptr, pte = nb.predict(Xtr), nb.predict(Xte)
    tr, te = accuracy(ytr, ptr), accuracy(yte, pte)
    fa = derived_family_accuracy(pte, d["yf_test"], d["s2f"])
    print("=== 5. Gaussian Naive Bayes (flat, Species) ===")
    print(f"train {pct(tr)}  test {pct(te)}  derived family (test) {pct(fa)}")

    print("\nPer-species test accuracy:")
    for k, name in enumerate(d["species_names"]):
        m = yte == k
        print(f"  {name:24s} n={m.sum():4d}  {pct(accuracy(yte[m], pte[m]))}")

    # correlation matrix of the 22 training features
    C = np.corrcoef(Xtr, rowvar=False)
    off = np.abs(C[np.triu_indices_from(C, k=1)])
    print("\n=== Correlation matrix of the 22 training features ===")
    print("      " + " ".join(f"{j + 1:>5d}" for j in range(C.shape[0])))
    for i in range(C.shape[0]):
        print(f"{i + 1:>5d} " + " ".join(f"{v:5.2f}" for v in C[i]))
    print(f"\nOff-diagonal |r|: mean {off.mean():.3f}, max {off.max():.3f}; "
          f"pairs with |r| > 0.5: {np.sum(off > 0.5)} of {len(off)}, > 0.7: {np.sum(off > 0.7)}")
    iu = np.triu_indices_from(C, k=1)
    top = np.argsort(-np.abs(C[iu]))[:5]
    print("Strongest pairs: " + ", ".join(
        f"MFCC{iu[0][t] + 1}-MFCC{iu[1][t] + 1} ({C[iu][t]:+.2f})" for t in top))

    print("\n=== Results matrix row: 5. Gaussian Naive Bayes ===")
    print(f"Train {pct(tr)} | Test {pct(te)} | Family (test) {pct(fa)}")


if __name__ == "__main__":
    main()
