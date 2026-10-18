"""Part 2: Bagging and AdaBoost (multi-class SAMME) on flat trees.

Run:  python part2_bagging_boosting.py
"""
import time
import numpy as np

from common import get_split, accuracy, derived_family_accuracy, majority_vote, pct
from part1_decision_tree import DecisionTree


class Bagging:
    """T unconstrained trees on bootstrap samples (all features), majority vote."""

    def __init__(self, n_estimators=25, criterion="gini", max_depth=None,
                 min_samples_split=2, seed=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        rng = np.random.RandomState(self.seed)        # one fresh generator per ensemble
        N = len(y)
        self.n_classes_ = int(y.max()) + 1
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, N, size=N)              # bootstrap: N rows with replacement
            tree = DecisionTree(self.criterion, self.max_depth, self.min_samples_split)
            self.trees_.append(tree.fit(X[idx], y[idx]))
        return self

    def predict_all(self, X):
        """(T, n) matrix of individual tree predictions."""
        return np.array([t.predict(X) for t in self.trees_])

    def predict(self, X):
        return majority_vote(self.predict_all(X), self.n_classes_)   # ties -> smallest label


class AdaBoostSAMME:
    """Multi-class AdaBoost (SAMME) with weighted DecisionTree weak learners."""

    def __init__(self, n_estimators=50, criterion="gini", max_depth=3):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        N = len(y)
        K = int(y.max()) + 1
        self.n_classes_ = K
        w = np.full(N, 1.0 / N)                                   # 1. w_i = 1/N
        self.trees_, self.alphas_, self.errors_ = [], [], []
        self.stopped_early_ = False
        for _ in range(self.n_estimators):
            tree = DecisionTree(self.criterion, max_depth=self.max_depth).fit(X, y, sample_weight=w)  # 2.
            wrong = tree.predict(X) != y
            eps = float(np.sum(w[wrong]))                         # 3. weighted error (sum w = 1)
            eps = min(max(eps, 1e-10), 1 - 1e-10)
            if eps >= 1 - 1.0 / K:                                # no better than random: discard, stop
                self.stopped_early_ = True
                break
            alpha = np.log((1 - eps) / eps) + np.log(K - 1)       # 4. amount of say
            w = w * np.exp(alpha * wrong)                         # 5. boost misclassified samples
            w = w / w.sum()
            self.trees_.append(tree)
            self.alphas_.append(alpha)
            self.errors_.append(eps)
        return self

    def decision_scores(self, X):
        """(n, K) matrix of sum_t alpha_t * 1[h_t(x) = k]."""
        X = np.asarray(X, dtype=float)
        scores = np.zeros((len(X), self.n_classes_))
        rows = np.arange(len(X))
        for tree, alpha in zip(self.trees_, self.alphas_):
            scores[rows, tree.predict(X)] += alpha
        return scores

    def predict(self, X):
        return np.argmax(self.decision_scores(X), axis=1)        # 6. ties -> smallest label


# ====================================================================== checks
def sanity_checks():
    print("=== Sanity checks ===")
    # K = 2: SAMME alpha = 2 * (classic AdaBoost alpha) -> identical classifier
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    y = (X[:, 0] + 0.5 * X[:, 1] ** 2 > 0.3).astype(int)
    ab = AdaBoostSAMME(10, "gini", max_depth=1).fit(X, y)
    # re-run classic binary AdaBoost by hand with labels in {-1,+1}
    w = np.full(len(y), 1 / len(y)); ys = 2 * y - 1; F = np.zeros(len(y))
    for t, (tree, a) in enumerate(zip(ab.trees_, ab.alphas_)):
        h = 2 * tree.predict(X) - 1
        e = np.sum(w[h != ys]); a_bin = 0.5 * np.log((1 - e) / e)
        assert np.isclose(a, 2 * a_bin)
        w = w * np.exp(-a_bin * ys * h); w /= w.sum()
        F += a_bin * h
    # SAMME score(1) - score(0) = 2F, so argmax (ties -> 0) equals sign(F) with F = 0 -> class 0
    assert np.array_equal(np.where(F > 0, 1, 0), ab.predict(X))
    print("K=2: SAMME alpha = 2 x binary AdaBoost alpha, same predictions  OK")

    # majority-vote tie -> smallest label
    assert np.array_equal(majority_vote(np.array([[3, 1], [1, 3]]), 4), [1, 1])
    print("Vote tie -> smallest label  OK\n")


def report(name, model, d):
    ptr = model.predict(d["X_train"])
    pte = model.predict(d["X_test"])
    tr, te = accuracy(d["ys_train"], ptr), accuracy(d["ys_test"], pte)
    fa = derived_family_accuracy(pte, d["yf_test"], d["s2f"])
    return tr, te, fa


# ====================================================================== main
def main():
    sanity_checks()
    d = get_split()
    Xtr, ytr = d["X_train"], d["ys_train"]

    single = {}
    for crit in ("gini", "entropy"):
        t = DecisionTree(crit).fit(Xtr, ytr)
        single[crit] = accuracy(d["ys_test"], t.predict(d["X_test"]))

    print("=== 2a. Bagging (T=25, bootstrap N, all 22 features, unconstrained) ===")
    bag_res = {}
    for crit in ("gini", "entropy"):
        t0 = time.time()
        m = Bagging(25, crit).fit(Xtr, ytr)
        bag_res[crit] = report("Bagging", m, d)
        tr, te, fa = bag_res[crit]
        print(f"{crit:8s} train {pct(tr)}  test {pct(te)}  derived family (test) {pct(fa)}  [{time.time() - t0:.1f}s]")

    print("\n=== 2b. AdaBoost SAMME (T=50, max_depth=3) ===")
    ada_res = {}
    for crit in ("gini", "entropy"):
        t0 = time.time()
        m = AdaBoostSAMME(50, crit, max_depth=3).fit(Xtr, ytr)
        ada_res[crit] = report("AdaBoost", m, d)
        tr, te, fa = ada_res[crit]
        print(f"{crit:8s} train {pct(tr)}  test {pct(te)}  derived family (test) {pct(fa)}"
              f"  [{len(m.trees_)} rounds used, early stop={m.stopped_early_}, {time.time() - t0:.1f}s]")

    print("\n=== AdaBoost with stumps (max_depth=1, T=50) vs max_depth=3 (not in matrix) ===")
    for crit in ("gini", "entropy"):
        m = AdaBoostSAMME(50, crit, max_depth=1).fit(Xtr, ytr)
        te = accuracy(d["ys_test"], m.predict(d["X_test"]))
        print(f"{crit:8s} stumps test {pct(te)}  ({len(m.trees_)} rounds, first eps {m.errors_[0]:.3f}, "
              f"last eps {m.errors_[-1]:.3f})   vs depth-3 test {pct(ada_res[crit][1])}")

    print("\n=== Test accuracy vs single tree (Part 1) ===")
    for crit in ("gini", "entropy"):
        print(f"{crit:8s} single {pct(single[crit])} | bagging {pct(bag_res[crit][1])} | adaboost {pct(ada_res[crit][1])}")

    print("\n=== Results matrix rows ===")
    for label, res in (("2a. Bagging", bag_res), ("2b. AdaBoost", ada_res)):
        for crit, name in (("gini", "Gini"), ("entropy", "Entropy")):
            tr, te, fa = res[crit]
            print(f"{label:13s} {name:8s} Train {pct(tr)} | Test {pct(te)} | Family (test) {pct(fa)}")


if __name__ == "__main__":
    main()
