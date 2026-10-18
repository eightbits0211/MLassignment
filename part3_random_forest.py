"""Part 3: Random Forest on flat trees (bagging + feature subsetting at every split).

Run:  python part3_random_forest.py
"""
import time
import numpy as np

from common import get_split, accuracy, derived_family_accuracy, majority_vote, pct
from part1_decision_tree import DecisionTree
from part2_bagging_boosting import Bagging


class RandomForest:
    def __init__(self, n_estimators=50, criterion="gini", max_features=4,
                 max_depth=None, min_samples_split=2, seed=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        rng = np.random.RandomState(self.seed)
        N = len(y)
        self.n_classes_ = int(y.max()) + 1
        self.trees_ = []
        for _ in range(self.n_estimators):
            idx = rng.randint(0, N, size=N)
            # the tree calls rng.choice(d, m, replace=False) afresh at every node
            tree = DecisionTree(self.criterion, self.max_depth, self.min_samples_split,
                                max_features=self.max_features, rng=rng)
            self.trees_.append(tree.fit(X[idx], y[idx]))
        return self

    def predict_all(self, X):
        return np.array([t.predict(X) for t in self.trees_])

    def predict(self, X):
        return majority_vote(self.predict_all(X), self.n_classes_)


def mean_pairwise_agreement(preds):
    T = len(preds)
    total, pairs = 0.0, 0
    for a in range(T):
        # compare tree a with all later trees at once
        total += np.mean(preds[a + 1:] == preds[a], axis=1).sum()
        pairs += T - a - 1
    return total / pairs


def main():
    d = get_split()
    Xtr, ytr, Xte, yte = d["X_train"], d["ys_train"], d["X_test"], d["ys_test"]
    m = int(np.floor(np.sqrt(Xtr.shape[1])))
    print(f"d = {Xtr.shape[1]} features -> m = floor(sqrt(d)) = {m} per split\n")

    print("=== 3. Random Forest (T=50, bootstrap N, m=4 per split, unconstrained) ===")
    rf_res, rf_models = {}, {}
    for crit in ("gini", "entropy"):
        t0 = time.time()
        rf = RandomForest(50, crit, max_features=m).fit(Xtr, ytr)
        pte = rf.predict(Xte)
        tr, te = accuracy(ytr, rf.predict(Xtr)), accuracy(yte, pte)
        fa = derived_family_accuracy(pte, d["yf_test"], d["s2f"])
        rf_res[crit], rf_models[crit] = (tr, te, fa), rf
        print(f"{crit:8s} train {pct(tr)}  test {pct(te)}  derived family (test) {pct(fa)}  [{time.time() - t0:.1f}s]")

    print("\n=== Comparison of test accuracy and tree agreement ===")
    print(f"{'':8s} {'single':>8s} {'bagging':>8s} {'RF':>8s} | {'agree bag':>9s} {'agree RF':>9s}")
    for crit in ("gini", "entropy"):
        single = DecisionTree(crit).fit(Xtr, ytr)
        bag = Bagging(25, crit).fit(Xtr, ytr)
        ag_bag = mean_pairwise_agreement(bag.predict_all(Xte))
        ag_rf = mean_pairwise_agreement(rf_models[crit].predict_all(Xte))
        print(f"{crit:8s} {pct(accuracy(yte, single.predict(Xte))):>8s} "
              f"{pct(accuracy(yte, bag.predict(Xte))):>8s} {pct(rf_res[crit][1]):>8s} | "
              f"{pct(ag_bag):>9s} {pct(ag_rf):>9s}")

    print("\n=== Results matrix rows: 3. Random Forest ===")
    for crit, name in (("gini", "Gini"), ("entropy", "Entropy")):
        tr, te, fa = rf_res[crit]
        print(f"{name:8s} Train {pct(tr)} | Test {pct(te)} | Family (test) {pct(fa)}")


if __name__ == "__main__":
    main()
