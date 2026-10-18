"""Part 1: Decision tree (Gini / information gain) from scratch.

Run:  python part1_decision_tree.py
"""
import time
import numpy as np

from common import get_split, accuracy, derived_family_accuracy, pct

# Tolerance for comparing floating-point gains (treat |a-b| <= TOL as a tie,
# and require gain > TOL to count as "positive").
TOL = 1e-12


def gini_from_weights(cw):
    """cw: (..., K) class weight sums -> Gini impurity (...)."""
    tot = cw.sum(axis=-1, keepdims=True)
    p = np.divide(cw, tot, out=np.zeros_like(cw, dtype=float), where=tot > 0)
    return 1.0 - np.sum(p * p, axis=-1)


def entropy_from_weights(cw):
    """cw: (..., K) class weight sums -> entropy in bits (...), 0*log2(0) := 0."""
    tot = cw.sum(axis=-1, keepdims=True)
    p = np.divide(cw, tot, out=np.zeros_like(cw, dtype=float), where=tot > 0)
    logp = np.log2(p, out=np.zeros_like(p), where=p > 0)
    return -np.sum(p * logp, axis=-1)


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2,
                 max_features=None, rng=None):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.rng = rng
        self._imp = gini_from_weights if criterion == "gini" else entropy_from_weights

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        n, d = X.shape
        if sample_weight is None:
            w = np.full(n, 1.0 / n)
        else:
            w = np.asarray(sample_weight, dtype=float)
        self.n_features_ = d
        self.n_classes_ = int(y.max()) + 1
        if self.max_features is not None and self.rng is None:
            self.rng = np.random.RandomState(42)

        # one-hot class weights, shape (n, K)
        Wy = np.zeros((n, self.n_classes_))
        Wy[np.arange(n), y] = w

        # node arrays (grown as lists, converted to numpy at the end)
        self._feat, self._thr, self._left, self._right = [], [], [], []
        self._value, self._gain = [], []

        root = self._new_node()
        stack = [(root, np.arange(n), 0)]
        while stack:
            node, idx, depth = stack.pop()
            cw = Wy[idx].sum(axis=0)
            self._value[node] = int(np.argmax(cw))  # weighted majority, ties -> smallest label

            if (np.all(y[idx] == y[idx[0]])                      # pure
                    or (self.max_depth is not None and depth >= self.max_depth)
                    or len(idx) < self.min_samples_split):
                continue

            f, t, gain = self._best_split(X[idx], Wy[idx], cw)
            if f is None:                                         # no positive gain
                continue

            go_left = X[idx, f] <= t
            l, r = self._new_node(), self._new_node()
            self._feat[node], self._thr[node] = f, t
            self._left[node], self._right[node] = l, r
            self._gain[node] = gain
            # push right first so the left subtree is built first (order is cosmetic)
            stack.append((r, idx[~go_left], depth + 1))
            stack.append((l, idx[go_left], depth + 1))

        self.feature_ = np.array(self._feat, dtype=int)
        self.threshold_ = np.array(self._thr, dtype=float)
        self.left_ = np.array(self._left, dtype=int)
        self.right_ = np.array(self._right, dtype=int)
        self.value_ = np.array(self._value, dtype=int)
        self.gain_ = np.array(self._gain, dtype=float)
        return self

    def _new_node(self):
        self._feat.append(-1)
        self._thr.append(np.nan)
        self._left.append(-1)
        self._right.append(-1)
        self._value.append(0)
        self._gain.append(0.0)
        return len(self._feat) - 1

    def _best_split(self, Xn, Wn, cw_parent):
        """Best (feature, threshold, gain) over all features/thresholds at a node.

        All thresholds of all candidate features are scored at once:
        sort each feature column, cumulative-sum the one-hot class weights, and
        read off left/right class weights for every cut position.
        """
        n, d = Xn.shape
        if self.max_features is None:
            feats = np.arange(d)
        else:
            m = min(self.max_features, d)
            # fresh subset at every node; sort so ties go to the lower index
            feats = np.sort(self.rng.choice(d, m, replace=False))

        Xf = Xn[:, feats]                                   # (n, m)
        order = np.argsort(Xf, axis=0, kind="stable")       # (n, m)
        xs = np.take_along_axis(Xf, order, axis=0)          # sorted values
        left_cw = np.cumsum(Wn[order], axis=0)[:-1]         # (n-1, m, K): left = first i+1 rows
        right_cw = cw_parent - left_cw

        W = cw_parent.sum()
        Wl = left_cw.sum(axis=-1)
        Wr = right_cw.sum(axis=-1)
        gain = (self._imp(cw_parent)
                - (Wl / W) * self._imp(left_cw)
                - (Wr / W) * self._imp(right_cw))           # (n-1, m)

        valid = xs[1:] > xs[:-1]                            # cut only between distinct values
        gain = np.where(valid, gain, -np.inf)

        best_f, best_t, best_g = None, None, TOL           # must beat TOL to be "positive"
        for j in range(len(feats)):                         # ascending feature index
            g = gain[:, j]
            gmax = g.max()
            if gmax > best_g + TOL:                         # strictly better (ties keep lower feature)
                i = int(np.argmax(g >= gmax - TOL))         # first = lowest threshold among ties
                best_f = int(feats[j])
                best_t = 0.5 * (xs[i, j] + xs[i + 1, j])
                best_g = float(g[i])
        if best_f is None:
            return None, None, 0.0
        return best_f, best_t, best_g

    # -------------------------------------------------------------- predict
    def apply(self, X):
        """Leaf index for every row (vectorised: one loop step per tree level)."""
        X = np.asarray(X, dtype=float)
        node = np.zeros(len(X), dtype=int)
        active = self.left_[node] != -1
        while active.any():
            rows = np.where(active)[0]
            nd = node[rows]
            go_left = X[rows, self.feature_[nd]] <= self.threshold_[nd]
            node[rows] = np.where(go_left, self.left_[nd], self.right_[nd])
            active = self.left_[node] != -1
        return node

    def predict(self, X):
        return self.value_[self.apply(X)]

    # -------------------------------------------------------------- info
    @property
    def n_nodes(self):
        return len(self.value_)

    @property
    def n_leaves(self):
        return int(np.sum(self.left_ == -1))

    def depth(self):
        depth = np.zeros(self.n_nodes, dtype=int)
        for i in range(self.n_nodes):          # children always have larger ids
            if self.left_[i] != -1:
                depth[self.left_[i]] = depth[self.right_[i]] = depth[i] + 1
        return int(depth.max())


# ====================================================================== checks
def sanity_checks():
    print("=== Sanity checks ===")
    # (a)
    g = gini_from_weights(np.array([3.0, 3.0]))
    h = entropy_from_weights(np.array([3.0, 3.0]))
    assert np.isclose(g, 0.5) and np.isclose(h, 1.0)
    print(f"(a) counts (3,3): G = {g:.4f}, H = {h:.4f} bits  OK")

    # (b)
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0, 0, 1, 1])
    t = DecisionTree("entropy").fit(X, y)
    n_splits = int(np.sum(t.left_ != -1))
    assert n_splits == 1 and np.isclose(t.threshold_[0], 2.5) and np.isclose(t.gain_[0], 1.0)
    assert accuracy(y, t.predict(X)) == 1.0
    print(f"(b) splits = {n_splits}, t = {t.threshold_[0]}, IG = {t.gain_[0]:.4f}, "
          f"train acc = {accuracy(y, t.predict(X)):.1f}  OK")

    # (c)
    X = np.zeros((3, 1))
    y = np.array([0, 0, 1])
    w = np.array([0.1, 0.1, 0.8])
    for crit in ("gini", "entropy"):
        t = DecisionTree(crit).fit(X, y, sample_weight=w)
        assert t.n_nodes == 1 and t.predict(X[:1])[0] == 1
    print("(c) identical samples, weights [0.1,0.1,0.8] -> single leaf predicting 1  OK")

    # (d)
    X = np.random.RandomState(0).randn(5, 3)
    for y in (np.zeros(5, dtype=int), np.full(5, 7)):
        t = DecisionTree("gini").fit(X, y)
        assert t.n_nodes == 1 and np.all(t.predict(X) == y[0])
    print("(d) pure node / single-class data -> one leaf, no crash  OK")

    # extra: tie-break -> lower feature index, then lower threshold
    X = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    t = DecisionTree("gini").fit(X, np.array([0, 0, 1, 1]))
    assert t.feature_[0] == 0
    X = np.array([[1.0], [2.0], [3.0]])
    t = DecisionTree("gini", max_depth=1).fit(X, np.array([0, 1, 0]))
    assert np.isclose(t.threshold_[0], 1.5)
    # extra: leaf tie -> smallest label
    t = DecisionTree("gini").fit(np.zeros((2, 1)), np.array([1, 0]))
    assert t.predict(np.zeros((1, 1)))[0] == 0
    print("(extra) split tie-breaks and leaf tie -> smallest label  OK\n")


# ====================================================================== main
def main():
    sanity_checks()
    d = get_split()
    Xtr, Xte = d["X_train"], d["X_test"]
    ytr, yte = d["ys_train"], d["ys_test"]
    print(f"Train rows: {len(ytr)}, test rows: {len(yte)}\n")

    print("=== Flat single tree on Species (max_depth=None, min_samples_split=2) ===")
    results = {}
    for crit in ("gini", "entropy"):
        t0 = time.time()
        tree = DecisionTree(crit, max_depth=None, min_samples_split=2).fit(Xtr, ytr)
        fit_s = time.time() - t0
        ptr, pte = tree.predict(Xtr), tree.predict(Xte)
        tr_acc, te_acc = accuracy(ytr, ptr), accuracy(yte, pte)
        fam_acc = derived_family_accuracy(pte, d["yf_test"], d["s2f"])
        results[crit] = (tr_acc, te_acc, fam_acc)
        print(f"{crit:8s} train {pct(tr_acc)}  test {pct(te_acc)}  derived family (test) {pct(fam_acc)}"
              f"   [{tree.n_nodes} nodes, {tree.n_leaves} leaves, depth {tree.depth()}, fit {fit_s:.2f}s]")

    print("\n=== Depth sweep (not part of the matrix) ===")
    print(f"{'max_depth':>9s} | {'gini train':>10s} {'gini test':>10s} | {'ent train':>10s} {'ent test':>10s}")
    for md in (1, 2, 3, 5, 8, 12, None):
        row = []
        for crit in ("gini", "entropy"):
            t = DecisionTree(crit, max_depth=md).fit(Xtr, ytr)
            row += [accuracy(ytr, t.predict(Xtr)), accuracy(yte, t.predict(Xte))]
        print(f"{str(md):>9s} | {pct(row[0]):>10s} {pct(row[1]):>10s} | {pct(row[2]):>10s} {pct(row[3]):>10s}")

    print("\n=== Results matrix rows: 1. Single Decision Tree ===")
    for crit, name in (("gini", "Gini"), ("entropy", "Entropy")):
        tr, te, fa = results[crit]
        print(f"{name:8s} Train {pct(tr)} | Test {pct(te)} | Family (test) {pct(fa)}")


if __name__ == "__main__":
    main()
