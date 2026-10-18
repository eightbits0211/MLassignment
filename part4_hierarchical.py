"""Part 4: Hierarchical decision tree (router on Family -> per-family Species experts).

Run:  python part4_hierarchical.py
"""
import numpy as np

from common import get_split, accuracy, confusion_matrix, derived_family_accuracy, pct
from part1_decision_tree import DecisionTree


class HierarchicalClassifier:
    def __init__(self, router, experts):
        self.router = router
        self.experts = experts
        # family ids are the sorted family names, same as in load_data
        self.family_names = np.array(sorted(experts))

    def predict_family(self, X):
        return self.router.predict(X)

    def predict_with_families(self, X, families):
        X = np.asarray(X, dtype=float)
        out = np.full(len(X), -1, dtype=int)
        for f in np.unique(families):
            rows = np.where(families == f)[0]
            out[rows] = self.experts[self.family_names[f]].predict(X[rows])
        return out

    def predict(self, X):
        return self.predict_with_families(X, self.predict_family(X))


def fit_hierarchical(X, y_species, y_family, family_names, criterion):
    router = DecisionTree(criterion).fit(X, y_family)
    experts = {}
    for f, name in enumerate(family_names):
        rows = y_family == f  # experts use the true family, not the router's
        experts[name] = DecisionTree(criterion).fit(X[rows], y_species[rows])
    h = HierarchicalClassifier(router, experts)
    assert np.array_equal(h.family_names, family_names)
    return h


def print_cm(cm, names, title):
    short = [n[:10] for n in names]
    print(title + "  (rows = true, cols = predicted)")
    print(" " * 24 + " ".join(f"{s:>10s}" for s in short))
    for i, n in enumerate(names):
        print(f"{n:>24s}" + " ".join(f"{v:>10d}" for v in cm[i]))


def evaluate(h, d, crit):
    Xtr, Xte = d["X_train"], d["X_test"]
    ys_tr, ys_te, yf_tr, yf_te = d["ys_train"], d["ys_test"], d["yf_train"], d["yf_test"]
    fam, spc = d["family_names"], d["species_names"]

    ff_tr, ff_te = h.predict_family(Xtr), h.predict_family(Xte)
    ps_tr, ps_te = h.predict(Xtr), h.predict(Xte)
    ps_oracle = h.predict_with_families(Xte, yf_te)

    r_tr, r_te = accuracy(yf_tr, ff_tr), accuracy(yf_te, ff_te)
    o_tr, o_te = accuracy(ys_tr, ps_tr), accuracy(ys_te, ps_te)
    oracle = accuracy(ys_te, ps_oracle)
    router_ok, species_ok = ff_te == yf_te, ps_te == ys_te
    pA = float(np.mean(router_ok & ~species_ok))
    pB = float(np.mean(~router_ok))
    assert not np.any(species_ok & ~router_ok)

    print(f"\n==================== Hierarchical tree: {crit} ====================")
    for name, e in h.experts.items():
        cls = np.unique(e.value_)
        print(f"  expert {name:16s}: {e.n_nodes:4d} nodes, species {', '.join(str(s) for s in spc[cls])}")
    print(f"Router accuracy   train {pct(r_tr)}  test {pct(r_te)}")
    print(f"Overall accuracy  train {pct(o_tr)}  test {pct(o_te)}")
    print(f"Oracle accuracy (test, true-family routing) {pct(oracle)}")
    print(f"pA (router right, expert wrong) {pct(pA)}   pB (router wrong) {pct(pB)}")
    print(f"Check: A_overall = {o_te:.6f}, A_router - pA = {r_te - pA:.6f}, "
          f"1 - pA - pB = {1 - pA - pB:.6f} -> {'OK' if np.isclose(o_te, r_te - pA) else 'MISMATCH'}")
    print(f"Price of router errors (oracle - overall) = {pct(oracle - o_te)}\n")

    print_cm(confusion_matrix(yf_te, ff_te, len(fam)), fam, "Router confusion matrix (test)")
    print()
    print_cm(confusion_matrix(ys_te, ps_te, len(spc)), spc, "Species confusion matrix (test)")

    print("\nPer-family test accuracy (rows grouped by TRUE family):")
    print(f"{'family':>16s} {'n':>5s} {'router':>9s} {'species':>9s} {'oracle':>9s}")
    for f, name in enumerate(fam):
        m = yf_te == f
        print(f"{name:>16s} {m.sum():5d} {pct(accuracy(yf_te[m], ff_te[m])):>9s} "
              f"{pct(accuracy(ys_te[m], ps_te[m])):>9s} {pct(accuracy(ys_te[m], ps_oracle[m])):>9s}")
    return dict(r_tr=r_tr, r_te=r_te, o_tr=o_tr, o_te=o_te, oracle=oracle, pA=pA, pB=pB)


def sanity_checks(d):
    print("=== Sanity checks ===")
    # order preserved: predicting a shuffled X gives the shuffled predictions
    h = fit_hierarchical(d["X_train"], d["ys_train"], d["yf_train"], d["family_names"], "gini")
    perm = np.random.RandomState(0).permutation(len(d["X_test"]))
    assert np.array_equal(h.predict(d["X_test"])[perm], h.predict(d["X_test"][perm]))
    # matches a slow per-sample reference
    ref = [h.experts[d["family_names"][h.router.predict(x[None])[0]]].predict(x[None])[0]
           for x in d["X_test"][:300]]
    assert np.array_equal(ref, h.predict(d["X_test"][:300]))
    # every expert only predicts species of its own family; single-species experts are one leaf
    for f, name in enumerate(d["family_names"]):
        e = h.experts[name]
        assert np.all(d["s2f"][e.value_] == f)
        if len(np.unique(d["ys_train"][d["yf_train"] == f])) == 1:
            assert e.n_nodes == 1
    print("Order preserved, matches per-sample reference, experts stay in own family,"
          " single-species experts are one leaf  OK")


def main():
    d = get_split()
    sanity_checks(d)
    res, flat = {}, {}
    for crit in ("gini", "entropy"):
        h = fit_hierarchical(d["X_train"], d["ys_train"], d["yf_train"], d["family_names"], crit)
        res[crit] = evaluate(h, d, crit)
        t = DecisionTree(crit).fit(d["X_train"], d["ys_train"])
        p = t.predict(d["X_test"])
        flat[crit] = (accuracy(d["ys_test"], p), derived_family_accuracy(p, d["yf_test"], d["s2f"]))

    print("\n=== Comparison with flat single tree (test) ===")
    for crit in ("gini", "entropy"):
        r = res[crit]
        print(f"{crit:8s} species: hierarchical {pct(r['o_te'])} vs flat {pct(flat[crit][0])} | "
              f"family: router {pct(r['r_te'])} vs flat-derived {pct(flat[crit][1])}")

    print("\n=== Results matrix rows: 4. Hierarchical Decision Tree ===")
    for crit, name in (("gini", "Gini"), ("entropy", "Entropy")):
        r = res[crit]
        print(f"{name:8s} Train {pct(r['o_tr'])} | Test {pct(r['o_te'])} | Router: {pct(r['r_te'])}")
    print("\n=== Table 3: failure analysis (test) ===")
    for crit, name in (("gini", "Gini"), ("entropy", "Entropy")):
        r = res[crit]
        print(f"{name:8s} Router {pct(r['r_te'])} | Oracle {pct(r['oracle'])} | "
              f"(A) {pct(r['pA'])} | (B) {pct(r['pB'])}")


if __name__ == "__main__":
    main()
