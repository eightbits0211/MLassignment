"""Recompute every number in the results matrix and write results_matrix.pdf.

Run:  python make_results_pdf.py      (~1.5 min; uses matplotlib only for the PDF)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import get_split, accuracy, derived_family_accuracy
from part1_decision_tree import DecisionTree
from part2_bagging_boosting import Bagging, AdaBoostSAMME
from part3_random_forest import RandomForest
from part4_hierarchical import fit_hierarchical
from part5_naive_bayes import GaussianNB


def flat_row(model, d):
    pte = model.predict(d["X_test"])
    return (accuracy(d["ys_train"], model.predict(d["X_train"])),
            accuracy(d["ys_test"], pte),
            derived_family_accuracy(pte, d["yf_test"], d["s2f"]))


def compute():
    d = get_split()
    X, y = d["X_train"], d["ys_train"]
    rows, fail = [], []
    makers = [("1. Single Decision Tree", lambda c: DecisionTree(c)),
              ("2a. Bagging", lambda c: Bagging(25, c)),
              ("2b. AdaBoost", lambda c: AdaBoostSAMME(50, c, max_depth=3)),
              ("3. Random Forest", lambda c: RandomForest(50, c, max_features=4))]
    for name, make in makers:
        for crit in ("gini", "entropy"):
            rows.append((name, crit, *flat_row(make(crit).fit(X, y), d), False))
            print(name, crit, ["%.2f" % (100 * v) for v in rows[-1][2:5]])
    for crit in ("gini", "entropy"):
        h = fit_hierarchical(X, y, d["yf_train"], d["family_names"], crit)
        ff, ps = h.predict_family(d["X_test"]), h.predict(d["X_test"])
        r_te = accuracy(d["yf_test"], ff)
        o_te = accuracy(d["ys_test"], ps)
        oracle = accuracy(d["ys_test"], h.predict_with_families(d["X_test"], d["yf_test"]))
        pA = float(np.mean((ff == d["yf_test"]) & (ps != d["ys_test"])))
        rows.append(("4. Hierarchical Decision Tree", crit,
                     accuracy(y, h.predict(X)), o_te, r_te, True))
        fail.append((crit, r_te, oracle, pA, 1 - r_te))
        print("hier", crit, ["%.2f" % (100 * v) for v in rows[-1][2:5]])
    rows.append(("5. Gaussian Naive Bayes", None, *flat_row(GaussianNB().fit(X, y), d), False))
    print("nb", ["%.2f" % (100 * v) for v in rows[-1][2:5]])
    return rows, fail


def draw(rows, fail, path="results_matrix.pdf"):
    p = lambda v: f"{100 * v:.2f}"
    fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 140); ax.axis("off")
    T = lambda x, y, s, **k: ax.text(x, y, s, va=k.pop("va", "center"), fontsize=k.pop("fs", 9), **k)
    hl = lambda y, x0, x1, lw=0.6: ax.plot([x0, x1], [y, y], "k-", lw=lw)
    vl = lambda x, y0, y1: ax.plot([x, x], [y0, y1], "k-", lw=0.6)

    T(50, 133, "BITS F464 Machine Learning — Assignment 1: Results Matrix", ha="center", fs=13, weight="bold")
    T(50, 129.5, "Anuran Calls (MFCCs) · stratified 80/20 split, seed 42 · 5,756 train / 1,439 test rows",
      ha="center", fs=8.5, color="0.3")

    # ---------------- Table 2
    T(50, 124, "Table 2: Results matrix (flat models predict Species; † see note)", ha="center", fs=10, weight="bold")
    xs = [6, 38, 52, 68, 82, 96]            # column boundaries
    top, hh, rh = 120, 4, 4.2
    ax.add_patch(plt.Rectangle((xs[0], top - 2 * hh), xs[-1] - xs[0], 2 * hh, color="0.92", zorder=0))
    T((xs[0] + xs[1]) / 2, top - hh, "Method", ha="center", weight="bold")
    T((xs[1] + xs[2]) / 2, top - hh, "Criterion", ha="center", weight="bold")
    T((xs[2] + xs[4]) / 2, top - hh / 2, "Overall accuracy (Species) %", ha="center", weight="bold")
    T((xs[4] + xs[5]) / 2, top - hh / 2, "Family acc. † %", ha="center", weight="bold")
    for i, s in enumerate(["Train", "Test", "Test"]):
        T((xs[2 + i] + xs[3 + i]) / 2, top - 1.5 * hh, s, ha="center", weight="bold")
    hl(top, xs[0], xs[-1], 1.2); hl(top - hh, xs[2], xs[-1]); hl(top - 2 * hh, xs[0], xs[-1], 1.0)
    y = top - 2 * hh
    i = 0
    while i < len(rows):
        name = rows[i][0]
        group = [r for r in rows if r[0] == name]
        gh = rh * len(group)
        if group[0][5]:
            ax.add_patch(plt.Rectangle((xs[0], y - gh), xs[-1] - xs[0], gh, color="#e6e6fa", zorder=0))
        T(xs[0] + 1, y - gh / 2, name)
        for j, (_, crit, tr, te, fa, hier) in enumerate(group):
            yc = y - rh * (j + 0.5)
            T((xs[1] + xs[2]) / 2, yc, {"gini": "Gini", "entropy": "Entropy", None: "—"}[crit], ha="center")
            T((xs[2] + xs[3]) / 2, yc, p(tr), ha="center")
            T((xs[3] + xs[4]) / 2, yc, p(te), ha="center")
            T((xs[4] + xs[5]) / 2, yc, ("Router: " if hier else "") + p(fa), ha="center",
              weight="bold" if hier else "normal")
            if j:
                hl(y - rh * j, xs[1], xs[-1], 0.3)
        y -= gh
        hl(y, xs[0], xs[-1], 0.6)
        i += len(group)
    hl(y, xs[0], xs[-1], 1.2)
    vl(xs[1], y, top); vl(xs[2], y, top); vl(xs[4], y, top)
    vl(xs[3], y, top - hh)                      # Train|Test divider under the merged header
    vl(xs[0], y, top); vl(xs[-1], y, top)
    note = ("† For flat models, the derived family accuracy (each predicted species mapped to its family via the\n"
            "species→family map read from the training data). For the Hierarchical rows, the router accuracy;\n"
            "their 'Overall accuracy' columns hold the final species accuracy.")
    T(xs[0], y - 5, note, fs=7.5, color="0.25")

    # ---------------- Table 3
    y3 = y - 15
    T(50, y3, "Table 3: Supplementary — hierarchical failure analysis (test set, %)", ha="center", fs=10, weight="bold")
    cx = [6, 20, 36, 52, 78, 96]
    t3 = y3 - 4
    ax.add_patch(plt.Rectangle((cx[0], t3 - hh), cx[-1] - cx[0], hh, color="0.92", zorder=0))
    heads = ["Criterion", "Router acc.", "Oracle acc.", "(A) Router right, expert wrong", "(B) Router wrong"]
    for k, s in enumerate(heads):
        T((cx[k] + cx[k + 1]) / 2, t3 - hh / 2, s, ha="center", weight="bold", fs=8.5)
    hl(t3, cx[0], cx[-1], 1.2); hl(t3 - hh, cx[0], cx[-1], 1.0)
    yy = t3 - hh
    for crit, r, o, pa, pb in fail:
        vals = [crit.capitalize(), p(r), p(o), p(pa), p(pb)]
        for k, s in enumerate(vals):
            T((cx[k] + cx[k + 1]) / 2, yy - rh / 2, s, ha="center")
        yy -= rh
        hl(yy, cx[0], cx[-1], 0.4)
    hl(yy, cx[0], cx[-1], 1.2)
    for x in cx:
        vl(x, yy, t3)
    T(cx[0], yy - 4, "Check: overall accuracy = router acc. − (A), and (B) = 100 − router acc., for both criteria.",
      fs=7.5, color="0.25")

    fig.savefig(path)
    print("wrote", path)


if __name__ == "__main__":
    draw(*compute())
