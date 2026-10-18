BITS F464 Machine Learning - Assignment 1
Flat vs. Hierarchical Classification From Scratch
=================================================

Environment
-----------
Python  3.14.0
numpy   2.5.4
pandas  3.0.6
(matplotlib is used only by the optional make_results_pdf.py to draw the PDF.)

No machine-learning libraries are used. Every model, metric, split and
confusion matrix is implemented with numpy / pandas / the standard library.

Setup
-----
Put all .py files in one folder together with Frogs_MFCCs.csv
(UCI "Anuran Calls (MFCCs)"). Paths are relative; run every command
from that folder.

    pip install numpy pandas            # matplotlib only for make_results_pdf.py

How to run (one command per part)
---------------------------------
    python part1_decision_tree.py       # Part 1: single tree (Gini / entropy), sanity checks, depth sweep      (~5 s)
    python part2_bagging_boosting.py    # Part 2: Bagging (T=25) and AdaBoost SAMME (T=50), stumps comparison  (~45 s)
    python part3_random_forest.py       # Part 3: Random Forest (T=50, m=4), pairwise tree agreement          (~35 s)
    python part4_hierarchical.py        # Part 4: router + experts, oracle, pA/pB, confusion matrices         (~3 s)
    python part5_naive_bayes.py         # Part 5: Gaussian Naive Bayes, feature correlation matrix            (~2 s)

Optional:
    python make_results_pdf.py          # re-runs all models and writes results_matrix.pdf                    (~90 s)

Each script prints its sanity checks, its analysis, and at the end the
rows that go into the results matrix.

Files
-----
common.py                   data loading, label encoding, stratified_split (as given, seed 42),
                            species->family map (from training data), accuracy, confusion matrix,
                            majority vote (ties -> smallest label)
part1_decision_tree.py      DecisionTree(criterion, max_depth, min_samples_split, max_features, rng)
part2_bagging_boosting.py   Bagging, AdaBoostSAMME
part3_random_forest.py      RandomForest
part4_hierarchical.py       HierarchicalClassifier(router, experts)
part5_naive_bayes.py        GaussianNB(var_smoothing=1e-9)
make_results_pdf.py         optional: generates results_matrix.pdf
results_matrix.pdf          filled-in Table 2 and Table 3

Notes
-----
- Class labels are encoded in sorted (alphabetical) order, so "ties go to the
  smallest class label" means the alphabetically first class.
- Split gains are compared with a tolerance of 1e-12: a split must have gain
  > 1e-12 to count as positive, and gains within 1e-12 are treated as ties
  (lower feature index, then lower threshold).
