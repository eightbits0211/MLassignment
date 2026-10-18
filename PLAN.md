# BITS F464 – Assignment 1: Flat vs. Hierarchical Classification From Scratch

**Due:** 23:55, 26-10-2026 · **Demo/Viva:** 27–29 Oct (20 min/group)
**Allowed:** Python 3, numpy, pandas, stdlib, matplotlib (plots only). **No** sklearn/scipy/torch/etc.

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked/issue

---

## Progress overview

| Phase | Status | Target date |
|---|---|---|
| 0. Setup & data | [x] | 16 Oct |
| 1. `common.py` | [x] | 16 Oct |
| 2. Part 1 – Decision Tree | [x] | 16 Oct |
| 3. Part 2 – Bagging & AdaBoost | [x] | 16 Oct |
| 4. Part 3 – Random Forest | [x] | 16 Oct |
| 5. Part 4 – Hierarchical | [x] | 16 Oct |
| 6. Part 5 – Gaussian NB | [x] | 16 Oct |
| 7. Results matrix PDF | [x] | 16 Oct |
| 8. README, cleanup, zip | [ ] | 24 Oct |
| 9. Viva prep | [ ] | 25–26 Oct |
| Buffer | — | 25–26 Oct |

---

## Phase 0 – Setup & data
- [x] Download `Frogs_MFCCs.csv` from UCI into this folder
- [x] Confirm 7,195 rows; columns `MFCCs_1..MFCCs_22`, `Family`, `Genus`, `Species`, `RecordID`
  - ⚠️ Feature column names in the CSV are `MFCCs_ 1`…`MFCCs_ 9` (with a space), then `MFCCs_10`…`MFCCs_22` → select by `startswith("MFCCs_")`
- [x] Check for missing values / dtypes — 0 missing; 22 float64 features
- [x] Note class counts per Species and per Family
  - Family: Bufonidae 68 · Dendrobatidae 542 · Hylidae 2165 · Leptodactylidae 4420
  - Species: AdenomeraAndre 672 · AdenomeraHylaedactylus 3478 · Ameeregatrivittata 542 · HylaMinuta 310 · HypsiboasCinerascens 472 · HypsiboasCordobae 1121 · LeptodactylusFuscus 270 · OsteocephalusOophagus 114 · Rhinellagranulosa 68 · ScinaxRuber 148
- [x] Note species→family mapping; identify single-species families (needed for Part 4)
  - Bufonidae → {Rhinellagranulosa} (**single species**)
  - Dendrobatidae → {Ameeregatrivittata} (**single species**)
  - Hylidae → {HylaMinuta, HypsiboasCinerascens, HypsiboasCordobae, OsteocephalusOophagus, ScinaxRuber} (5)
  - Leptodactylidae → {AdenomeraAndre, AdenomeraHylaedactylus, LeptodactylusFuscus} (3)
  - Every species maps to exactly one family ✔
  - Classes are heavily imbalanced (AdenomeraHylaedactylus ≈ 48%; Rhinellagranulosa < 1%)
- [x] Record Python / numpy / pandas versions (for README) — Python 3.14.0 · numpy 2.5.4 · pandas 3.0.6 (venv at `.venv/`, run with `.venv/bin/python`)

## Phase 1 – `common.py`
- [x] `load_data()` → `X` (n×22 float), `y_species`, `y_family` (relative path `Frogs_MFCCs.csv`)
- [x] Label encoding (`np.unique(..., return_inverse=True)`; species ids 0–9 alphabetical, family ids 0=Bufonidae 1=Dendrobatidae 2=Hylidae 3=Leptodactylidae)
  - sorted class names → ints 0..K-1 (so "smallest label" tie-break = sorted order)
- [x] `stratified_split(labels, test_frac=0.2, seed=42)` — copied **exactly** from the PDF
- [x] Verify split sizes ≈ 5,756 train / 1,439 test — **exactly 5,756 / 1,439**
  - test per species (ids 0–9): 134, 696, 108, 62, 94, 224, 54, 23, 14, 30
- [x] `species_to_family` mapping built from **training data only**
- [x] `accuracy(y_true, y_pred)` (numpy)
- [x] `confusion_matrix(y_true, y_pred, K)` (numpy, e.g. `np.add.at`)
- [x] `majority_vote(pred_matrix, K)` with ties → smallest label (`np.argmax` on counts does this)
- [x] `derived_family_accuracy(species_pred, family_true)`
- [ ] Assert no prohibited imports anywhere (grep at the end — done in Phase 8)
- [x] Extra: `get_split()` returns all arrays + names + `s2f` in one dict; `pct()` formats as `xx.xx%`
- [x] Helpers tested: majority vote tie → smallest label, confusion matrix, accuracy

## Phase 2 – Part 1: Decision Tree (`part1_decision_tree.py`)
### Implementation
- [x] Class signature `DecisionTree(criterion, max_depth=None, min_samples_split=2, max_features=None, rng=None)`
- [x] `fit(X, y, sample_weight=None)`; default weights = 1/N; record `n_classes`
- [x] Impurity functions: Gini `1-Σp²`, entropy `-Σp log2 p` with `0·log0 = 0`
- [x] Node class probabilities from **weight sums** (not counts)
- [x] Vectorised best split per feature:
  - [x] sort feature once per node (`argsort`)
  - [x] cumulative weighted class sums (`cumsum` of one-hot × weight) → left/right class weights at every cut
  - [x] only cut between **distinct** consecutive values; threshold = midpoint
  - [x] gain = Imp(parent) − Σ (W_j/W)·Imp(child_j) for all cuts at once
- [x] Feature selection: all features, or `rng.choice(d, max_features, replace=False)` **per node** when `max_features` set
- [x] Tie-break: lower feature index, then lower threshold (iterate features in ascending order; use strict `>`; `argmax` picks first threshold)
- [x] Stopping: pure node · `max_depth` reached · `n < min_samples_split` · best gain ≤ 0
- [x] Leaf = weighted majority class, ties → smallest label
- [x] Storage: flat arrays (feature, threshold, left, right, value) or node objects
- [x] `predict(X)` vectorised (or iterative per-node masking), no per-sample Python loop if possible
- [x] Unconstrained tree on full train set fits in a few seconds

### Sanity checks (must pass)
- [x] (a) counts (3,3) → G = 0.5, H = 1.0
- [x] (b) X=[[1],[2],[3],[4]], y=[0,0,1,1] → one split at t=2.5, gain 1, train acc 1.0
- [x] (c) 3 identical samples, y=[0,0,1], w=[0.1,0.1,0.8] → leaf predicts 1
- [x] (d) pure node / single-class data → single leaf, no crash

### Task outputs
- [x] Flat Species tree, Gini: train acc, test acc
- [x] Flat Species tree, Entropy: train acc, test acc
- [x] Derived family test accuracy for both
- [x] Depth sweep `{1,2,3,5,8,12,None}` × both criteria: train & test acc (printed, not in matrix)
- [x] Confirm train acc = 100% for unconstrained tree
- [x] Record numbers in **Results log** below
- [x] Verification: brute-force reference tree (loops over every threshold) built in scratchpad; **96/96 configs identical** in node count + test predictions (gini/entropy × depth None/3 × all features / max_features=2 with rng × uniform/random weights × heavy duplicate values)
- [x] Timing: unconstrained tree fits in ~0.5 s; full script ~5 s

**Part 1 numbers**
- Gini: 421 nodes, depth 17 · Entropy: 361 nodes, depth 11
- Depth sweep (train/test): gini 1→60.98/61.01, 3→78.51/79.22, 8→97.85/93.95, None→100/94.93; entropy 1→61.78/62.40, 3→84.40/84.50, 8→98.68/94.86, 12→100/95.41
- Implementation notes: gains compared with tolerance 1e-12 (gain must exceed it to count as positive; near-equal gains treated as ties); `apply()` walks the tree one level at a time for all rows at once


## Phase 3 – Part 2: Bagging & AdaBoost (`part2_bagging_boosting.py`)
### Bagging
- [x] `Bagging(n_estimators=25, criterion)`; fresh `RandomState(42)` per model
- [x] Bootstrap via `rng.randint(0, N, size=N)` and array indexing
- [x] Unconstrained trees, all 22 features
- [x] Majority vote, ties → smallest label
- [x] Run Gini & Entropy: train, test, derived family acc

### AdaBoost (SAMME)
- [x] `AdaBoostSAMME(n_estimators=50, max_depth=3, criterion)`
- [x] Init w = 1/N
- [x] Each round: fit weighted tree, ε = Σ w·1[wrong]
- [x] Clip ε to [1e-10, 1−1e-10]
- [x] If ε ≥ 1 − 1/K: discard learner and stop
- [x] α = ln((1−ε)/ε) + ln(K−1)
- [x] w ← w·exp(α·1[wrong]); normalise
- [x] Store (tree, α); predict = argmax_k Σ α·1[h(x)=k] (ties → smallest label)
- [x] Run Gini & Entropy: train, test, derived family acc
- [x] Extra: stumps (max_depth=1, T=50) test acc printed beside depth-3
- [x] Write a note: why stumps struggle with 10 classes
  - Stumps test: gini 83.32%, entropy 84.16% (vs depth-3: 95.34% / 94.16%). A stump has 2 leaves, so it can output at most 2 of the 10 species; on its own it's barely better than the 1−1/K = 0.9 random threshold late in boosting (ε rose from ~0.39 to ~0.78–0.80). Each round can only separate one or two classes from the rest.

### Analysis
- [x] Compare test acc vs single tree (Part 1)
  - Gini: single 94.93 → bagging 96.66, AdaBoost 95.34 · Entropy: single 95.41 → bagging 96.87, AdaBoost 94.16
  - Bagging reduces variance of unconstrained trees (+1.5–1.7 pts). AdaBoost with depth-3 trees is roughly on par with a single deep tree (better with Gini, worse with entropy).
- [x] Sanity: K=2 SAMME α = 2 × binary AdaBoost α, same predictions
- [x] Verification: AdaBoost re-done with pure-Python loops (ε, α, weight update, weighted vote) → identical; Bagging trees reproduce `RandomState(42).randint(0,N,N)` sequence
- [x] All 50 AdaBoost rounds used for both criteria (no early stop); script runtime ~45 s

## Phase 4 – Part 3: Random Forest (`part3_random_forest.py`)
- [x] `RandomForest(n_estimators=50, criterion, max_features=4)`; fresh `RandomState(42)`
- [x] Bootstrap N rows per tree (same rng)
- [x] Pass `max_features=4, rng=rng` into DecisionTree → new subset **at every node**
- [x] Node becomes leaf if no sampled feature gives positive gain
- [x] Majority vote, ties → smallest label
- [x] Run Gini & Entropy: train, test, derived family acc
- [x] Compare with single tree and Bagging
  - Test: Gini single 94.93 → bagging 96.66 → RF 97.78 · Entropy single 95.41 → bagging 96.87 → RF 97.64
- [x] Extra: average pairwise agreement of tree test predictions, Bagging vs RF
  - Gini: bagging 92.18% vs RF 89.91% · Entropy: bagging 93.20% vs RF 90.26% → RF trees agree less (decorrelated), yet the forest is more accurate ✔
- [x] Verification: each RF tree uses all 22 features overall with different root features (subset drawn per node, not per tree); per-node sampling path matched the brute-force reference in Phase 2; agreement metric matches naive pair loop; results reproducible with seed 42
- [x] Note: one shared `RandomState(42)` per forest drives both bootstraps and per-node `rng.choice(22, 4, replace=False)`; script runtime ~35 s

## Phase 5 – Part 4: Hierarchical (`part4_hierarchical.py`)
- [x] Router: DecisionTree on all train rows → Family
- [x] Experts: partition train by **true** Family; one tree per family → Species; dict `{family_name: expert}`
- [x] Single-species family → expert is a constant predictor (works via Part 1 check (d))
- [x] Handle label spaces: experts predict global species ids (or map local→global)
- [x] `HierarchicalClassifier(router, experts)` with `predict_family(X)` and `predict(X)`
- [x] `predict`: route all → group by predicted family → one expert call per group → write back to original indices
- [x] Metrics per criterion:
  - [x] Router acc (train & test)
  - [x] Overall acc (train & test)
  - [x] Oracle acc (route by true family)
  - [x] pA (router right, expert wrong), pB = 1 − A_router
  - [x] 4×4 router confusion matrix
  - [x] 10×10 species confusion matrix
  - [x] Per-family test accuracy
  - [x] Numerical check A_overall = A_router − pA
- [x] Compare with flat tree (same criterion): species acc vs flat; router acc vs flat derived family acc
- [x] Notes: router vs experts bottleneck? does hierarchy help?
  - **Router is the bottleneck**: pB (router wrong) 4.93% / 4.38% vs pA (expert wrong) 1.25% / 0.97%; oracle 98.26% / 98.33% → router errors cost 4.45 / 3.68 pts
  - **Hierarchy does not help here**: species 93.82 vs flat 94.93 (Gini), 94.65 vs 95.41 (Entropy); router 95.07 / 95.62 is *worse* than flat-derived family 96.25 / 96.80 — a tree trained only on 4 coarse families separates them worse than a species tree's implied families
  - Small families hurt most: Bufonidae router 71.43% (10/14), Dendrobatidae ~88%; their experts are perfect (single species) so all their errors are routing errors
- [x] Sanity: predictions in original order (shuffle test), match per-sample reference, experts only predict own-family species, single-species experts = 1 leaf; no sample with wrong family gets right species
- [x] Experts: Bufonidae 1 node, Dendrobatidae 1 node, Hylidae 125/119 nodes, Leptodactylidae 33/31 nodes (gini/entropy); script ~3 s

## Phase 6 – Part 5: Gaussian NB (`part5_naive_bayes.py`)
- [x] `GaussianNB(var_smoothing=1e-9)` with `fit`, `predict`
- [x] Priors π_c = n_c/N; means; variances with 1/n_c (population)
- [x] ε = 1e-9 · max_j Var_train(x_j) added to every variance
- [x] Log-likelihood via broadcasting (n × K × d), log space
- [x] Sanity check: {0,2} vs {10,12} → x=3 → 0, x=9 → 1
- [x] Train acc, test acc, derived family acc
- [x] 22×22 correlation matrix of training features (numpy `corrcoef`)
- [x] Notes: effect of MFCC correlation on NB; comparison with trees
  - Correlation: mean off-diagonal |r| 0.30, max 0.91; 43/231 pairs |r|>0.5, 14 pairs >0.7. Strongest: MFCC13–15 (−0.91), 15–17 (−0.88), 11–13 (−0.87), 9–11 (−0.85), 12–14 (−0.84) — mostly neighbouring odd/even coefficients
  - Independence is violated → NB counts correlated evidence several times, giving over-confident, distorted posteriors; axis-aligned per-class Gaussians also can't model the curved/multimodal class shapes
  - NB test 90.34% is the lowest of all models (single tree 94.93–95.41, RF 97.78). Weakest species: HylaMinuta 43.55%, AdenomeraAndre 64.18%; Ameeregatrivittata 100%
  - Train ≈ test (91.47 vs 90.34): high bias, low variance — opposite of the unconstrained tree
- [x] Verification: predictions match a pure-Python loop reference (means, variances, ε, log posterior) on 150 test rows; ε = 2.6e-08; script < 2 s

## Phase 7 – Results matrix (`results_matrix.pdf`)
- [x] Table 2 filled (all rows, % with 2 decimals)
- [x] Hierarchical rows: overall acc in species columns, router acc in "Router:" cell
- [x] NB row: criterion "—"
- [x] Table 3 filled: router acc, oracle acc, pA, pB (test, %)
- [x] Typed (not scanned) → generate PDF via `make_results_pdf.py` (re-runs all models, draws both tables with matplotlib, A4)
- [x] Cross-check every number against script output — all 11 rows + Table 3 identical to Results log; PDF rendered and visually checked
- [ ] Decide: include `make_results_pdf.py` in zip? (optional helper, not required)

## Compliance audit vs problem statement (2026-10-18)
- [x] Only numpy / pandas / stdlib in the 5 part scripts + common.py; no sklearn/scipy/etc. anywhere (grep). matplotlib only in optional `make_results_pdf.py`
- [x] Only the 22 MFCC columns as features; Genus / RecordID never used
- [x] `stratified_split` copied verbatim, seed 42, in `common.py`, same split in all parts (5,756 / 1,439)
- [x] No test-set use in fitting: species→family map and NB ε from training data only
- [x] Table 1 hyperparameters: tree unconstrained, mss=2 · Bagging T=25 · AdaBoost T=50, depth 3, SAMME · RF T=50, m=4 per node · Hier. unconstrained router+experts · NB ε=1e-9·max Var
- [x] One fresh `RandomState(42)` per ensemble; bootstrap via `rng.randint(0,N,size=N)`; RF `rng.choice(d,m,replace=False)` per node
- [x] Midpoint thresholds between distinct values; ties → lower feature, lower threshold; leaf/vote ties → smallest label
- [x] `DecisionTree(criterion, max_depth, min_samples_split, max_features=None, rng=None)`, `fit(X,y,sample_weight=None)`, `predict(X)`; default weights 1/N; all 4 sanity checks
- [x] SAMME exactly as steps 1–6 (clip, early stop at ε ≥ 1−1/K, α = ln((1−ε)/ε)+ln(K−1))
- [x] `RandomForest(n_estimators, criterion, max_features, ...)`
- [x] `HierarchicalClassifier(router, experts)` — **fixed**: previously took an extra `family_names` arg; now derives family order from sorted expert names (results unchanged)
- [x] Experts in dict `{family_name: expert}`, trained on TRUE family; predict routes → groups → one call per expert → original order
- [x] Part 4 prints: router & overall acc (train/test), oracle, pA, pB, 4×4 + 10×10 CMs (numpy), per-family acc, A_overall = A_router − pA check, flat comparison
- [x] `GaussianNB(var_smoothing=1e-9)`, log space, n×K×d broadcasting, sanity check
- [x] Extras done: depth sweep, stumps, pairwise agreement, correlation matrix
- [x] Each script runs with one command from the folder with the CSV, relative path only
- [ ] Remaining (Phase 8): README.txt, fresh-folder run, zip named `<StudentID>_A1.zip`

## Phase 8 – Packaging
- [ ] Each script runs with one command (`python partX_....py`) from folder with CSV, no absolute paths
- [ ] Each script prints the matrix numbers clearly labelled
- [ ] `grep` for prohibited imports (sklearn, scipy, torch, tensorflow, xgboost, lightgbm, statsmodels)
- [ ] Confirm no test-set use in fitting/tuning/statistics
- [ ] `README.txt`: versions + run commands
- [ ] Fresh-folder test: copy files + CSV to a temp dir and run all five scripts
- [ ] Zip as `<StudentID>_A1.zip` (5 scripts + common.py + README.txt + results_matrix.pdf)
- [ ] Submit before 23:55, 26-10-2026

## Phase 9 – Viva prep
- [ ] Explain vectorised split search (cumsum trick) line by line
- [ ] Gini vs entropy; why unconstrained tree overfits (depth sweep)
- [ ] Bagging = variance reduction; RF per-node subsetting decorrelates trees
- [ ] SAMME: role of ln(K−1), stopping rule, why stumps fail at K=10
- [ ] Hierarchical: error propagation, oracle gap, A_overall = A_router − pA
- [ ] NB independence assumption vs correlated MFCCs
- [ ] Leakage caveat (same recordings in train & test)

---

## Results log (fill as numbers come in)

| Method | Criterion | Train % | Test % | Family % (test) |
|---|---|---|---|---|
| Single Tree | Gini | 100.00 | 94.93 | 96.25 |
| Single Tree | Entropy | 100.00 | 95.41 | 96.80 |
| Bagging | Gini | 99.97 | 96.66 | 97.50 |
| Bagging | Entropy | 99.98 | 96.87 | 97.50 |
| AdaBoost | Gini | 97.52 | 95.34 | 96.73 |
| AdaBoost | Entropy | 96.35 | 94.16 | 95.00 |
| Random Forest | Gini | 100.00 | 97.78 | 98.40 |
| Random Forest | Entropy | 100.00 | 97.64 | 98.19 |
| Hierarchical | Gini | 100.00 | 93.82 | Router: 95.07 |
| Hierarchical | Entropy | 100.00 | 94.65 | Router: 95.62 |
| Gaussian NB | — | 91.47 | 90.34 | 91.94 |

| Criterion | Router acc | Oracle acc | (A) pA | (B) pB |
|---|---|---|---|---|
| Gini | 95.07 | 98.26 | 1.25 | 4.93 |
| Entropy | 95.62 | 98.33 | 0.97 | 4.38 |

## Notes / issues log
- 2026-10-16 — Dataset downloaded from UCI; created `.venv` (numpy, pandas, matplotlib). CSV column names contain a space for MFCCs 1–9.
- 2026-10-16 — Phase 1 done: `common.py` written and tested; split gives exactly 5,756 / 1,439.
- 2026-10-16 — Phase 2 done: `part1_decision_tree.py` passes all sanity checks; verified against brute-force reference.
- 2026-10-16 — Phase 3 done: `part2_bagging_boosting.py` (Bagging + AdaBoost SAMME) verified against loop references.
- 2026-10-16 — Phase 4 done: `part3_random_forest.py`; RF best so far, lower tree agreement than bagging.
- 2026-10-16 — Phase 5 done: `part4_hierarchical.py`; A_overall = A_router − pA verified; hierarchy loses to flat tree, router is the bottleneck.
- 2026-10-16 — Phase 6 done: `part5_naive_bayes.py`; all five model scripts complete, results log full.
- 2026-10-16 — Phase 7 done: `results_matrix.pdf` generated from fresh model runs.
- 2026-10-18 — Full compliance audit vs PDF; fixed `HierarchicalClassifier` signature to `(router, experts)`.
- 2026-10-18 — Local git repo initialised on `main` (.venv, __pycache__, zip ignored); first commit made. Waiting for remote URL to push.
- 2026-10-18 — Pushed to https://github.com/eightbits0211/MLassignment (branch `main`).
