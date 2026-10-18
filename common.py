"""Shared utilities for BITS F464 Assignment 1 (numpy / pandas only)."""
import numpy as np
import pandas as pd

DATA_FILE = "Frogs_MFCCs.csv"


def load_data(path=DATA_FILE):
    df = pd.read_csv(path)
    # CSV names are 'MFCCs_ 1'..'MFCCs_ 9', 'MFCCs_10'..'MFCCs_22'
    feat_cols = [c for c in df.columns if c.startswith("MFCCs_")]
    X = df[feat_cols].to_numpy(dtype=float)
    species_names, y_species = np.unique(df["Species"].to_numpy(dtype=str), return_inverse=True)
    family_names, y_family = np.unique(df["Family"].to_numpy(dtype=str), return_inverse=True)
    return X, y_species, y_family, species_names, family_names


def stratified_split(labels, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    test = []
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        test.extend(idx[:int(round(test_frac * len(idx)))])
    test = np.sort(np.array(test))
    train = np.setdiff1d(np.arange(len(labels)), test)
    return train, test


def get_split():
    X, ys, yf, sn, fn = load_data()
    tr, te = stratified_split(ys, 0.2, 42)
    return {
        "X_train": X[tr], "X_test": X[te],
        "ys_train": ys[tr], "ys_test": ys[te],
        "yf_train": yf[tr], "yf_test": yf[te],
        "species_names": sn, "family_names": fn,
        "s2f": species_to_family(ys[tr], yf[tr], len(sn)),
    }


def species_to_family(y_species, y_family, n_species):
    m = np.full(n_species, -1, dtype=int)
    m[y_species] = y_family
    return m


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def majority_vote(preds, n_classes):
    preds = np.asarray(preds)
    counts = np.zeros((n_classes, preds.shape[1]), dtype=int)
    for p in preds:  # loop over models, not samples
        counts[p, np.arange(preds.shape[1])] += 1
    return np.argmax(counts, axis=0)  # argmax returns first max = smallest label


def derived_family_accuracy(species_pred, family_true, s2f):
    return accuracy(family_true, s2f[np.asarray(species_pred)])


def pct(a):
    return f"{100 * a:.2f}%"


if __name__ == "__main__":
    d = get_split()
    print("train:", len(d["ys_train"]), "test:", len(d["ys_test"]))
    print("species->family:")
    for i, s in enumerate(d["species_names"]):
        print(f"  {i} {s:25s} -> {d['family_names'][d['s2f'][i]]}")
