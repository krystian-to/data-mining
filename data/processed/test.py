from pathlib import Path
import pandas as pd
import numpy as np


PROCESSED_DIR = Path(__file__).resolve().parent

FILES = [
    "X_train.pkl", "X_val.pkl", "X_test.pkl",
    "y_train.pkl", "y_val.pkl", "y_test.pkl"
]


def _load(name: str):
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Brak pliku: {path}")
    return pd.read_pickle(path)


def _shape(x):
    # scipy sparse ma .shape, ale len(x) wywala błąd
    if hasattr(x, "shape"):
        return x.shape
    return (len(x),)


def _is_sparse(x):
    # bez scipy importu: sprawdzamy po atrybucie
    return hasattr(x, "tocsr") or hasattr(x, "toarray")


def main():
    print(f"== Sanity check: {PROCESSED_DIR} ==")

    # load
    X_train = _load("X_train.pkl")
    X_val = _load("X_val.pkl")
    X_test = _load("X_test.pkl")
    y_train = _load("y_train.pkl")
    y_val = _load("y_val.pkl")
    y_test = _load("y_test.pkl")

    # basic info
    print("\n[1] Shapes")
    print("X_train:", _shape(X_train), "| y_train:", _shape(y_train))
    print("X_val  :", _shape(X_val),   "| y_val  :", _shape(y_val))
    print("X_test :", _shape(X_test),  "| y_test :", _shape(y_test))

    # row count consistency
    print("\n[2] Row count consistency")
    ntr, nva, nte = _shape(X_train)[0], _shape(X_val)[0], _shape(X_test)[0]
    assert len(y_train) == ntr, "y_train nie pasuje do X_train"
    assert len(y_val) == nva, "y_val nie pasuje do X_val"
    assert len(y_test) == nte, "y_test nie pasuje do X_test"
    print("OK")

    # y sanity
    print("\n[3] y sanity (binary, NaNs, balance)")
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        y_arr = np.asarray(y)
        uniq = np.unique(y_arr[~pd.isna(y_arr)])
        print(
            f"{name}: unique={uniq}, nan={pd.isna(y_arr).sum()}, mean={np.mean(y_arr):.4f}")
        assert set(uniq).issubset(
            {0, 1}), f"{name}: y ma wartości spoza {{0,1}}"

    # X sanity
    print("\n[4] X sanity (type, NaNs if dense)")
    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        print(f"{name}: type={type(X)}, sparse={_is_sparse(X)}")
        # jeśli to jest gęste (np. numpy), sprawdź NaNy
        if not _is_sparse(X) and hasattr(X, "__array__"):
            X_arr = np.asarray(X)
            n_nan = np.isnan(X_arr).sum() if np.issubdtype(
                X_arr.dtype, np.number) else 0
            print(f"  dense NaNs: {n_nan}")

    # simple "not identical" check (hash a few rows)
    print("\n[5] Simple split difference check")

    def head_fingerprint(X, k=5):
        if _is_sparse(X):
            A = X[:k].toarray()
        else:
            A = np.asarray(X[:k])
        return hash(A.tobytes())

    fp_tr = head_fingerprint(X_train)
    fp_va = head_fingerprint(X_val)
    fp_te = head_fingerprint(X_test)
    print("fingerprints:", fp_tr, fp_va, fp_te)
    assert len({fp_tr, fp_va, fp_te}
               ) == 3, "Wygląda jakby splity były identyczne (podejrzane)"
    print("OK")

    print("\n✅ Wszystko wygląda sensownie. PKL mają poprawne dane.")


if __name__ == "__main__":
    main()
