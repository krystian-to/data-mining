import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.config import RAW_DATA, PROCESSED_DIR, TARGET, RANDOM_STATE, TEST_SIZE, VAL_SIZE


def load_raw():
    return pd.read_csv(RAW_DATA)


def preprocess(df: pd.DataFrame):
    y = df[TARGET].map({"Yes": 1, "No": 0})
    X = df.drop(columns=[TARGET])

    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    return X_processed, y, preprocessor


def make_splits(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_train
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def save_processed(X_train, X_val, X_test, y_train, y_val, y_test):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pd.to_pickle(X_train, PROCESSED_DIR / "X_train.pkl")
    pd.to_pickle(X_val, PROCESSED_DIR / "X_val.pkl")
    pd.to_pickle(X_test, PROCESSED_DIR / "X_test.pkl")
    pd.to_pickle(y_train, PROCESSED_DIR / "y_train.pkl")
    pd.to_pickle(y_val, PROCESSED_DIR / "y_val.pkl")
    pd.to_pickle(y_test, PROCESSED_DIR / "y_test.pkl")
