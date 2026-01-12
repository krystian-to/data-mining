import pandas as pd
from src.config import PROCESSED_DIR, RESULTS_DIR


def load_data():
    return (
        pd.read_pickle(PROCESSED_DIR / "X_train.pkl"),
        pd.read_pickle(PROCESSED_DIR / "X_val.pkl"),
        pd.read_pickle(PROCESSED_DIR / "X_test.pkl"),
        pd.read_pickle(PROCESSED_DIR / "y_train.pkl"),
        pd.read_pickle(PROCESSED_DIR / "y_val.pkl"),
        pd.read_pickle(PROCESSED_DIR / "y_test.pkl"),
    )


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    leaderboard = []

    # from src.models.decision_tree import run
    # leaderboard.append(run())

    pd.DataFrame(leaderboard).to_csv(
        RESULTS_DIR / "leaderboard.csv", index=False
    )


if __name__ == "__main__":
    main()
