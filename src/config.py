from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
RAW_DATA = DATA_DIR / "raw" / "telecom_churn.csv"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"

TARGET = "Churn"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
