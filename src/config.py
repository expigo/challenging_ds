from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_PATH = DATA_DIR / "train.parquet"
TEST_DATA_PATH = DATA_DIR / "test.parquet"

OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"

RANDOM_SEED = 42

# Feature definitions based on task description
NUMERICAL_FEATURES = [
        'Age', 'Income', 'CreditScore', 'LoanAmount',
        'EmploymentYears', 'NumDependents', 'DebtToIncome'
]

ORDINAL_FEATURES = ['EducationLevel']

CATEGORICAL_FEATURES = ['FavoriteColor', 'Hobby']

TARGET_FEATURE = 'Default'

ALL_FEATURES = NUMERICAL_FEATURES + ORDINAL_FEATURES + CATEGORICAL_FEATURES
