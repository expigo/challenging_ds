from pathlib import Path
from enum import Enum

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_PATH = DATA_DIR / "train.parquet"
TEST_DATA_PATH = DATA_DIR / "test.parquet"

OUTPUT_DIR = PROJECT_ROOT / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_DIR = OUTPUT_DIR / "results"
PROCESSED_DIR = OUTPUT_DIR / "processed_data"
PIPELINE_DIR = OUTPUT_DIR / "preprocessors"

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

class Color(Enum):
    """
    Standard color palette for all visualizations.
    """
    BLACK = "black"
    RED = "orangered"        # Used for negative outcomes, warnings, defaults
    GREEN = "limegreen"      # Used for positive outcomes, solvent customers
    BLUE = "cornflowerblue"  # Used for neutral information, presence
    GRAY = "darkgray"        # Used for background elements, missing data
    YELLOW = "gold"          # Used for highlights, attention

    @classmethod
    def get_default_palette(cls):
        """Returns a list of colors suitable for categorical data."""
        return [cls.BLUE.value, cls.RED.value, cls.GREEN.value,
                cls.YELLOW.value, cls.GRAY.value]

    @classmethod
    def get_binary_palette(cls):
        """Returns colors for binary outcomes."""
        return [cls.GREEN.value, cls.RED.value]
