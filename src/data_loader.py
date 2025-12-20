import pandas as pd
from typing import Tuple

from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the training and test datasets.

    Returns:
        Tuple[pd.DataFrame, pd.Dataframe]: (train_df, test_df)

    Raises:
        FileNotFoundError: THe data not found
    """
    try:
        print(f"Loading training data from {TRAIN_DATA_PATH}")
        train_df = pd.read_parquet(TRAIN_DATA_PATH)

        print(f"Loading test data from {TEST_DATA_PATH}")
        test_df = pd.read_parquet(TEST_DATA_PATH)

        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")

    except FileNotFoundError as e:
        print(f"Error laoding data: {e}")
        print(f"Please ensure that .parquet files are in the data/ dir")

    return train_df, test_df


