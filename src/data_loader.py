import pandas as pd
from typing import Tuple

def load_train_data() -> pd.DataFrame:
    print("Loading training data")
    train_df = pd.read_parquet('./data/train.parquet')
    print(train_df.shape)
    return train_df


