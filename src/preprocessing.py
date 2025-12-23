"""
Preprocessing pipelines for credit default prediction.

This module provides two preprocessing pipelines:
- Pipeline A: Excludes categorical features based on EDA findings
- Pipeline B (Full): Includes all features with one-hot encoding for comparison

Based on Phase 1 EDA:
- FavoriteColor: Chi-square p=0.879 (no predictive value)
- Hobby: Chi-square p=0.636 (no predictive value)
- CreditScore missingness: seems like MCAR pattern (p=0.12)
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

from src.config import (
    NUMERICAL_FEATURES,
    ORDINAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_FEATURE
)


def fix_negative_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix negative values in Income and LoanAmount.

    Args:
        df: DataFrame with potential negative values

    Returns:
        DataFrame with abs values for Income and LoanAmount
    """

    df = df.copy()
    df["Income"] = df["Income"].abs()
    df["LoanAmount"] = df["LoanAmount"].abs()
    return df


def get_pipeline_A() -> Pipeline:
    """
    Create efficient piepline (no redundant features based on EDA)

    Features: 7 numerical + 1 ordinal
    Excluded: FavoriteCOlor, Hobby

    Processing steps:
    1. Numerical features: Median imputation + Standard Scaler
    2. Oridnal features: Most freq imputation, no scaling

    Returns:
        sklearn Pipeline obj
    """

    # numerical features
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # ordinal
    ordinal_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="most_frequent"))
    ])

    # combine transforms
    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, NUMERICAL_FEATURES),
        ('ordinal', ordinal_pipeline, ORDINAL_FEATURES),
    ], remainder='drop') # drop categorical

    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    return pipeline


if __name__ == "__main__":
    from src.data_loader import load_data

    print(">> testing module...")
    train_df, test_df = load_data()
    
    train_df = fix_negative_values(train_df)
    test_df = fix_negative_values(test_df)

    # separate features and target
    X_train = train_df.drop(columns=[TARGET_FEATURE])
    Y_train = train_df[TARGET_FEATURE].values

    X_test = test_df.drop(columns=[TARGET_FEATURE])
    Y_test = test_df[TARGET_FEATURE].values

    # test pipeline A
    print(">> testing pipeline A...")
    pipeline = get_pipeline_A()
    pipeline.fit(X_train)

    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    print("All good!")


                            
