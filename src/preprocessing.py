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
import pickle
from pathlib import Path
from typing import Tuple

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

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


def get_pipeline_B() -> Pipeline:
    """
    Create full piepline (full features with one-hot encoding)

    Features: 7 numerical + 1 ordinal + 6 one-hot

    Processing steps:
    1. Numerical features: Median imputation + Standard Scaler
    2. Oridnal features: Most freq imputation, no scaling
    3. Categorical features: One-hot encoding (drop first category)

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

    categorical_pipeline = Pipeline([
        ('onehot', OneHotEncoder(
            drop='first',
            sparse_output=False,
            handle_unknown='ignore'
        ))
    ])

    # combine transforms
    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipeline, NUMERICAL_FEATURES),
        ('ordinal', ordinal_pipeline, ORDINAL_FEATURES),
        ('categorical', categorical_pipeline, CATEGORICAL_FEATURES),
    ], remainder='drop') # drop categorical

    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])

    return pipeline


def preprocess_data(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        pipeline_type: str = 'A'
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete preprocessing pipeline.

    Args:
        train_df: train DataFrame
        test_df: test DataFrame
        pipeline_type: 'A' or 'B' (full)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    
    # fix data quality issues
    train_df = fix_negative_values(train_df)
    test_df = fix_negative_values(test_df)

    # separate features and target
    X_train = train_df.drop(columns=[TARGET_FEATURE])
    y_train = train_df[TARGET_FEATURE].values

    X_test = test_df.drop(columns=[TARGET_FEATURE])
    y_test = test_df[TARGET_FEATURE].values

    # Get and fit pipeline
    if pipeline_type == 'A':
        pipeline = get_pipeline_A()
        pipeline.fit(X_train)
    elif pipeline_type == 'B':
        pipeline = get_pipeline_B()
        pipeline.fit(X_train)
    else:
        raise ValueError("pipeline_type must be 'A' or 'B'")

    # transform
    X_train_processed = pipeline.transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test

def save_pipeline(pipeline: Pipeline, filepath: Path) -> None:
    """
    Save fitted pipeline to disk.
    
    Args:
        pipeline: Fitted sklearn Pipeline
        filepath: Path to save pickle file
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"✓ Pipeline saved to {filepath}")


def load_pipeline(filepath: Path) -> Pipeline:
    """
    Load fitted pipeline from disk.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Fitted sklearn Pipeline
    """
    with open(filepath, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline


if __name__ == "__main__":
    from src.data_loader import load_data

    print(">> testing module...")
    train_df, test_df = load_data()
    
    # test pipeline A
    print("\n>> testing pipeline A...")
    X_train_A, X_test_A, y_train, y_test = preprocess_data(
            train_df, test_df, pipeline_type='A'
    )

    print(f"Out shape: {X_train_A.shape}")
    print(f"features: {X_test_A.shape}")


    # test pipeline B
    print("\n>> testing pipeline B...")
    X_train_B, X_test_B, y_train, y_test = preprocess_data(
            train_df, test_df, pipeline_type='B'
    )

    print(f"Out shape: {X_train_B.shape}")
    print(f"features: {X_test_B.shape}")

    print("\nAll good!")


