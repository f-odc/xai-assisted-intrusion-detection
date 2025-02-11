"""
utilib: A utility library for helpful data transformations.

Functions:
- min_max_normalize: Perform Min-Max normalization on a DataFrame.
- Additional utility functions can be added here.

Usage:
------
import utilib as ut
normalized_df = ut.min_max_normalize(data_frame)
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def min_max_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Min-Max normalization to the features of the given DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing features to be normalized.

    Returns:
    pd.DataFrame: A DataFrame with normalized features.
    """
    print("Min-Max Normalization....")
    min_max_scaler = MinMaxScaler()
    normalized_df = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)
    print(f"Normalized DataFrame Shape: {normalized_df.shape}")
    return normalized_df