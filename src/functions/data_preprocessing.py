"""
data_preprocessing: A module for preprocessing raw CICIDS2017 data.

Functions:
- preprocess: Preprocesses raw CICIDS2017 data to create a well-suited dataset for machine learning models.
- combine_all_data_files: Combines all CSV files in the specified directory into a single DataFrame.
- remove_nan_values: Removes NaN and Infinity values from the DataFrame.
- extract_labels: Extracts given labels from the DataFrame.
- sample_balanced_data: Samples a given number of rows from each label class to create a balanced dataset.
- split_label_and_features: Splits the DataFrame into features and labels.
- binary_one_hot_label_encoding: Encodes the labels as binary one-hot values.
- multi_class_one_hot_label_encoding: Encodes the labels as multi-class values.
- remove_irrelevant_features: Removes irrelevant features that contain only zeros from the DataFrame.
- min_max_normalization: Normalizes the features using Min-Max Normalization.
- standardization_normalization: Normalizes the features using Standardization.
- combine_label_and_features: Combines the labels and features into a single DataFrame.

Usage:
------
import data_preprocessing as dp
preprocessed_data = dp.preprocess(encoding_type=0, norm_type=0)

"""

import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess(encoding_type: int, norm_type: int, label_names=None, sample_size=None):
    """
    Preprocesses raw CICIDS2017 data to create a well-suited dataset for machine learning models.

    Args:
        encoding_type (int): Specify the encoding method, 0 = binary one-hot encoding, 1 = multi-class one-hot encoding.
        norm_type (int): Specify the normalization method, 0 = Min-Max Normalization, 1 = Standardization.
        label_names (Array, optional): A array of column names to extract as labels e.g. ['BENIGN', 'DDoS']. If None, all labels will be used. Defaults to None.
        sample_size (int, optional): The size of the sample to take from each class. If None, all data will be used. Defaults to None.

    Returns:
        DataFrame: A DataFrame containing the preprocessed labels.
        DataFrame: A DataFrame containing the preprocessed features.
    """
    # Combine data files
    df = combine_all_data_files()
    # Remove NaN and Infinity values
    df = remove_nan_values(df)
    # Extract unwante labels
    if label_names != None:
        df = extract_labels(df, label_names)
    # TODO: Sample data
    if sample_size != None:
        df = sample_balanced_data(df, sample_size)
    # Split data into labels and features
    label_df, feature_df = split_label_and_features(df)
    # Encode data
    if encoding_type == 0: # binary-encoding
        label_df = binary_one_hot_label_encoding(label_df)
    elif encoding_type == 1: # multi-class-encoding
        label_df = multi_class_one_hot_label_encoding(label_df)
    else: raise ValueError("Invalid encoding type")
    # Remove irrelevant features
    feature_df = remove_irrelevant_features(feature_df)
    # Normalize data
    if norm_type == 0: # min-max-normalization
        feature_df = min_max_normalization(feature_df)
    elif norm_type == 1: # standardization
        feature_df = standardization_normalization(feature_df)
    else: raise ValueError("Invalid normalization type")

    return label_df, feature_df

def combine_all_data_files():
    """
    Combines all CSV files in the specified directory into a single DataFrame.

    Returns:
        DataFrame: A pandas DataFrame containing all combined data from CSV files.
    """
    print("--- Combining all CICIDS2017 files ---")
    # combine all CICIDS2017 files
    path = '../../CICIDS2017/raw/'
    combined_df = pd.DataFrame()
    for file in os.listdir(path):
        if file.endswith('.csv'):
            print(file)
            df = pd.read_csv(path + file)
            combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

def remove_nan_values(df):
    """
    Removes NaN and Infinity values from the DataFrame.

    Args:
        df (DataFrame): The DataFrame from which to remove NaN and Infinity values.

    Returns:
        DataFrame: The DataFrame with NaN and Infinity values removed.
    """
    print("--- Removing NaN and Infinity values ---")
    # print number of rows with NaN values
    print("Number of rows with NaN values: ", df.isnull().sum().sum())
    print("Removing NaN values....")
    # remove NaN values
    df.dropna(inplace=True)

    # print number of rows with Infinity values
    inf_rows = df.isin([np.inf, -np.inf]).any(axis=1)
    print(f"Number of rows with Infinity values: {inf_rows.sum()}")
    print("Removing Infinity values....")
    # remove Infinity values
    df = df[~inf_rows]
    return df


# TODO: remove default None parameter
def extract_labels(df, label_names=None):
    """
    Extracts given labels from the DataFrame.

    Args:
        df (DataFrame): The DataFrame from which to extract labels.
        label_names (Array, optional): A array of column names to extract as labels e.g. ['BENIGN', 'DDoS']. If None, the given DataFrame will be returned. Defaults to None.

    Returns:
        DataFrame: A DataFrame containing the extracted labels.
    """
    print("--- Extracting labels ---")
    if label_names is None:
        return df
    
    extract_df = df[df[' Label'].isin(label_names)]
    print(extract_df[' Label'].value_counts())
    return extract_df


def sample_balanced_data(df, sample_size):
    """
    Samples a given number of rows from each label class to create a balanced dataset.

    Args:
        df (DataFrame): The DataFrame from which to sample balanced data.
        sample_size (int): The size of the sample to take from each class.

    Returns:
        DataFrame: A DataFrame containing the sampled balanced data.
    """
    print("--- Sampling balanced data ---")
    # for each label in label_names sample sample_size rows
    df = df.groupby(' Label').apply(lambda x: x.iloc[:sample_size])
    print(f"Sample to shape: {df.shape}")
    return df


def split_label_and_features(df):
    """
    Splits the DataFrame into features and labels.

    Args:
        df (DataFrame): The DataFrame to split into features and labels.

    Returns:
        DataFrame: A DataFrame containing the labels.
        DataFrame: A DataFrame containing the features.
    """
    print("--- Splitting labels and features ---")
    feature_df = df.drop(columns=[' Label'])
    label_df = df[' Label']
    return label_df, feature_df


def binary_one_hot_label_encoding(df):
    """
    Encodes the labels as binary one-hot values.

    Args:
        df (DataFrame): The DataFrame containing only the labels to encode.

    Returns:
        DataFrame: The DataFrame with encoded labels. 'BENIGN' is [1, 0] and 'ATTACK' is [0, 1].
    """
    print("--- Encoding labels as binary one-hot values ---")
    binary_label_df = df.apply(lambda x: 0 if x == 'BENIGN' else 1)
    binary_one_hot_label_df = pd.get_dummies(binary_label_df)
    binary_one_hot_label_df.columns = ['BENIGN', 'ATTACK']
    return binary_one_hot_label_df


def multi_class_one_hot_label_encoding(df):
    """
    Encodes the labels as multi-class values.

    Args:
        df (DataFrame): The DataFrame containing only the labels to encode.

    Returns:
        DataFrame: The DataFrame with encoded labels. 0 for 'BENIGN', 1-X for each attack type.
    """
    multiclass_label_df = pd.get_dummies(df)
    return multiclass_label_df


def remove_irrelevant_features(feature_df: pd.DataFrame):
    """
    Removes irrelevant features that contain only zeros from the DataFrame.

    Args:
        feature_df (DataFrame): The DataFrame from which to remove irrelevant features.

    Returns:
        DataFrame: The DataFrame with irrelevant features removed.
    """
    print("--- Removing irrelevant features ---")
    zero_columns = feature_df.columns[(feature_df.sum() == 0)]
    print(f"Removed Zero Columns: {zero_columns.tolist()}")
    # drop columns with only 0 values
    feature_df.drop(columns=zero_columns, inplace=True)
    return feature_df


def min_max_normalization(feature_df: pd.DataFrame):
    """
    Normalizes the features using Min-Max Normalization.

    Args:
        feature_df (DataFrame): The DataFrame containing the features to normalize.

    Returns:
        DataFrame: The DataFrame with normalized features.
    """
    print("--- Normalizing features using Min-Max Normalization ---")
    min_max_scaler = MinMaxScaler()
    feature_df_normalized = pd.DataFrame(min_max_scaler.fit_transform(feature_df), columns=feature_df.columns, index=feature_df.index)
    return feature_df_normalized


def standardization_normalization(feature_df: pd.DataFrame):
    """
    Normalizes the features using Standardization.

    Args:
        feature_df (DataFrame): The DataFrame containing the features to normalize.

    Returns:
        DataFrame: The DataFrame with normalized features.
    """
    print("--- Normalizing features using Standardization ---")
    standard_scaler = StandardScaler()
    feature_df_standardized = pd.DataFrame(standard_scaler.fit_transform(feature_df), columns=feature_df.columns, index=feature_df.index)
    return feature_df_standardized


def combine_label_and_features(label_df, feature_df: pd.DataFrame):
    """
    Combines the labels and features into a single DataFrame.

    Args:
        label_df (DataFrame): The DataFrame containing the labels.
        feature_df (DataFrame): The DataFrame containing the features.

    Returns:
        DataFrame: A DataFrame containing the labels and features.
    """
    print("--- Combining labels and features ---")
    combined_df = pd.concat([label_df, feature_df], axis=1)
    return combined_df