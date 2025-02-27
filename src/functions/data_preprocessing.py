"""
A module for preprocessing raw CICIDS2017 data to create a well-suited dataset for machine learning models.

Functions:
- build_dataset: Builds a dataset from the CICIDS2017 data. Include only the specified labels.
- generate_normalizer: Generates a normalizer for the given DataFrame.
- preprocess_data: Preprocess a dataset to create a well-suited dataset for machine learning models.
- combine_all_data_files: Combines all CSV files in the specified directory into a single DataFrame.
- remove_nan_values: Removes NaN and Infinity values from the DataFrame.
- extract_labels: Extracts given labels from the DataFrame.
- sample_balanced_data: Samples a given number of rows from each label class to create a balanced dataset.
- split_label_and_features: Splits the DataFrame into features and labels.
- binary_one_hot_label_encoding: Encodes the labels as binary one-hot values.
- multi_class_one_hot_label_encoding: Encodes the labels as multi-class values.
- get_irrelevant_features: Gets irrelevant features that contain only zeros from the DataFrame.
- normalization: Normalizes the features using a specified normalization technique.
- generate_min_max_normalizer: Generates a Min-Max normalizer fitted on the given DataFrame.
- generate_standardizazion_normalizer: Generates a Standardization normalizer fitted on the given DataFrame.

Usage:
------
>>> import data_preprocessing as dp
>>> df = dp.build_dataset(label_names=['BENIGN', 'DDoS'])
>>> normalizer, zero_columns = dp.generate_normalizer(df, norm_type=0)
>>> X, y = dp.preprocess_data(df, encoding_type=0, normalizer, zero_columns)

"""


import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def build_dataset(label_names=None):
    """
    Builds a dataset from the CICIDS2017 data. Include only the specified labels.
    
    Args:
        label_names (Array, optional): A array of column names to extract as labels e.g. ['BENIGN', 'DDoS']. If None, all labels will be used. Defaults to None.
    
    Returns:
        DataFrame: The generated CICIDS2017 dataset.
    """
    print("-- Building CICIDS2017 dataset --")
    # Combine data files
    df = combine_all_data_files()
    # Remove NaN and Infinity values
    df = remove_nan_values(df)
    # Extract unwante labels
    if label_names != None:
        df = extract_labels(df, label_names)

    return df


def generate_normalizer(df:pd.DataFrame, norm_type: int):
    """
    Generates a normalizer for the given DataFrame.

    Args:
        df (DataFrame): The DataFrame to generate a normalizer for.
        norm_type (int): Specify the normalization method, 0 = Min-Max Normalization, 1 = Standardization.

    Returns:
        TransformerMixin: A scikit-learn transformer (e.g., MinMaxScaler, StandardScaler, Normalizer).
    """
    print("-- Generating normalizer --")
    # Split data into labels and features
    _, feature_df = split_label_and_features(df)
    # Remove irrelevant features
    zero_columns = get_irrelevant_features(feature_df)
    feature_df.drop(columns=zero_columns, inplace=True)
    if norm_type == 0: # min-max-normalization
        return generate_min_max_normalizer(feature_df), zero_columns
    elif norm_type == 1: # standardization
        return generate_standardizazion_normalizer(feature_df), zero_columns
    else: raise ValueError("Invalid normalization type")


def preprocess_data(df:pd.DataFrame, encoding_type: int, normalizer, zero_columns, sample_size=None, random_sample_state=1503):
    """
    Preprocess a dataset to create a well-suited dataset for machine learning models.
    
    Args:
        df (DataFrame): The DataFrame to preprocess.
        encoding_type (int): Specify the encoding method, 0 = binary one-hot encoding, 1 = multi-class one-hot encoding.
        normalizer: A scikit-learn transformer (e.g., MinMaxScaler, StandardScaler, Normalizer).
        zero_columns (List): A list of irrelevant features that contain only zeros.
        sample_size (int, optional): The size of the sample to take from each class. If None, all data will be used. Defaults to None.
        random_sample_state (int, optional): The random state to use for sampling. Defaults to 1503.
    
    Returns:
        DataFrame: Feature DataFrame
        DataFrame: Label DataFrame
    """
    print("-- Preprocessing data --")
    # Sample data
    if sample_size != None:
        df = sample_balanced_data(df, sample_size, random_sample_state)
    # Split data into labels and features
    label_df, feature_df = split_label_and_features(df)
    # Remove irrelevant features
    feature_df.drop(columns=zero_columns, inplace=True)
    # Encode data
    if encoding_type == 0: # binary-encoding
        label_df = binary_one_hot_label_encoding(label_df)
    elif encoding_type == 1: # multi-class-encoding
        label_df = multi_class_one_hot_label_encoding(label_df)
    else: raise ValueError("Invalid encoding type")
    # Normalize data
    feature_df = normalization(normalizer, feature_df)
    return feature_df, label_df


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
    print(f"Removing {df.isnull().sum().sum()} Rows with NaN values")
    # remove NaN values
    df.dropna(inplace=True)

    # print number of rows with Infinity values
    inf_rows = df.isin([np.inf, -np.inf]).any(axis=1)
    print(f"Removing {inf_rows.sum()} Rows with Infinity values")
    # remove Infinity values
    df = df[~inf_rows]
    return df


def extract_labels(df, label_names):
    """
    Extracts given labels from the DataFrame.

    Args:
        df (DataFrame): The DataFrame from which to extract labels.
        label_names (Array): A array of column names to extract as labels e.g. ['BENIGN', 'DDoS'].

    Returns:
        DataFrame: A DataFrame containing the extracted labels.
    """
    print("--- Extracting labels ---")
    extract_df = df[df[' Label'].isin(label_names)]
    print(extract_df[' Label'].value_counts())
    return extract_df


def sample_balanced_data(df, sample_size, random_state):
    """
    Samples a given number of rows from each label class to create a balanced dataset.

    Args:
        df (DataFrame): The DataFrame from which to sample balanced data.
        sample_size (int): The size of the sample to take from each class.
        random_state (int): The random state to use for sampling.

    Returns:
        DataFrame: A DataFrame containing the sampled balanced data.
    """
    print("--- Sampling balanced data ---")
    # for each label in label_names sample sample_size rows
    df = df.groupby(' Label', group_keys=False).apply(lambda x: x.sample(sample_size, random_state=random_state)).reset_index(drop=True)
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


def get_irrelevant_features(feature_df: pd.DataFrame):
    """
    Gets irrelevant features that contain only zeros from the DataFrame.

    Args:
        feature_df (DataFrame): The DataFrame from which to get irrelevant features.

    Returns:
        List: A list of irrelevant features that contain only zeros.
    """
    zero_columns = feature_df.columns[(feature_df.sum() == 0)]
    print(f"Zero Columns: {zero_columns.tolist()}")
    return zero_columns


def normalization(normalizer, feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the features using a specified normalization technique.

    Args:
        normalizer: A fitted scikit-learn normalizer (e.g., MinMaxScaler, StandardScaler, RobustScaler, Normalizer).
        feature_df (pd.DataFrame): The DataFrame containing the features to normalize.

    Returns:
        DataFrame: The DataFrame with normalized features.
    """
    print(f"--- Normalizing features using {type(normalizer).__name__} ---")
    feature_df_normalized = pd.DataFrame(normalizer.transform(feature_df), columns=feature_df.columns, index=feature_df.index)
    return feature_df_normalized


def generate_min_max_normalizer(df:pd.DataFrame) -> MinMaxScaler:
    """
    Generates a Min-Max normalizer fitted on the given DataFrame.
    
    Args:
        df (DataFrame): The DataFrame to generate a Min-Max normalizer for.
    
    Returns:
        MinMaxScaler: A fitted MinMaxScaler.
    """
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(df)
    return min_max_scaler


def generate_standardizazion_normalizer(df:pd.DataFrame) -> StandardScaler:
    """
    Generates a Standardization normalizer fitted on the given DataFrame.
    
    Args:
        df (DataFrame): The DataFrame to generate a StandardScaler.
    
    Returns:
        MinMaxScaler: A fitted StandardScaler.
    """
    standard_scaler = StandardScaler()
    standard_scaler.fit(df)
    return standard_scaler