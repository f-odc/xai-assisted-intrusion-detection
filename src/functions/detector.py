"""
detector: This module contains functions to build and train a deep neural network to detect adversarial attacks.

Functions:
- create_min_max_normalizer: Creates a Min-Max normalizer fitted to the given DataFrame.
- normalize_shap_values: Normalizes SHAP values using Min-Max normalization.
- build_train_datasets: Build feature dataset and label dataset.
- build_detector: Builds and trains a deep neural network to detect adversarial attacks. Evaluate the model using the test data.
- create_dnn: Creates a deep neural network model.
- evaluate_model: Evaluates the model using the predicted and actual labels.

Usage:
------
>>> import detector as det
>>> min_max_normalizer = det.create_min_max_normalizer(df)
"""


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import setuptools.dist # needed to avoid error
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def create_min_max_normalizer(df: pd.DataFrame):
    """
    Creates a Min-Max normalizer fitted to the given DataFrame.

    Args:
        df (pd.DataFrame): Features DataFrame to fit the normalizer

    Returns:
        MinMaxScaler: A MinMaxScaler object.
    """
    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(df)
    return min_max_scaler


def normalize_shap_values(df: pd.DataFrame):
    """
    Normalizes SHAP values using Min-Max normalization. Max is 0.1 and min is -0.1.

    Args:
        df (pd.DataFrame): SHAP values DataFrame to normalize.

    Returns:
        pd.DataFrame: A DataFrame with normalized SHAP values.
    """
    scaler = MinMaxScaler()
    num_features = df.shape[1]
    # fit the scaler
    scaler.fit([[-0.1] * num_features, [0.1] * num_features])
    return scaler.transform(df)


def build_train_datasets(shap_values:pd.DataFrame, adv_shap_values:pd.DataFrame):
    """
    Build feature dataset and label dataset. Features include SHAP values generated from normal and adversarial samples. Labels include incicator whether the sample is normal or adversarial ([1, 0] for BENIGN, [0, 1] for ADVERSARIAL).

    Args:
        shap_values (pd.DataFrame): SHAP values generated from normal samples.
        adv_shap_values (pd.DataFrame): SHAP values generated from adversarial samples.

    Returns:
        pd.DataFrame: Feature dataset.
        pd.DataFrame: Label dataset.
    """
    y_normal = np.array([[1, 0]] * shap_values.shape[0])  
    y_adv = np.array([[0, 1]] * (adv_shap_values.shape[0]))
    y = np.concatenate([y_normal, y_adv])
    y = pd.DataFrame(y, columns=['BENIGN', 'ADVERSARIAL'])

    X = pd.concat([shap_values, adv_shap_values])
    return X, y


def build_detector(X_train, y_train, X_test, y_test):
    """
    Builds and trains a deep neural network to detect adversarial attacks. Evaluate the model using the test data.

    Args:
        X_train (DataFrame): The training features.
        y_train (DataFrame): The training labels.
        X_test (DataFrame): The test features.
        y_test (DataFrame): The test labels.

    Returns:
        Sequential: The trained Keras sequential model.
    """
    # Create
    model = create_dnn(X_train, y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # Train
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=100)
    # Predict
    y_pred = model.predict(X_test)
    # Evaluate
    evaluate_model(y_pred, y_test)
    return model


def create_dnn(X_train:pd.DataFrame, y_train):
    """
    Creates a deep neural network model.
    
    Args:
        X_train (DataFrame): The training features.
        y_train (DataFrame): The training labels.
    
    Returns:
        Sequential: A compiled Keras sequential model.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),  # Define input shape explicitly
        keras.layers.Dense(50, activation='relu'),  # Hidden layer
        keras.layers.Dense(30, activation='relu'),  # Hidden layer
        keras.layers.Dropout(0.2), # TODO: test
        keras.layers.Dense(10, activation='relu'),  # Hidden layer
        keras.layers.Dropout(0.2), # TODO: test
        keras.layers.Dense(y_train.shape[1], activation='softmax')  # Output layer with softmax for one-hot encoding
    ])
    # set learning rate
    opt = keras.optimizers.Adam(learning_rate=0.001)
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def evaluate_model(y_pred, y_test:pd.DataFrame):
    """
    Evaluates the model using the predicted and actual labels.
    
    Args:
        y_pred (DataFrame): The predicted labels.
        y_test (DataFrame): The actual labels.
    """
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test, axis=1)

    # print accuracy
    print(f"Global Accuracy: {accuracy_score(y_test_classes, y_pred_classes)*100:.2f}%")

    # precision, recall, f1-score
    print(classification_report(y_test_classes, y_pred_classes, target_names=y_test.columns, zero_division=0))

    tn, fp, fn, tp = confusion_matrix(y_test_classes, y_pred_classes).ravel()
    print(f"True Negative Rate: {tn/(tn+fp)*100:.2f}%")
    print(f"False Positive Rate: {fp/(tn+fp)*100:.2f}%")
    print(f"True Positive Rate: {tp/(tp+fn)*100:.2f}%")
    print(f"False Negative Rate: {fn/(tp+fn)*100:.2f}%")