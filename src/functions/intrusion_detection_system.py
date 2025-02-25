"""
This module contains functions for building an intrusion detection system using a deep neural network.

Functions:
- build_intrusion_detection_system: Builds an intrusion detection system using a deep neural network.
- create_model: Creates a deep neural network model for intrusion detection.
- evaluate_model: Evaluates the model using the predicted and actual labels.

Usage:
------
import intrusion_detection_system as ids
model = ids.build_intrusion_detection_system(X_train, y_train, X_test, y_test)

"""

import setuptools.dist # needed to avoid error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd


def build_intrusion_detection_system(X_train, y_train, X_test, y_test):
    """
    Builds an intrusion detection system using a deep neural network.
    
    Args:
        X_train (DataFrame): The training features.
        y_train (DataFrame): The training labels.
        X_test (DataFrame): The test features.
        y_test (DataFrame): The test labels.

    Returns:
        Sequential: The trained Keras sequential model.
    """
    # Create
    model = create_model(X_train, y_train)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    # Train
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=100)
    # Predict
    y_pred = model.predict(X_test)
    # Evaluate
    evaluate_model(y_pred, y_test)
    return model


def create_model(X_train, y_train):
    """
    Creates a deep neural network model for intrusion detection.
    
    Args:
        X_train (DataFrame): The training features.
        y_train (DataFrame): The training labels.

    Returns:
        Sequential: A compiled Keras sequential model.
    """
    # keras model for handling one hot encoded labels -> needed for attack creation
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),  # Define input shape explicitly
        keras.layers.Dense(50, activation='relu'),  # Hidden layer
        keras.layers.Dense(30, activation='relu'),  # Hidden layer
        keras.layers.Dense(10, activation='relu'),  # Hidden layer
        keras.layers.Dense(y_train.shape[1], activation='softmax')  # Output layer with softmax for one-hot encoding
    ])

    # set learning rate
    opt = keras.optimizers.Adam(learning_rate=0.001)
    # compile the keras model
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def evaluate_model(y_pred, y_test: pd.DataFrame):
    """
    Evaluates the model using the predicted and actual labels.
    
    Args:
        y_pred (DataFrame): The predicted labels.
        y_test (DataFrame): The actual labels.
    """
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_test_classes = np.argmax(y_test.to_numpy(), axis=1)

    # print accuracy
    print(f"Global Accuracy: {accuracy_score(y_test_classes, y_pred_classes)*100:.2f}%")

    # precision, recall, f1-score
    print(classification_report(y_test_classes, y_pred_classes, target_names=y_test.columns, zero_division=0))
