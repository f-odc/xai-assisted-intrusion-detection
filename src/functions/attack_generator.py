"""
Generates adversarial samples using the ART library. Applicable for White- and Black-Box attacks.

Functions:
- convert_to_art_model: Converts a Keras model to an ART model.
- evaluate_art_model: Evaluates an ART model on a test set.
- generate_cw_attacks: Generates Carlini & Wagner White-Box attacks on a model.
- generate_cw_attacks_parallel: Generates Carlini & Wagner White-Box attacks on a model using parallel processing.
- generate_fgsm_attacks: Generates Fast Gradient Sign Method White-Box attacks on a model.

Usage:
------
>>> import attack_generator as ag
>>> art_model = ag.convert_to_art_model(ids_model, X_train)

"""


from art.estimators.classification import TensorFlowV2Classifier
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from art.attacks.evasion import CarliniL2Method, FastGradientMethod
import numpy as np
import pandas as pd
import os
import multiprocessing
import logging

# Optionally: Suppress TensorFlow warnings globally
tf.get_logger().setLevel(logging.ERROR)


def convert_to_art_model(model, X_train):
    """
    Converts a Keras model to an ART model.
    
    Args:
        model (Sequential): The Keras model to convert.
        
    Returns:
        TensorFlowV2Classifier: The ART model.
    """
    # Define loss function
    loss_object = keras.losses.BinaryCrossentropy()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    input_dim = X_train.shape[1] 

    @tf.function
    def custom_train_step(model, x_batch, y_batch):
        with tf.GradientTape() as tape:
            predictions = model(x_batch, training=True)
            loss = loss_object(y_batch, predictions)
        
        # Compute and apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss

    # KerasClassifier uses tf.keras.backend.placeholder, which has been removed in TensorFlow 2.10+.so we need to use TensorFlowV2Classifier
    classifier = TensorFlowV2Classifier(
        model=model,
        nb_classes=2,  # Binary classification (0 or 1)
        input_shape=(input_dim,),  # Input shape
        clip_values=(0, 1), # because of the min-max normalization
        optimizer=optimizer, 
        loss_object=loss_object,
        train_step=custom_train_step  # Use default training function
    )
    return classifier


def evaluate_art_model(model, X_test:pd.DataFrame, y_test:pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates an ART model on a test set. Prints the accuracy, classification report, and true/false positives/negatives.
    
    Args:
        model (TensorFlowV2Classifier): The ART model to evaluate.
        X_test (DataFrame): The test features.
        y_test (DataFrame): The test labels.
        
    Returns:
        np.ndarray: The model's predictions.
    """
    # Predict and Convert to binary
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    y_test_binary = np.array(y_test).argmin(axis=1)
    y_pred_binary = np.array(y_pred).argmin(axis=1)
    # Evaluate
    accuracy = accuracy_score(y_test_binary, y_pred_binary)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(classification_report(y_test_binary, y_pred_binary, target_names=y_test.columns[::-1], zero_division=0)) # Reverse target_names because classification_reports starts displaying the class 0 (ATTACK) and then 1 (BENIGN)
    print("Confusion Matrix: Positive == BENIGN")
    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    # Create DataFrame with the same columns and indices as y_test
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns, index=X_test.index)
    return y_pred


def generate_cw_attacks(classifier, X:pd.DataFrame, target_label=None) -> pd.DataFrame:
    """
    Generates Carlini & Wagner White-Box attacks on a model.
    
    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (int, optional): The label the model should predict after the attack: `1` for `BENIGN`, `0` for `ATTACK`. If None, the attack is untargeted. Defaults to None.
        
    Returns:
        DataFrame: The adversarial examples
    """
    attack_cw = CarliniL2Method(classifier=classifier, confidence=0.0, targeted=(target_label is not None))

    # generate one-hot-encoded target labels
    if target_label is not None:
        target_array = np.zeros((X.shape[0], 2))
        target_array[:, 1 - target_label] = 1 # ensures that the array is [1, 0] for target_label=1 and [0, 1] for target_label=0

    # Generate adversarial examples
    X_np = X.to_numpy()
    X_adv_cw = attack_cw.generate(x=X_np, y=target_array if target_label is not None else None)
    X_adv_cw = pd.DataFrame(X_adv_cw, columns=X.columns, index=X.index)
    print(f'Adversarial C&W examples generated. Shape: {X_adv_cw.shape}')

    return X_adv_cw


def generate_cw_attacks_parallel(classifier, X:pd.DataFrame, target_label=None, num_cores=1) -> pd.DataFrame:
    """
    Generates Carlini & Wagner White-Box attacks on a model using parallel processing.

    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (int, optional): The label the model should predict after the attack: `1` for `BENIGN`, `0` for `ATTACK`. If None, the attack is untargeted. Defaults to None.
        num_cores (int, optional): The number of CPU cores to use for parallel processing. Defaults to 1.

    Returns:
        DataFrame: The generated adversarial samples
    """
    print(f"Running attack using {num_cores} CPU cores...\n")

    # Split data into `num_cores` equal parts
    def split_into_batches(data, num_splits):
        split_size = len(data) // num_splits
        return [data[i * split_size: (i + 1) * split_size] for i in range(num_splits - 1)] + [data[(num_splits - 1) * split_size:]]

    # convert X
    X_np = X.to_numpy()
    # generate batches for parallel processing
    X_batches = split_into_batches(X_np, num_cores)
    # generate one-hot-encoded target labels
    if target_label is not None:
        target_array = np.zeros((X.shape[0], 2))
        target_array[:, 1 - target_label] = 1 # ensures that the array is [1, 0] for target_label=1 and [0, 1] for target_label=0
    target_batches = split_into_batches(target_array, num_cores) if target_label is not None else None

    # Start parallel processing
    with multiprocessing.Pool(processes=num_cores, initializer=init_parallel_process, initargs=(classifier,)) as pool:
        if target_label is None:
            results = pool.map(generate_cw_attack_batch, X_batches)
        else:
            results = pool.starmap(generate_cw_attack_batch, zip(X_batches, target_batches))

    # Merge results back into a single NumPy array
    X_adv_cw = np.vstack(results)
    # Create new DataFrame with old indices and column names    
    X_adv_cw = pd.DataFrame(X_adv_cw, columns=X.columns, index=X.index)

    return X_adv_cw


def generate_fgsm_attacks(classifier, X:pd.DataFrame, target_label=None) -> pd.DataFrame:
    """
    Generates Fast Gradient Sign Method Whote-Box attacks on a model.
    
    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (int, optional): The label the model should predict after the attack: `1` for `BENIGN`, `0` for `ATTACK`. If None, the attack is untargeted. Defaults to None.
        
    Returns:
        DataFrame: The adversarial examples
    """
    attack_fgsm = FastGradientMethod(estimator=classifier, eps=0.1, targeted=(target_label is not None)) # ε tune this for stronger/weaker attacks: 0.01 weak, 0.1 balanced, 0.3-0.5 strong, 1 very strong
    # the higher the epsilon, the easier it will be detected

    # generate one-hot-encoded target labels
    if target_label is not None:
        target_array = np.zeros((X.shape[0], 2))
        target_array[:, 1 - target_label] = 1 # ensures that the array is [1, 0] for target_label=1 and [0, 1] for target_label=0

    # Generate adversarial examples
    X_np = X.to_numpy()
    X_adv_fgsm = attack_fgsm.generate(x=X_np, y=target_array if target_label is not None else None)
    X_adv_fgsm = pd.DataFrame(X_adv_fgsm, columns=X.columns, index=X.index)
    print(f'Adversarial FGSM examples generated. Shape: {X_adv_fgsm.shape}')

    return X_adv_fgsm


def init_parallel_process(model):
    """
    Initializes a process with shared variables. Necessary for multiprocessing.
    
    Args:
        model (TensorFlowV2Classifier): The ART model to attack.
    """
    global classifier_shared
    classifier_shared = model


def generate_cw_attack_batch(batch, batch_target=None):
    """
    Generates adversarial examples for a batch of samples using the Carlini & Wagner White-Box attack. Used in parallel processing.

    Args:
        batch (Array): The batch of samples to attack.
        batch_target (Array, optional): The one-hot encoded label the model should predict after the attack. If None, the attack is untargeted. Defaults to None.

    Returns:
        Array: The adversarial modified batches.
    """
    pid = os.getpid()  # Get process ID for debugging
    print(f"Process {pid} is generating adversarial examples for batch of size {len(batch)} \n")
    # Create a new attack instance (ART objects may not be shared directly)
    attack = CarliniL2Method(classifier=classifier_shared, confidence=0.1, targeted=(batch_target is not None))
    # Generate adversarial examples
    adv_samples = attack.generate(x=np.array(batch), y=batch_target if batch_target is not None else None)
    return adv_samples