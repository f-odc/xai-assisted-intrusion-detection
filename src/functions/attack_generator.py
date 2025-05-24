"""
Generates adversarial samples using the ART library. Applicable for White- and Black-Box attacks.

Functions:
- convert_to_art_model: Converts a Keras model to an ART model.
- split_into_attack_classes: Splits the dataset evenly into specified classes with given labels.
- evaluate_art_model: Evaluates an ART model on a test set.
- generate_cw_attacks: Generates Carlini & Wagner White-Box attacks on a model.
- generate_cw_attacks_parallel: Generates Carlini & Wagner White-Box attacks on a model using parallel processing.
- generate_fgsm_attacks: Generates Fast Gradient Sign Method White-Box attacks on a model.
- generate_hsj_attacks_parallel: Generates HopSkipJump Black-Box attacks on a model using parallel processing.
- generate_jsma_attacks: Generates Jacobian Saliency Map Attack White-Box attacks on a model.
- generate_pgd_attacks: Generates Projected Gradient Descent White-Box attacks on a model.
- generate_boundary_attacks_parallel: Generates Boundary Black-Box attacks on a model using parallel processing.

Usage:
------
>>> import attack_generator as ag
>>> art_model = ag.convert_to_art_model(ids_model, X_train)

"""


from art.estimators.classification import TensorFlowV2Classifier
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from art.attacks.evasion import CarliniL2Method, FastGradientMethod, HopSkipJump, SaliencyMapMethod, ProjectedGradientDescentTensorFlowV2, BoundaryAttack, BasicIterativeMethod, DeepFool
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import os
import keras
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
    loss_object = keras.losses.CategoricalCrossentropy()
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


def split_into_classes(X:pd.DataFrame, y:pd.DataFrame, normal_class_label, attack_class_labels, split=None):
    """
    Splits the dataset into normal and attack classes.

    Args:
        X (numpy.ndarray): The input samples.
        y (numpy.ndarray): The labels.
        normal_class_label (str): The label for the normal class.
        attack_class_labels (list of str): The labels for the attack classes.
        split (float, optional): The proportion of data to use for the normal class. Defaults to None. If None, 50% of the data will be used for the normal class.

    Returns:
        dict: A dictionary where keys are class names and values are tuples (X_subset, y_subset) of the normal and attack classes.
    """
    if split is None:
        split = 0.5  # Default split if not provided
    # get normal samples
    X_normal_class = X.sample(frac=split, random_state=42)
    y_normal_class = y.loc[X_normal_class.index]
    X_attack = X.drop(X_normal_class.index)
    y_attack = y.drop(y_normal_class.index)

    # Compute samples per attack class
    num_classes = len(attack_class_labels)
    total_samples = len(X_attack)
    base_samples_per_class = total_samples // num_classes
    remainder = total_samples % num_classes
    # Dictionary to store the data splits
    class_splits = {}
    start = 0
    for i, label in enumerate(attack_class_labels):
        extra = remainder if i == 0 else 0  # First class gets the remainder
        end = start + base_samples_per_class + extra
        class_splits[label] = (X_attack[start:end], y_attack[start:end])
        start = end

    # add normal class to the dictionary
    class_splits[normal_class_label] = (X_normal_class, y_normal_class)
    return class_splits


def split_into_attack_classes(X, y, class_labels):
    """
    Splits the dataset evenly into specified attack classes with given labels.

    Args:
        X (numpy.ndarray): The input samples.
        y (numpy.ndarray): The labels.
        class_labels (list of str): The names of the classes (e.g., ["normal", "cw", "fgsm", "hsj"]).

    Returns:
        dict: A dictionary where keys are class names and values are tuples (X_subset, y_subset).
    """
    num_classes = len(class_labels)
    # Shuffle data to avoid biases
    X, y = shuffle(X, y, random_state=42)
    # Compute samples per class
    total_samples = len(X)
    base_samples_per_class = total_samples // num_classes
    remainder = total_samples % num_classes
    # Dictionary to store the data splits
    class_splits = {}
    start = 0
    for i, label in enumerate(class_labels):
        extra = remainder if i == 0 else 0  # First class gets the remainder
        end = start + base_samples_per_class + extra
        class_splits[label] = (X[start:end], y[start:end])
        start = end

    return class_splits


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
    print(classification_report(y_test_binary, y_pred_binary, target_names=y_test.columns[::-1], zero_division=0, digits=4)) # Reverse target_names because classification_reports starts displaying the class 0 (ATTACK) and then 1 (BENIGN)
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


def generate_cw_attacks_parallel(classifier, X:pd.DataFrame, target_label=None, num_cores=1, feature_mask=None) -> pd.DataFrame:
    """
    Generates Carlini & Wagner White-Box attacks on a model using parallel processing.

    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (int, optional): The label the model should predict after the attack: `1` for `BENIGN`, `0` for `ATTACK`. If None, the attack is untargeted. Defaults to None.
        num_cores (int, optional): The number of CPU cores to use for parallel processing. Defaults to 1.
        feature_mask (numpy.ndarray, optional): A mask to specify which features to modify e.g.: [1, 0, 0, 1]. Only the features with a value of 1 will be modified. Defaults to None. If None, all features will be modified.

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
        results = pool.starmap(generate_cw_attack_batch, zip(
            X_batches, 
            target_batches if target_batches is not None else [None] * len(X_batches), # None in zip is not allowed so we generate a None list which is identified as None in the called function
            feature_mask if feature_mask is not None else [None] * len(X_batches)))

    # Merge results back into a single NumPy array
    X_adv_cw = np.vstack(results)
    # Create new DataFrame with old indices and column names    
    X_adv_cw = pd.DataFrame(X_adv_cw, columns=X.columns, index=X.index)

    return X_adv_cw


def generate_fgsm_attacks(classifier, X:pd.DataFrame, target_label=None, feature_mask=None) -> pd.DataFrame:
    """
    Generates Fast Gradient Sign Method Whote-Box attacks on a model.
    
    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (int, optional): The label the model should predict after the attack: `1` for `BENIGN`, `0` for `ATTACK`. If None, the attack is untargeted. Defaults to None.
        feature_mask (numpy.ndarray, optional): A mask to specify which features to modify e.g.: [1, 0, 0, 1]. Only the features with a value of 1 will be modified. Defaults to None. If None, all features will be modified.
        
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
    X_adv_fgsm = attack_fgsm.generate(x=X_np, y=target_array if target_label is not None else None, mask=feature_mask)
    X_adv_fgsm = pd.DataFrame(X_adv_fgsm, columns=X.columns, index=X.index)
    print(f'Adversarial FGSM examples generated. Shape: {X_adv_fgsm.shape}')

    return X_adv_fgsm


def generate_bim_attacks(classifier, X:pd.DataFrame, target_label=None, feature_mask=None) -> pd.DataFrame:
    """
    Generates Basic Iterative Method White-Box attacks on a model.
    
    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (int, optional): The label the model should predict after the attack: `1` for `BENIGN`, `0` for `ATTACK`. If None, the attack is untargeted. Defaults to None.
        feature_mask (numpy.ndarray, optional): A mask to specify which features to modify e.g.: [1, 0, 0, 1]. Only the features with a value of 1 will be modified. Defaults to None. If None, all features will be modified.
        
    Returns:
        DataFrame: The adversarial examples
    """
    attack_bim = BasicIterativeMethod(estimator=classifier, eps=0.1, targeted=(target_label is not None), max_iter=10) # ε tune this for stronger/weaker attacks: 0.01 weak, 0.1 balanced, 0.3-0.5 strong, 1 very strong

    # generate one-hot-encoded target labels
    if target_label is not None:
        target_array = np.zeros((X.shape[0], 2))
        target_array[:, 1 - target_label] = 1 # ensures that the array is [1, 0] for target_label=1 and [0, 1] for target_label=0

    # Generate adversarial examples
    X_np = X.to_numpy()
    X_adv_bim = attack_bim.generate(x=X_np, y=target_array if target_label is not None else None, mask=feature_mask)
    X_adv_bim = pd.DataFrame(X_adv_bim, columns=X.columns, index=X.index)
    print(f'Adversarial BIM examples generated. Shape: {X_adv_bim.shape}')

    return X_adv_bim


def generate_deepfool_attacks(classifier, X:pd.DataFrame, target_label=None, feature_mask=None) -> pd.DataFrame:
    """
    Generates DeepFool White-Box attacks on a model.
    
    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (int, optional): The label the model should predict after the attack: `1` for `BENIGN`, `0` for `ATTACK`. If None, the attack is untargeted. Defaults to None.
        feature_mask (numpy.ndarray, optional): A mask to specify which features to modify e.g.: [1, 0, 0, 1]. Only the features with a value of 1 will be modified. Defaults to None. If None, all features will be modified.
        
    Returns:
        DataFrame: The adversarial examples
    """
    attack_deepfool = DeepFool(classifier=classifier, epsilon=0.1) # ε tune this for stronger/weaker attacks: 0.01 weak, 0.1 balanced, 0.3-0.5 strong, 1 very strong

    # generate one-hot-encoded target labels
    if target_label is not None:
        target_array = np.zeros((X.shape[0], 2))
        target_array[:, 1 - target_label] = 1 # ensures that the array is [1, 0] for target_label=1 and [0, 1] for target_label=0

    # Generate adversarial examples
    X_np = X.to_numpy()
    X_adv_deepfool = attack_deepfool.generate(x=X_np, y=target_array if target_label is not None else None, mask=feature_mask)
    X_adv_deepfool = pd.DataFrame(X_adv_deepfool, columns=X.columns, index=X.index)
    print(f'Adversarial DeepFool examples generated. Shape: {X_adv_deepfool.shape}')

    return X_adv_deepfool


def generate_hsj_attacks_parallel(classifier, X:pd.DataFrame, target_label=None, num_cores=1, feature_mask=None) -> pd.DataFrame:
    """
    Generates HopSkipJump Black-Box attacks on a model using parallel processing.

    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (int, optional): The label the model should predict after the attack: `1` for `BENIGN`, `0` for `ATTACK`. If None, the attack is untargeted. Defaults to None.
        num_cores (int, optional): The number of CPU cores to use for parallel processing. Defaults to 1.
        feature_mask (numpy.ndarray, optional): A mask to specify which features to modify e.g.: [1, 0, 0, 1]. Only the features with a value of 1 will be modified. Defaults to None. If None, all features will be modified.

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
        results = pool.starmap(generate_hsj_attack_batch, zip(
            X_batches, 
            target_batches if target_batches is not None else [None] * len(X_batches), # None in zip is not allowed so we generate a None list which is identified as None in the called function
            feature_mask if feature_mask is not None else [None] * len(X_batches)))

    # Merge results back into a single NumPy array
    X_adv_hsj = np.vstack(results)
    # Create new DataFrame with old indices and column names    
    X_adv_hsj = pd.DataFrame(X_adv_hsj, columns=X.columns, index=X.index)
    print(f'Adversarial HopSkipJump examples generated. Shape: {X_adv_hsj.shape}')

    return X_adv_hsj


def generate_jsma_attacks(classifier, X:pd.DataFrame, target_label=None, feature_mask=None) -> pd.DataFrame:
    """
    Generates Jacobian Saliency Map Attack White-Box attacks on a model.
    
    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (int, optional): The label the model should predict after the attack: `1` for `BENIGN`, `0` for `ATTACK`. If None, the attack is untargeted. Defaults to None.
        feature_mask (numpy.ndarray, optional): A mask to specify which features to modify e.g.: [1, 0, 0, 1]. Only the features with a value of 1 will be modified. Defaults to None. If None, all features will be modified.
        
    Returns:
        DataFrame: The adversarial examples
    """
    attack_jsma = SaliencyMapMethod(classifier=classifier, theta=0.1, gamma=0.8)

    # generate one-hot-encoded target labels
    if target_label is not None:
        target_array = np.zeros((X.shape[0], 2))
        target_array[:, 1 - target_label] = 1 # ensures that the array is [1, 0] for target_label=1 and [0, 1] for target_label=0

    # Generate adversarial examples
    X_np = X.to_numpy()
    X_adv_jsma = attack_jsma.generate(x=X_np, y=target_array if target_label is not None else None, mask=feature_mask)
    X_adv_jsma = pd.DataFrame(X_adv_jsma, columns=X.columns, index=X.index)
    print(f'Adversarial JSMA examples generated. Shape: {X_adv_jsma.shape}')

    return X_adv_jsma


def generate_pgd_attacks(classifier, X:pd.DataFrame, target_label=None, feature_mask=None) -> pd.DataFrame:
    """
    Generates Projected Gradient Descent White-Box attacks on a model.
    
    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (int, optional): The label the model should predict after the attack: `1` for `BENIGN`, `0` for `ATTACK`. If None, the attack is untargeted. Defaults to None.
        feature_mask (numpy.ndarray, optional): A mask to specify which features to modify e.g.: [1, 0, 0, 1]. Only the features with a value of 1 will be modified. Defaults to None. If None, all features will be modified.
        
    Returns:
        DataFrame: The adversarial examples
    """
    attack_pgd = ProjectedGradientDescentTensorFlowV2(estimator=classifier, eps=0.1, targeted=(target_label is not None), batch_size=32, max_iter=10) # ε tune this for stronger/weaker attacks: 0.01 weak, 0.1 balanced, 0.3-0.5 strong, 1 very strong

    # generate one-hot-encoded target labels
    if target_label is not None:
        target_array = np.zeros((X.shape[0], 2))
        target_array[:, 1 - target_label] = 1 # ensures that the array is [1, 0] for target_label=1 and [0, 1] for target_label=0

    # Generate adversarial examples
    X_np = X.to_numpy()
    X_adv_pgd = attack_pgd.generate(x=X_np, y=target_array if target_label is not None else None, mask=feature_mask)
    X_adv_pgd = pd.DataFrame(X_adv_pgd, columns=X.columns, index=X.index)
    print(f'Adversarial PGD examples generated. Shape: {X_adv_pgd.shape}')

    return X_adv_pgd


def generate_boundary_attacks_parallel(classifier, X:pd.DataFrame, target_label=None, num_cores=1) -> pd.DataFrame:
    """
    Generates Boundary Black-Box attacks on a model.
    
    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (int, optional): The label the model should predict after the attack: `1` for `BENIGN`, `0` for `ATTACK`. If None, the attack is untargeted. Defaults to None.
        num_cores (int, optional): The number of CPU cores to use for parallel processing. Defaults to 1.
        
    Returns:
        DataFrame: The adversarial examples
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
            results = pool.map(generate_boundary_attack_batch, X_batches)
        else:
            results = pool.starmap(generate_boundary_attack_batch, zip(X_batches, target_batches))

    # Merge results back into a single NumPy array
    X_adv_boundary = np.vstack(results)
    # Create new DataFrame with old indices and column names    
    X_adv_boundary = pd.DataFrame(X_adv_boundary, columns=X.columns, index=X.index)
    print(f'Adversarial Boundary examples generated. Shape: {X_adv_boundary.shape}')

    return X_adv_boundary


# --- internal functions for parallel processing ---


def init_parallel_process(model):
    """
    Initializes a process with shared variables. Necessary for multiprocessing.
    
    Args:
        model (TensorFlowV2Classifier): The ART model to attack.
    """
    global classifier_shared
    classifier_shared = model


def generate_cw_attack_batch(batch, batch_target=None, feature_mask=None):
    """
    Generates adversarial examples for a batch of samples using the Carlini & Wagner White-Box attack. Used in parallel processing.

    Args:
        batch (Array): The batch of samples to attack.
        batch_target (Array, optional): The one-hot encoded label the model should predict after the attack. If None, the attack is untargeted. Defaults to None.
        feature_mask (numpy.ndarray, optional): A mask to specify which features to modify. Defaults to None.
        
    Returns:
        Array: The adversarial modified batches.
    """
    pid = os.getpid()  # Get process ID for debugging
    print(f"Process {pid} is generating adversarial examples for batch of size {len(batch)} \n")
    # Create a new attack instance (ART objects may not be shared directly)
    attack = CarliniL2Method(classifier=classifier_shared, confidence=0.1, targeted=(batch_target is not None))
    # Generate adversarial examples
    adv_samples = attack.generate(x=np.array(batch), y=batch_target if batch_target is not None else None, mask=feature_mask)
    return adv_samples


def generate_hsj_attack_batch(batch, batch_target=None, feature_mask=None):
    """
    Generates adversarial examples for a batch of samples using the HopSkipJump White-Box attack. Used in parallel processing.

    Args:
        batch (Array): The batch of samples to attack.
        batch_target (Array, optional): The one-hot encoded label the model should predict after the attack. If None, the attack is untargeted. Defaults to None.
        feature_mask (numpy.ndarray, optional): A mask to specify which features to modify. Defaults to None. If None, all features will be modified.

    Returns:
        Array: The adversarial modified batches.
    """
    pid = os.getpid()  # Get process ID for debugging
    print(f"Process {pid} is generating adversarial examples for batch of size {len(batch)} \n")
    # Create a new attack instance (ART objects may not be shared directly)
    attack = HopSkipJump(classifier=classifier_shared, targeted=(batch_target is not None), norm=2, init_eval=10) # set init_eval to be less than the sample size of each batch
    # Generate adversarial examples
    adv_samples = attack.generate(x=np.array(batch), y=batch_target if batch_target is not None else None, mask=feature_mask)
    return adv_samples


def generate_boundary_attack_batch(batch, batch_target=None):
    """
    Generates adversarial examples for a batch of samples using the HopSkipJump White-Box attack. Used in parallel processing.

    Args:
        batch (Array): The batch of samples to attack.
        batch_target (Array, optional): The one-hot encoded label the model should predict after the attack. If None, the attack is untargeted. Defaults to None.

    Returns:
        Array: The adversarial modified batches.
    """
    pid = os.getpid()  # Get process ID for debugging
    print(f"Process {pid} is generating adversarial examples for batch of size {len(batch)} \n")
    # Create a new attack instance (ART objects may not be shared directly)
    attack = BoundaryAttack(estimator=classifier_shared, targeted=(batch_target is not None), max_iter=200, epsilon=0.1, delta=0.01 ,verbose=False) # ε tune this for stronger/weaker attacks: 0.01 weak, 0.1 balanced, 0.3-0.5 strong, 1 very strong
    # Generate adversarial examples
    adv_samples = attack.generate(x=np.array(batch), y=batch_target if batch_target is not None else None)
    return adv_samples
