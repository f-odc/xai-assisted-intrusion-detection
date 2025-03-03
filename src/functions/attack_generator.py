"""
Generates adversarial samples using the ART library. Applicable for White- and Black-Box attacks.

Functions:
- convert_to_art_model: Converts a Keras model to an ART model.
- evaluate_art_model: Evaluates an ART model on a test set.
- generate_cw_attacks: Generates Carlini & Wagner White-Box attacks on a model.
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


def evaluate_art_model(model, X_test, y_test):
    """
    Evaluates an ART model on a test set. Prints the accuracy, classification report, and true/false positives/negatives.
    
    Args:
        model (TensorFlowV2Classifier): The ART model to evaluate.
        X_test (DataFrame): The test features.
        y_test (DataFrame): The test labels.
    """
    # Predict
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    y_test_binary = np.array(y_test).argmin(axis=1)
    y_pred_binary = np.array(y_pred).argmin(axis=1)
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=y_test.columns))
    print("Confusion Matrix: Positive == BENIGN")
    tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
    return accuracy


def generate_cw_attacks(classifier, X:pd.DataFrame, target_label=None) -> pd.DataFrame:
    """
    Generates Carlini & Wagner White-Box attacks on a model.
    
    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (Array, optional): The one-hot encoded label the model should predict after the attack, e.g. [0, 1] for 'ATTACK'. If None, the attack is untargeted. Defaults to None.
        
    Returns:
        DataFrame: The adversarial examples
    """
    attack_cw = CarliniL2Method(classifier=classifier, confidence=0.0, targeted=(target_label is not None))

    # Generate adversarial examples
    X_np = X.to_numpy()
    X_adv_cw = attack_cw.generate(x=X_np, y=target_label if target_label is not None else None)
    X_adv_cw = pd.DataFrame(X_adv_cw, columns=X.columns)
    print(f'Adversarial C&W examples generated. Shape: {X_adv_cw.shape}')

    return X_adv_cw


def generate_fgsm_attacks(classifier, X:pd.DataFrame, target_label=None) -> pd.DataFrame:
    """
    Generates Fast Gradient Sign Method Whote-Box attacks on a model.
    
    Args:
        classifier (TensorFlowV2Classifier): The ART model to attack.
        X (DataFrame): The features to modify.
        target_label (Array, optional): The one-hot encoded label the model should predict after the attack, e.g. [0, 1] for 'ATTACK'. If None, the attack is untargeted. Defaults to None.
        
    Returns:
        DataFrame: The adversarial examples
    """
    attack_fgsm = FastGradientMethod(estimator=classifier, eps=0.1, targeted=(target_label is not None)) # ε tune this for stronger/weaker attacks: 0.01 weak, 0.1 balanced, 0.3-0.5 strong, 1 very strong
    # the higher the epsilon, the easier it will be detected

    # Generate adversarial examples
    X_np = X.to_numpy()
    X_adv_fgsm = attack_fgsm.generate(x=X_np, y=target_label if target_label is not None else None)
    X_adv_fgsm = pd.DataFrame(X_adv_fgsm, columns=X.columns)
    print(f'Adversarial FGSM examples generated. Shape: {X_adv_fgsm.shape}')

    return X_adv_fgsm