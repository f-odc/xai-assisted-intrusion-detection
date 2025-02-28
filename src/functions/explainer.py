"""
This module contains functions for generating explanations for ML models.

Functions:
- generate_shap_explainer: Generates a SHAP explainer for a model.
- generate_shap_values: Generates SHAP values for a model.
- plot_shap_summary: Plots a SHAP summary plot.

Usage:
------
>>> import explainer as exp
>>> shap_explainer = exp.generate_shap_explainer(model, mask)
"""

import pandas as pd
import shap


def generate_shap_explainer(model, mask:pd.DataFrame):
    """
    Generates a SHAP explainer for a model.

    Args:
        model (TensorFlowV2Classifier): The model to explain.
        mask (DataFrame): The baseline mask features. Used for the explainer baseline.

    Returns:
        KernelExplainer: The SHAP explainer.
    """
    shap_explainer = shap.Explainer(model, mask, feature_names=mask.columns)
    return shap_explainer


def generate_shap_values(shap_explainer, X:pd.DataFrame) -> pd.DataFrame:
    """
    Generates SHAP values for a model.

    Args:
        shap_explainer (KernelExplainer): The SHAP explainer to use.
        X (DataFrame): The data to generate SHAP values for.

    Returns:
        np.array: The generated explanations from the SHAP explainer.
        DataFrame: Only the SHAP values as a DataFrame.
    """
    shap_values = shap_explainer(X)
    shap_values = shap_values[:, :, 0]
    shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns)
    return shap_values, shap_values_df


def plot_shap_summary(shap_values, X:pd.DataFrame, num:int = None):
    """
    Plots a SHAP summary plot.

    Args:
        shap_values (np.array): The SHAP values to plot.
        X (DataFrame): The data to plot SHAP values for.
        num (int, optional): The number of features to display. If None, 10 features are displayed. Defaults to None.
    """
    shap.summary_plot(shap_values, X, max_display=num if num is not None else 10)
