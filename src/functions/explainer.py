"""
This module contains functions for generating explanations for ML models.

Functions:
- generate_shap_explainer: Generates a SHAP explainer for a model.
- generate_lime_explainer: Generates a LIME explainer for a model.
- generate_shap_values: Generates SHAP values for a model.
- generate_lime_explanation: Generates a single LIME explanation for the prediction of a model.
- plot_shap_summary: Plots a SHAP summary plot.
- plot_shap_summary_comparison: Plots two SHAP summary plot side by side.

Usage:
------
>>> import explainer as exp
>>> shap_explainer = exp.generate_shap_explainer(model, mask)
"""

import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


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


def generate_lime_explainer(mask:pd.DataFrame, label_names):
    """
    Generates a LIME explainer for a model.

    Args:
        model (TensorFlowV2Classifier): The model to explain.
        mask (DataFrame): The baseline mask features. Used for the explainer baseline.

    Returns:
        LimeTabularExplainer: The LIME explainer.
    """
    lime_explainer = LimeTabularExplainer(mask.values, feature_names=mask.columns, class_names=label_names, discretize_continuous=True)
    return lime_explainer


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
    shap_values_df = pd.DataFrame(shap_values.values, columns=X.columns, index=X.index)
    return shap_values, shap_values_df


def generate_lime_explanation(lime_explainer:LimeTabularExplainer, X, model, num:int = None):
    """
    Generates a single LIME explanation for the prediction of a model.

    Args:
        lime_explainer (LimeTabularExplainer): The LIME explainer to use.
        X : The single data to generate LIME explanations for, e.g. X_train.values[0].
        model (TensorFlowV2Classifier): The model to explain.
        num (int, optional): The number of features to display. If None, 10 features are displayed. Defaults to None.

    Returns:
        list: The generated explanations from the LIME explainer.
    """
    explanation = lime_explainer.explain_instance(X, model.predict, num_features=num if num is not None else 10)
    explanation.show_in_notebook(show_table=True)
    return explanation


def plot_shap_summary(shap_values, X:pd.DataFrame, num:int = None, target_indices=None):
    """
    Plots a SHAP summary plot.

    Args:
        shap_values (np.array): The SHAP values to plot.
        X (DataFrame): The data to plot SHAP values for.
        num (int, optional): The number of features to display. If None, 10 features are displayed. Defaults to None.
        target_indices (list, optional): The indices of the shap values that should be plotted. If None, all samples are used. Defaults to None.
    """
    if target_indices == None: # all samples
        shap.summary_plot(shap_values, X, max_display=num if num is not None else 10)
    else:
        shap.summary_plot(shap_values[target_indices], X.iloc[target_indices], max_display=num if num is not None else 10)


def plot_shap_summary_comparison(sv_1, X_1:pd.DataFrame, sv_2, X_2:pd.DataFrame, num:int = None, target_indices=None, title=''):
    """
    Plots two SHAP summary plot side by side.
    
    Args:
        sv_1 (np.array): The SHAP values of the first model.
        X_1 (DataFrame): The data to plot SHAP values for the first model.
        sv_2 (np.array): The SHAP values of the second model.
        X_2 (DataFrame): The data to plot SHAP values for the second model.
        num (int, optional): The number of features to display. If None, 10 features are displayed. Defaults to None.
        target_indices (list, optional): The indices of the shap values that should be plotted. If None, all samples are used. Defaults to None.
        title (str, optional): The title of the plot. Defaults to "".
    """
    plt.figure(figsize=(16,5))
    plt.suptitle(title)
    plt.subplot(1,2,1)
    # plot_size=None, show=False important to fit plots into figure
    if target_indices == None: # all samples
        shap.summary_plot(sv_1, X_1, plot_size=None, show=False, max_display=num if num is not None else 10)
    else:
        shap.summary_plot(sv_1[target_indices], X_1.iloc[target_indices], plot_size=None, show=False, max_display=num if num is not None else 10)
    plt.subplot(1,2,2)
    if target_indices == None: # all samples
        shap.summary_plot(sv_2, X_2, plot_size=None, show=False, max_display=num if num is not None else 10)
    else:
        shap.summary_plot(sv_2[target_indices], X_2.iloc[target_indices], plot_size=None, show=False, max_display=num if num is not None else 10)
    plt.tight_layout()
    plt.show()