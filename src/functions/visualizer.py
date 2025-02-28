"""
This module contains functions for visualizing data and data distributions.

Functions:
- visualize_data_distribution: Visualizes the distribution of two datasets using PCA.

Usage:
------
>>> import visualizer as vis
>>> vis.visualize_data_distribution(X_train, 'Original Data', X_adv, 'Adversarial Data')
"""

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def visualize_data_distribution(data, data_label, target, target_label):
    """
    Visualizes the distribution of two datasets using PCA.
    
    Args:
        data (np.array): The original data.
        data_label (str): The description for the original data.
        target (np.array): The target data.
        target_label (str): The description for the target data.

    Returns:
        None
    """
    # Perform PCA with 2 components
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)      # Fit and transform original data
    data_adv_pca = pca.transform(target)  # Transform target data (using same PCA)

    # Extract Principal Components
    pc1_data, pc2_data = data_pca[:, 0], data_pca[:, 1]
    pc1_adv, pc2_adv = data_adv_pca[:, 0], data_adv_pca[:, 1]

    # Create 2D Scatter Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(pc1_data, pc2_data, color='blue', alpha=0.6, label=data_label)
    plt.scatter(pc1_adv, pc2_adv, color='red', alpha=0.6, label=target_label)

    # Add labels, title, and legend
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title(f'2D PCA Plot of {data_label} and {target_label}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()