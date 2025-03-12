"""
This module contains functions for visualizing data and data distributions.

Functions:
- visualize_data_distribution: Visualizes the distribution of two datasets using PCA.
- pca_visualization_side_by_side: Visualizes the distribution of two datasets using PCA side by side.

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


def pca_visualization_side_by_side(data, data_label, target, target_label):
    """
    Visualizes the distribution of two datasets using PCA side by side.

    Args:
        data (np.array): The original data. PCA is fit on this data.
        data_label (str): The description for the original data.
        target (np.array): The target data.
        target_label (str): The description for the target data.

    Returns:
        None    
    """
    # Apply PCA given data
    pca = PCA(n_components=2)
    pca_data = pca.fit_transform(data)
    pca_target = pca.transform(target)

    # Plot PCA for data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(pca_data[:, 0], pca_data[:, 1], color='blue', alpha=0.7)
    plt.title(f"PCA of {data_label}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    # Plot PCA for target
    plt.subplot(1, 2, 2)
    plt.scatter(pca_target[:, 0], pca_target[:, 1], color='red', alpha=0.7)
    plt.title(f"PCA of {target_label}")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.tight_layout()
    plt.show()