import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 

def eval_model(name, y_true, y_pred):
    """Print Accuracy, F1, and Kappa for a model."""
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"\n{name} Results:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1:       {f1:.4f}")
    print(f"  Kappa:    {kappa:.4f}")

    return acc, f1, kappa

def align_clusters(y_true, y_pred):
    """Align cluster labels to ground truth using Hungarian matching."""
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(-cm)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    return np.array([mapping[c] for c in y_pred])

def plot_clustering_results(model_name, X_scaled, y_true, le, cluster_range, accuracies, clusters_k5):
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    # Plot 1: k vs accuracy
    axs[0].plot(list(cluster_range), accuracies, marker='o', label="Accuracy")
    axs[0].set_title("Number of Clusters vs. Accuracy")
    axs[0].set_xlabel("Number of Clusters (k)")
    axs[0].set_ylabel("Accuracy")
    axs[0].set_xticks(list(cluster_range))
    axs[0].set_ylim(0, 1.05)
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend()

    # Plot 2: KMeans k=5 clusters in PCA space
    for c in np.unique(clusters_k5):
        pts = X_pca[clusters_k5 == c]
        axs[1].scatter(pts[:, 0], pts[:, 1], label=f"Cluster {c}", alpha=0.7, s=30)
    axs[1].set_title(f"{model_name} (k=5) in PCA-2D")
    axs[1].set_xlabel("PC1")
    axs[1].set_ylabel("PC2")
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # Plot 3: True labels in PCA space
    for lab in np.unique(y_true):
        pts = X_pca[y_true == lab]
        name = str(le.inverse_transform([lab])[0]) if hasattr(le, "inverse_transform") else f"Class {lab}"
        axs[2].scatter(pts[:, 0], pts[:, 1], label=name, alpha=0.6, s=30)
    axs[2].set_title("True Labels in PCA-2D")
    axs[2].set_xlabel("PC1")
    axs[2].set_ylabel("PC2")
    axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
    axs[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    