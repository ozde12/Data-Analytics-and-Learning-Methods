# =========================
# Sensor Feature Selection Toolkit
# =========================
#
# This toolkit implements a *filter -> wrapper* feature selection pipeline
# tailored for classification tasks, matching your requirements:
#   - Variance filter (drop exactly-constant features)
#   - Correlation filter (drop one of each highly correlated pair)
#   - Mutual Information K-Best (keep top-K by MI with class labels)
#   - Wrapper methods (RFE / Sequential Forward/Backward Selection)
#
# Extras included:
#   - Correlation heatmap helper (nice for reports)
#   - Accuracy vs K (learning curve) for MI-KBest to choose K
#   - PCA cumulative explained variance helper (optional; for PCA-based pipelines)
#
# NOTE:
# - RFE requires an estimator exposing coef_ or feature_importances_ (e.g.,
#   LogisticRegression, DecisionTree, RandomForest). Use SFS for KNN / Naive Bayes.
# - Scaling is important for distance-based models (KNN) and helpful for LR.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.decomposition import PCA
from functools import partial

# -------------------------
# Helpers (classification only)
# -------------------------

def _cv(n_splits=5):
    """Stratified CV for classification."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def _default_estimator():
    """Default estimator for wrapper selection (works with RFE)."""
    return LogisticRegression(max_iter=2000, solver="liblinear", random_state=42)


def _supports_rfe(estimator):
    """Check if estimator can be used with RFE (needs coef_ or feature_importances_)."""
    try:
        X_tmp = np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]])
        y_tmp = np.array([0, 1, 0, 1])
        est_fit = clone(estimator).fit(X_tmp, y_tmp)
        return hasattr(est_fit, "coef_") or hasattr(est_fit, "feature_importances_")
    except Exception:
        return False

# -------------------------
# FILTER STAGE (variance -> correlation -> MI-KBest)
# -------------------------

def filter_select(
    X,
    y=None,
    feature_names=None,
    variance_thresh=0.0,   # 0.0 -> drop only EXACTLY constant features
    corr_thresh=0.95,      # feature-feature absolute correlation threshold
    k_best=None,           # integer: keep top-k by mutual information; None to skip
    scale_for_mi=True,
    mi_n_neighbors=None,   # if set, used via functools.partial in MI
    mi_random_state=None   # if set, used via functools.partial in MI
):
    """
    Apply FILTER methods for classification:
      1) VarianceThreshold: drop exactly constant features (threshold=0.0)
      2) Correlation filter: drop one of any highly-correlated pair (|r| > corr_thresh)
      3) Mutual Information (SelectKBest): keep top-k features by MI with labels

    Returns:
      X_filtered (ndarray), selected_names (list of str), report (pd.DataFrame)
    """
    # Normalize input types and names
    # Always use DataFrame's own column labels as feature names
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        feature_names = list(df.columns)
    else:
        X_mat = np.asarray(X)
        feature_names = [f"f{i}" for i in range(X_mat.shape[1])]
        df = pd.DataFrame(X_mat, columns=feature_names)

    report_rows = []

    # 1) Variance threshold (exactly-constant removal)
    vt = VarianceThreshold(threshold=variance_thresh)
    X_vt = vt.fit_transform(df)
    kept_mask_vt = vt.get_support()
    kept_ft_vt = [n for n, keep in zip(feature_names, kept_mask_vt) if keep]
    removed_ft_vt = [n for n, keep in zip(feature_names, kept_mask_vt) if not keep]

    print(f"[VarianceThreshold] Kept: {len(kept_ft_vt)}, Removed: {len(removed_ft_vt)}")
    if removed_ft_vt:
        print("Removed features (variance=0):", removed_ft_vt)

    report_rows.append({
        "stage": "variance_threshold",
        "kept": len(kept_ft_vt),
        "removed": len(removed_ft_vt),
        "removed_features": removed_ft_vt
    })

    # 2) Correlation filter on remaining features
    df_vt = pd.DataFrame(X_vt, columns=kept_ft_vt)
    corr = df_vt.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    ft_to_drop = [col for col in upper.columns if any(upper[col] > corr_thresh)]
    kept_ft_corr = [c for c in df_vt.columns if c not in ft_to_drop]
    X_corr = df_vt[kept_ft_corr].values

    print(f"[CorrelationFilter] Kept: {len(kept_ft_corr)}, Removed: {len(ft_to_drop)}")
    if ft_to_drop:
        print("Removed highly correlated features:", to_drop)

    report_rows.append({
        "stage": "corr_filter",
        "kept": len(kept_ft_corr),
        "removed": len(ft_to_drop),
        "removed_features": ft_to_drop
    })

    # 3) Mutual Information K-Best (optional)
    if (k_best is not None) and (y is not None) and (k_best < X_corr.shape[1]):
        X_for_scoring = StandardScaler().fit_transform(X_corr) if scale_for_mi else X_corr
        # Use functools.partial to lock MI estimator params when provided
        mi_func = mutual_info_classif if (mi_n_neighbors is None and mi_random_state is None) \
            else partial(mutual_info_classif, n_neighbors=mi_n_neighbors, random_state=mi_random_state)
        skb = SelectKBest(score_func=mi_func, k=k_best)
        X_k = skb.fit_transform(X_for_scoring, y)
        kept_mask_k = skb.get_support()
        kept_ft_k = [n for n, keep in zip(kept_ft_corr, kept_mask_k) if keep]
        removed_ft_k = [n for n, keep in zip(kept_ft_corr, kept_mask_k) if not keep]

        print(f"[MutualInfo KBest] Kept: {len(kept_ft_k)}, Removed: {len(removed_ft_k)}")
        if removed_ft_k:
            print("Removed low-MI features:", removed_ft_k)

        report_rows.append({
            "stage": f"select_k_best_mi({k_best})",
            "kept": len(kept_ft_k),
            "removed": len(removed_ft_k),
            "removed_features": removed_ft_k
        })
        return X_k, kept_ft_k, pd.DataFrame(report_rows)

    # If K-Best not applied, return correlation-filtered set
    return X_corr, kept_ft_corr, pd.DataFrame(report_rows)

# -------------------------
# WRAPPER STAGE (RFE or SFS)
# -------------------------

def wrapper_select(
    X,
    y,
    feature_names=None,
    method="rfe",               # "rfe", "sfs_forward", "sfs_backward"
    n_features_to_select=None,   # int
    estimator=None,              # e.g., LogisticRegression(), DecisionTreeClassifier(), KNeighborsClassifier(), GaussianNB()
    cv_splits=5,
    scoring=None,
    scale_inputs=True
):
    """
    Wrapper feature selection for classification using:
      - RFE (requires estimator with coef_ or feature_importances_)
      - SFS forward/backward (works with any estimator)

    Returns:
      X_selected (ndarray), selected_names (list), info dict
    """
    # Handle names and matrix
    if isinstance(X, pd.DataFrame):
        names = list(X.columns) if feature_names is None else feature_names
        X_mat = X.values
    else:
        X_mat = np.asarray(X)
        names = feature_names if feature_names is not None else [f"f{i}" for i in range(X_mat.shape[1])]

    # Choose estimator
    if estimator is None:
        estimator = _default_estimator()
    est = clone(estimator)

    # Optional scaling (recommended for LR/KNN; harmless for trees)
    if scale_inputs:
        scaler = StandardScaler()
        X_proc = scaler.fit_transform(X_mat)
    else:
        X_proc = X_mat

    cv = _cv(n_splits=cv_splits)

    if method == "rfe":
        if not _supports_rfe(est):
            raise ValueError(
                "RFE requires an estimator with coef_ or feature_importances_. "
                "Use LogisticRegression / DecisionTree / RandomForest, or switch to SFS."
            )
        n_keep = n_features_to_select if n_features_to_select is not None else max(1, X_proc.shape[1] // 2)
        selector = RFE(estimator=est, n_features_to_select=n_keep, step=1)
        selector.fit(X_proc, y)
        support = selector.get_support()
        ranks = getattr(selector, "ranking_", None)
        selected_names = [n for n, s in zip(names, support) if s]
        X_sel = X_proc[:, support]
        cv_score = cross_val_score(clone(est), X_sel, y, cv=cv, scoring=scoring, n_jobs=-1).mean()
        return X_sel, selected_names, {
            "method": method,
            "cv_score": cv_score,
            "support_mask": support,
            "ranks": ranks,
            "estimator": est
        }
    else:
        direction = "forward" if method == "sfs_forward" else "backward"
        sfs = SequentialFeatureSelector(
            est,
            n_features_to_select=n_features_to_select,
            direction=direction,
            scoring=scoring,
            cv=cv,
            n_jobs=-1
        )
        sfs.fit(X_proc, y)
        support = sfs.get_support()
        selected_names = [n for n, s in zip(names, support) if s]
        X_sel = X_proc[:, support]
        cv_score = cross_val_score(clone(est), X_sel, y, cv=cv, scoring=scoring, n_jobs=-1).mean()
        return X_sel, selected_names, {
            "method": method,
            "cv_score": cv_score,
            "support_mask": support,
            "estimator": est
        }

# -------------------------
# Visualization & Selection Aids
# -------------------------

def plot_correlation_heatmap(X, feature_names=None, figsize=(8,6)):
    """Plot an absolute correlation heatmap (after variance filtering is ideal)."""
    if isinstance(X, pd.DataFrame):
        df = X.copy()
    else:
        names = feature_names if feature_names is not None else [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=names)
    corr = df.corr().abs()
    plt.figure(figsize=figsize)
    im = plt.imshow(corr.values, aspect='auto')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Absolute Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def accuracy_vs_k_MI(
    X, y, estimator=None, k_values=None,
    variance_thresh=0.0, corr_thresh=0.95,
    scale_for_mi=True, scale_inputs=True,
    scoring="accuracy", cv_splits=5
):
    """
    Evaluate model accuracy vs number of features kept by MI-KBest.
    Returns lists (k_list, mean_acc_list) and plots the curve.
    """
    if k_values is None:
        # default sweep: 5 to min(50, n_features) step 5
        n_feats = X.shape[1] if not isinstance(X, pd.DataFrame) else X.shape[1]
        k_values = list(range(5, max(6, min(50, n_feats))+1, 5))

    if estimator is None:
        estimator = _default_estimator()

    k_list, acc_list = [], []

    for k in k_values:
        # run filter pipeline with MI-KBest=k
        Xf, names_f, _ = filter_select(
            X, y,
            feature_names=list(X.columns) if isinstance(X, pd.DataFrame) else None,
            variance_thresh=variance_thresh,
            corr_thresh=corr_thresh,
            k_best=k,
            scale_for_mi=scale_for_mi
        )
        # optional scaling for the model
        X_model = StandardScaler().fit_transform(Xf) if scale_inputs else Xf
        cv = _cv(n_splits=cv_splits)
        acc = cross_val_score(clone(estimator), X_model, y, cv=cv, scoring=scoring, n_jobs=-1).mean()
        k_list.append(len(names_f))
        acc_list.append(acc)

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(k_list, acc_list, marker='o')
    plt.xlabel("Number of features (K)")
    plt.ylabel(f"CV {scoring}")
    plt.title("Model performance vs. K (Mutual Information K-Best)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.show()

    return k_list, acc_list

# -------------------------
# PCA variance helper (optional)
# -------------------------

def pca_explained_variance(
    X,
    scale_inputs=True,
    n_components=None,
    show_plot=True
):
    """
    Plot cumulative explained variance from PCA (classification-agnostic).
    Use to decide how many PCA components to keep in PCA-based pipelines.
    """
    if isinstance(X, pd.DataFrame):
        X_mat = X.values
    else:
        X_mat = np.asarray(X)

    X_proc = StandardScaler().fit_transform(X_mat) if scale_inputs else X_mat
    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_proc)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)

    if show_plot:
        plt.figure(figsize=(6,4))
        plt.plot(np.arange(1, len(cum)+1), cum, marker='o')
        plt.xlabel("Number of PCA components")
        plt.ylabel("Cumulative explained variance")
        plt.title("PCA cumulative explained variance")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.show()

    return evr, cum, pca

# -------------------------
# Example usage (uncomment and adapt to your data)
# -------------------------
# X = ...  # DataFrame or ndarray of features
# y = ...  # class labels
#
# # 1) FILTERS (MI-only as requested)
# Xf, names_f, report = filter_select(
#     X, y,
#     variance_thresh=0.0,   # drop only exactly-constant features
#     corr_thresh=0.90,      # tighten if you want fewer redundant features
#     k_best=40,             # keep top-40 by mutual information
#     scale_for_mi=True
# )
# print(report)
#
# # Optional: visualize correlation after variance filtering
# # plot_correlation_heatmap(pd.DataFrame(Xf, columns=names_f))
#
# # 2a) WRAPPER with RFE + Logistic Regression (valid for RFE)
# Xw_lr, names_lr, info_lr = wrapper_select(
#     pd.DataFrame(Xf, columns=names_f), y,
#     method="rfe",
#     n_features_to_select=15,
#     estimator=LogisticRegression(max_iter=2000, solver="liblinear"),
#     scoring="accuracy",
#     cv_splits=5,
#     scale_inputs=True
# )
# print("RFE + LR selected:", names_lr, "CV acc:", info_lr["cv_score"]))
#
# # 2b) WRAPPER with SFS + KNN (use SFS for KNN/NB)
# Xw_knn, names_knn, info_knn = wrapper_select(
#     pd.DataFrame(Xf, columns=names_f), y,
#     method="sfs_forward",          # or "sfs_backward"
#     n_features_to_select=15,
#     estimator=KNeighborsClassifier(n_neighbors=5),
#     scoring="accuracy",
#     cv_splits=5,
#     scale_inputs=True
# )
# print("SFS + KNN selected:", names_knn, "CV acc:", info_knn["cv_score"]))
#
# # 2c) WRAPPER with SFS + Naive Bayes
# Xw_nb, names_nb, info_nb = wrapper_select(
#     pd.DataFrame(Xf, columns=names_f), y,
#     method="sfs_forward",
#     n_features_to_select=15,
#     estimator=GaussianNB(),
#     scoring="accuracy",
#     cv_splits=5,
#     scale_inputs=True
# )
# print("SFS + NB selected:", names_nb, "CV acc:", info_nb["cv_score"]))
#
# # 2d) WRAPPER with RFE + Decision Tree (valid for RFE)
# Xw_dt, names_dt, info_dt = wrapper_select(
#     pd.DataFrame(Xf, columns=names_f), y,
#     method="rfe",
#     n_features_to_select=15,
#     estimator=DecisionTreeClassifier(random_state=42),
#     scoring="accuracy",
#     cv_splits=5,
#     scale_inputs=False  # trees don't need scaling
# )
# print("RFE + DT selected:", names_dt, "CV acc:", info_dt["cv_score"]))
#
# # Accuracy vs K to justify K choice (nice for reports)
# # ks, accs = accuracy_vs_k_MI(X, y, estimator=LogisticRegression(max_iter=2000, solver="liblinear"))
#
# # Optional: PCA variance curve (if you also explore PCA-based approach)
# # evr, cum, pca = pca_explained_variance(X, scale_inputs=True, n_components=None, show_plot=True)
