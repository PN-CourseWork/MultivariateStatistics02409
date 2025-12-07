"""
Multivariate Analysis Methods
=============================

Principal Component Analysis, Discriminant Analysis,
Factor Analysis, and Canonical Correlation Analysis.
"""

import numpy as np
import pandas as pd
from scipy import linalg


def pca(X, n_components=None, standardize=True):
    """
    Principal Component Analysis.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    n_components : int, optional
        Number of components to keep. If None, keep all.
    standardize : bool, default=True
        Standardize data before PCA (use correlation matrix).
        If False, use covariance matrix.

    Returns
    -------
    result : dict
        PCA results including:
        - loadings: Principal component loadings (eigenvectors)
        - scores: Transformed data (principal component scores)
        - eigenvalues: Variance explained by each component
        - variance_ratio: Proportion of variance explained
        - cumulative_variance: Cumulative variance explained

    Examples
    --------
    >>> X = np.random.randn(100, 5)
    >>> result = pca(X, n_components=2)
    >>> print(result['variance_ratio'])
    """
    X = np.asarray(X)
    n, p = X.shape

    # Center (and optionally standardize) data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    if standardize:
        std = np.std(X, axis=0, ddof=1)
        X_centered = X_centered / std
        # Use correlation matrix
        cov = np.corrcoef(X, rowvar=False)
    else:
        # Use covariance matrix
        cov = np.cov(X, rowvar=False, ddof=1)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select components
    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

    # Compute scores
    scores = X_centered @ eigenvectors

    # Variance explained
    variance_ratio = eigenvalues / cov.trace()
    cumulative_variance = np.cumsum(variance_ratio)

    # Create loadings DataFrame
    n_comp = len(eigenvalues)
    loadings_df = pd.DataFrame(
        eigenvectors,
        columns=[f"PC{i + 1}" for i in range(n_comp)],
        index=[f"X{i + 1}" for i in range(p)],
    )

    return {
        "loadings": eigenvectors,
        "loadings_df": loadings_df,
        "scores": scores,
        "eigenvalues": eigenvalues,
        "variance_ratio": variance_ratio,
        "cumulative_variance": cumulative_variance,
        "mean": mean,
        "std": std if standardize else None,
        "n_components": n_comp,
    }


def lda(X, y, n_components=None):
    """
    Linear Discriminant Analysis.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    y : array-like of shape (n_samples,)
        Class labels.
    n_components : int, optional
        Number of discriminant functions. Max is min(p, k-1) where k is number of classes.

    Returns
    -------
    result : dict
        LDA results including:
        - coefficients: Discriminant function coefficients
        - scores: Discriminant scores for training data
        - eigenvalues: Eigenvalues (importance of each function)
        - group_means: Mean for each group
        - pooled_cov: Pooled within-group covariance

    Examples
    --------
    >>> X = np.vstack([np.random.randn(50, 3), np.random.randn(50, 3) + 2])
    >>> y = np.repeat([0, 1], 50)
    >>> result = lda(X, y)
    """
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    n, p = X.shape
    k = len(classes)

    # Maximum components
    max_components = min(p, k - 1)
    if n_components is None:
        n_components = max_components
    n_components = min(n_components, max_components)

    # Overall mean
    mean_overall = np.mean(X, axis=0)

    # Group means and sizes
    group_means = {}
    group_sizes = {}
    for c in classes:
        mask = y == c
        group_means[c] = np.mean(X[mask], axis=0)
        group_sizes[c] = np.sum(mask)

    # Between-class scatter matrix
    S_B = np.zeros((p, p))
    for c in classes:
        diff = group_means[c] - mean_overall
        S_B += group_sizes[c] * np.outer(diff, diff)

    # Within-class scatter matrix
    S_W = np.zeros((p, p))
    for c in classes:
        X_c = X[y == c]
        X_c_centered = X_c - group_means[c]
        S_W += X_c_centered.T @ X_c_centered

    # Solve generalized eigenvalue problem
    # S_B v = λ S_W v
    try:
        eigenvalues, eigenvectors = linalg.eigh(S_B, S_W)
    except np.linalg.LinAlgError:
        # Add small regularization if singular
        S_W_reg = S_W + 1e-6 * np.eye(p)
        eigenvalues, eigenvectors = linalg.eigh(S_B, S_W_reg)

    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx][:n_components]
    eigenvectors = eigenvectors[:, idx][:, :n_components]

    # Compute scores
    X_centered = X - mean_overall
    scores = X_centered @ eigenvectors

    # Proportion of trace
    total_eigen = np.sum(np.abs(eigenvalues))
    proportion = np.abs(eigenvalues) / total_eigen if total_eigen > 0 else eigenvalues

    return {
        "coefficients": eigenvectors,
        "scores": scores,
        "eigenvalues": eigenvalues,
        "proportion": proportion,
        "group_means": group_means,
        "overall_mean": mean_overall,
        "pooled_cov": S_W / (n - k),
        "classes": classes,
        "n_components": n_components,
    }


def qda(X, y):
    """
    Quadratic Discriminant Analysis (estimates separate covariances).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    y : array-like of shape (n_samples,)
        Class labels.

    Returns
    -------
    result : dict
        QDA results including group means, covariances, and priors.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    n = len(y)

    group_means = {}
    group_covs = {}
    priors = {}

    for c in classes:
        mask = y == c
        X_c = X[mask]
        group_means[c] = np.mean(X_c, axis=0)
        group_covs[c] = np.cov(X_c, rowvar=False, ddof=1)
        priors[c] = np.sum(mask) / n

    return {
        "group_means": group_means,
        "group_covs": group_covs,
        "priors": priors,
        "classes": classes,
    }


def predict_lda(result, X_new):
    """
    Predict class labels using LDA.

    Parameters
    ----------
    result : dict
        Output from lda().
    X_new : array-like
        New data to classify.

    Returns
    -------
    predictions : ndarray
        Predicted class labels.
    scores : ndarray
        Discriminant scores.
    """
    X_new = np.asarray(X_new)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    X_centered = X_new - result["overall_mean"]
    scores = X_centered @ result["coefficients"]

    # Simple nearest-centroid classification in discriminant space
    group_scores = {}
    for c in result["classes"]:
        mean_c = result["group_means"][c] - result["overall_mean"]
        group_scores[c] = mean_c @ result["coefficients"]

    predictions = []
    for score in scores:
        distances = {c: np.sum((score - gs) ** 2) for c, gs in group_scores.items()}
        predictions.append(min(distances, key=distances.get))

    return np.array(predictions), scores


def factor_analysis(X, n_factors, method="principal", rotation=None, max_iter=100):
    """
    Factor Analysis.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    n_factors : int
        Number of factors to extract.
    method : str, default='principal'
        Extraction method: 'principal' or 'ml' (maximum likelihood).
    rotation : str, optional
        Rotation method: 'varimax' or None.
    max_iter : int, default=100
        Maximum iterations for iterative methods.

    Returns
    -------
    result : dict
        Factor analysis results including loadings, communalities,
        and uniquenesses.
    """
    X = np.asarray(X)
    n, p = X.shape

    # Correlation matrix (factor analysis works on standardized variables)
    R = np.corrcoef(X, rowvar=False)

    if method == "principal":
        # Principal factor method
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx][:n_factors]
        eigenvectors = eigenvectors[:, idx][:, :n_factors]

        # Loadings
        loadings = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))

    else:
        # Simple principal factor as fallback
        eigenvalues, eigenvectors = np.linalg.eigh(R)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx][:n_factors]
        eigenvectors = eigenvectors[:, idx][:, :n_factors]
        loadings = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))

    # Varimax rotation
    if rotation == "varimax":
        loadings = _varimax(loadings, max_iter=max_iter)

    # Communalities and uniquenesses
    communalities = np.sum(loadings**2, axis=1)
    uniquenesses = 1 - communalities

    # Variance explained
    var_explained = np.sum(loadings**2, axis=0)
    var_ratio = var_explained / p

    # Create loadings DataFrame
    loadings_df = pd.DataFrame(
        loadings,
        columns=[f"Factor{i + 1}" for i in range(n_factors)],
        index=[f"X{i + 1}" for i in range(p)],
    )

    return {
        "loadings": loadings,
        "loadings_df": loadings_df,
        "communalities": communalities,
        "uniquenesses": uniquenesses,
        "var_explained": var_explained,
        "var_ratio": var_ratio,
        "n_factors": n_factors,
    }


def _varimax(loadings, max_iter=100, tol=1e-6):
    """Varimax rotation of factor loadings."""
    p, k = loadings.shape
    rotation = np.eye(k)

    for _ in range(max_iter):
        rotated = loadings @ rotation
        u, s, vt = np.linalg.svd(
            loadings.T @ (rotated**3 - rotated @ np.diag(np.sum(rotated**2, axis=0)) / p)
        )
        rotation_new = u @ vt

        if np.max(np.abs(rotation_new - rotation)) < tol:
            break
        rotation = rotation_new

    return loadings @ rotation


def canonical_correlation(X, Y):
    """
    Canonical Correlation Analysis.

    Parameters
    ----------
    X : array-like of shape (n_samples, p)
        First set of variables.
    Y : array-like of shape (n_samples, q)
        Second set of variables.

    Returns
    -------
    result : dict
        CCA results including canonical correlations and coefficients.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    n = len(X)
    p, q = X.shape[1], Y.shape[1]

    # Center
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    # Covariance matrices
    Sxx = X.T @ X / (n - 1)
    Syy = Y.T @ Y / (n - 1)
    Sxy = X.T @ Y / (n - 1)

    # Solve canonical correlation problem
    Sxx_inv_sqrt = linalg.sqrtm(np.linalg.inv(Sxx))
    Syy_inv_sqrt = linalg.sqrtm(np.linalg.inv(Syy))

    M = Sxx_inv_sqrt @ Sxy @ Syy_inv_sqrt

    U, s, Vt = np.linalg.svd(M)

    # Canonical correlations
    canonical_corr = s[: min(p, q)]

    # Canonical coefficients
    A = Sxx_inv_sqrt @ U[:, : min(p, q)]
    B = Syy_inv_sqrt @ Vt[: min(p, q), :].T

    # Canonical variates
    U_scores = X @ A
    V_scores = Y @ B

    return {
        "canonical_corr": canonical_corr,
        "x_coefficients": A,
        "y_coefficients": B,
        "x_scores": U_scores,
        "y_scores": V_scores,
    }


# ============================================================================
# CLASSIFICATION HELPERS
# ============================================================================


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Compute confusion matrix for classification results.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    labels : list, optional
        List of labels to include in the matrix.

    Returns
    -------
    result : dict
        - matrix: Confusion matrix (rows=true, cols=predicted)
        - labels: Class labels
        - accuracy: Overall accuracy
        - misclassified: Total number misclassified
        - misclassification_rate: Error rate
        - per_class: Dictionary with per-class metrics

    Examples
    --------
    >>> y_true = [0, 0, 0, 1, 1, 1, 1]
    >>> y_pred = [0, 0, 1, 0, 1, 1, 1]
    >>> result = confusion_matrix(y_true, y_pred)
    >>> print(f"Accuracy: {result['accuracy']:.2%}")
    >>> print(f"Misclassified: {result['misclassified']}")
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = list(labels)
    n_labels = len(labels)

    # Build confusion matrix
    cm = np.zeros((n_labels, n_labels), dtype=int)
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            cm[i, j] = np.sum((y_true == true_label) & (y_pred == pred_label))

    # Calculate metrics
    n_total = len(y_true)
    correct = np.sum(y_true == y_pred)
    misclassified = n_total - correct
    accuracy = correct / n_total
    error_rate = misclassified / n_total

    # Per-class metrics
    per_class = {}
    for i, label in enumerate(labels):
        n_true = np.sum(y_true == label)
        n_pred = np.sum(y_pred == label)
        n_correct = cm[i, i]
        per_class[label] = {
            "n_true": n_true,
            "n_pred": n_pred,
            "n_correct": n_correct,
            "n_misclassified": n_true - n_correct,
            "sensitivity": n_correct / n_true if n_true > 0 else 0,  # True positive rate
            "precision": n_correct / n_pred if n_pred > 0 else 0,
        }

    # Create DataFrame
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.index.name = "True"
    cm_df.columns.name = "Predicted"

    return {
        "matrix": cm,
        "matrix_df": cm_df,
        "labels": labels,
        "n_total": n_total,
        "correct": correct,
        "misclassified": misclassified,
        "accuracy": accuracy,
        "misclassification_rate": error_rate,
        "per_class": per_class,
    }


def classification_summary(y_true, y_pred, method_name="Classification"):
    """
    Print a formatted classification summary similar to SAS output.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels.
    method_name : str
        Name of the method (e.g., "LDA", "QDA").

    Returns
    -------
    result : dict
        Classification results.
    """
    result = confusion_matrix(y_true, y_pred)

    print("=" * 60)
    print(f"{method_name} CLASSIFICATION SUMMARY")
    print("=" * 60)
    print()
    print("Confusion Matrix:")
    print(result["matrix_df"].to_string())
    print()
    print(f"Total observations: {result['n_total']}")
    print(f"Correctly classified: {result['correct']} ({result['accuracy']:.2%})")
    print(f"Misclassified: {result['misclassified']} ({result['misclassification_rate']:.2%})")
    print()
    print("Per-class results:")
    for label, metrics in result["per_class"].items():
        print(
            f"  Class {label}: {metrics['n_correct']}/{metrics['n_true']} correct "
            f"({metrics['sensitivity']:.2%} sensitivity)"
        )

    return result


def compare_classifiers(y_true, predictions_dict):
    """
    Compare multiple classifiers on the same data.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    predictions_dict : dict
        Dictionary mapping method names to predictions.
        e.g., {'LDA': y_pred_lda, 'QDA': y_pred_qda}

    Returns
    -------
    result : dict
        Comparison results including misclassification reduction.

    Examples
    --------
    >>> y_true = [0, 0, 0, 1, 1, 1]
    >>> preds = {'LDA': [0, 0, 1, 0, 1, 1], 'QDA': [0, 0, 0, 1, 1, 1]}
    >>> result = compare_classifiers(y_true, preds)
    >>> print(f"QDA reduces misclassification by {result['comparison']['QDA vs LDA']['reduction']}")
    """
    results = {}
    for name, y_pred in predictions_dict.items():
        results[name] = confusion_matrix(y_true, y_pred)

    # Build comparison summary
    summary = pd.DataFrame(
        {
            name: {
                "Accuracy": r["accuracy"],
                "Misclassified": r["misclassified"],
                "Error Rate": r["misclassification_rate"],
            }
            for name, r in results.items()
        }
    ).T

    # Pairwise comparison
    method_names = list(predictions_dict.keys())
    comparison = {}
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            m1, m2 = method_names[i], method_names[j]
            diff = results[m1]["misclassified"] - results[m2]["misclassified"]
            comparison[f"{m2} vs {m1}"] = {
                "reduction": diff,
                "m1_misclassified": results[m1]["misclassified"],
                "m2_misclassified": results[m2]["misclassified"],
            }

    return {
        "individual": results,
        "summary": summary,
        "comparison": comparison,
    }


def predict_qda(result, X_new):
    """
    Predict class labels using QDA.

    Parameters
    ----------
    result : dict
        Output from qda().
    X_new : array-like
        New data to classify.

    Returns
    -------
    predictions : ndarray
        Predicted class labels.
    posteriors : ndarray
        Posterior probabilities for each class.
    """
    X_new = np.asarray(X_new)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    n = len(X_new)
    classes = result["classes"]
    k = len(classes)

    # Calculate discriminant score for each class
    scores = np.zeros((n, k))
    for j, c in enumerate(classes):
        mean_c = result["group_means"][c]
        cov_c = result["group_covs"][c]
        prior_c = result["priors"][c]

        # QDA discriminant function
        cov_inv = np.linalg.inv(cov_c)
        log_det = np.log(np.linalg.det(cov_c))

        for i in range(n):
            diff = X_new[i] - mean_c
            scores[i, j] = -0.5 * log_det - 0.5 * diff @ cov_inv @ diff + np.log(prior_c)

    # Predict class with highest score
    predictions = classes[np.argmax(scores, axis=1)]

    # Convert scores to posteriors (softmax)
    scores_shifted = scores - np.max(scores, axis=1, keepdims=True)
    posteriors = np.exp(scores_shifted) / np.sum(np.exp(scores_shifted), axis=1, keepdims=True)

    return predictions, posteriors


def backward_selection_order(p_values, var_names=None):
    """
    Determine the order of variable elimination in backward selection.

    Parameters
    ----------
    p_values : dict or array-like
        P-values for each variable. Can be dict {var_name: p_value} or array.
    var_names : list, optional
        Variable names if p_values is an array.

    Returns
    -------
    result : dict
        - order: List of variables in elimination order
        - p_values_sorted: P-values sorted by elimination order

    Examples
    --------
    >>> p_values = {'X1': 0.02, 'X2': 0.45, 'X3': 0.001, 'X4': 0.38}
    >>> result = backward_selection_order(p_values)
    >>> print(f"First to eliminate: {result['order'][0]}")  # X2
    """
    if isinstance(p_values, dict):
        var_names = list(p_values.keys())
        p_vals = list(p_values.values())
    else:
        p_vals = list(p_values)
        if var_names is None:
            var_names = [f"X{i + 1}" for i in range(len(p_vals))]

    # Sort by p-value (descending) to get elimination order
    sorted_indices = np.argsort(p_vals)[::-1]
    elimination_order = [var_names[i] for i in sorted_indices]
    p_values_sorted = [p_vals[i] for i in sorted_indices]

    return {
        "order": elimination_order,
        "p_values_sorted": p_values_sorted,
        "first_to_eliminate": elimination_order[0],
        "first_p_value": p_values_sorted[0],
    }


def screeplot_elbow(eigenvalues, threshold=0.1):
    """
    Determine number of components using screeplot elbow method.

    Parameters
    ----------
    eigenvalues : array-like
        Eigenvalues from PCA or factor analysis.
    threshold : float, default=0.1
        Relative drop threshold to detect elbow.

    Returns
    -------
    result : dict
        - n_components: Suggested number of components
        - kaiser_criterion: Number with eigenvalue > 1
        - cumulative_variance: Cumulative variance explained
        - elbow_location: Index of detected elbow

    Examples
    --------
    >>> eigenvalues = [2.806, 2.190, 0.789, 0.469, 0.312]
    >>> result = screeplot_elbow(eigenvalues)
    >>> print(f"Retain {result['n_components']} components")
    """
    eigenvalues = np.asarray(eigenvalues)
    n = len(eigenvalues)

    # Kaiser criterion: eigenvalue > 1
    kaiser = np.sum(eigenvalues > 1)

    # Cumulative variance
    total_var = np.sum(eigenvalues)
    cum_var = np.cumsum(eigenvalues) / total_var

    # Find elbow: largest relative drop
    if n > 1:
        drops = np.diff(eigenvalues) / eigenvalues[:-1]
        elbow = np.argmin(drops) + 1  # +1 because we want components up to the drop
    else:
        elbow = 1

    # 80% variance criterion
    var_80 = np.argmax(cum_var >= 0.8) + 1 if np.any(cum_var >= 0.8) else n

    return {
        "n_components": max(elbow, kaiser),  # Usually take the larger of elbow/Kaiser
        "kaiser_criterion": kaiser,
        "elbow_location": elbow,
        "variance_80_criterion": var_80,
        "eigenvalues": eigenvalues,
        "cumulative_variance": cum_var,
        "proportion_variance": eigenvalues / total_var,
    }
