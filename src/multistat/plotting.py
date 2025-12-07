"""
Plotting Utilities for Multivariate Statistics
===============================================

Visualization functions for multivariate data analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2


def plot_scatter_matrix(X, var_names=None, figsize=(10, 10), **kwargs):
    """
    Create a scatter plot matrix (pairs plot).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    var_names : list of str, optional
        Variable names.
    figsize : tuple, default=(10, 10)
        Figure size.

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    X = np.asarray(X)
    n, p = X.shape

    if var_names is None:
        var_names = [f"X{i + 1}" for i in range(p)]

    fig, axes = plt.subplots(p, p, figsize=figsize)

    for i in range(p):
        for j in range(p):
            ax = axes[i, j]
            if i == j:
                # Histogram on diagonal
                ax.hist(X[:, i], bins=20, edgecolor="black", alpha=0.7)
            else:
                # Scatter plot off-diagonal
                ax.scatter(X[:, j], X[:, i], alpha=0.5, s=10, **kwargs)

            if i == p - 1:
                ax.set_xlabel(var_names[j])
            if j == 0:
                ax.set_ylabel(var_names[i])

    plt.tight_layout()
    return fig, axes


def plot_bivariate_normal(mean, cov, ax=None, n_std=2, n_points=100, **kwargs):
    """
    Plot bivariate normal distribution contours.

    Parameters
    ----------
    mean : array-like of shape (2,)
        Mean vector.
    cov : array-like of shape (2, 2)
        Covariance matrix.
    ax : matplotlib axes, optional
        Axes to plot on.
    n_std : float, default=2
        Number of standard deviations for contour.
    n_points : int, default=100
        Number of points for contour.

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots()

    mean = np.asarray(mean)
    cov = np.asarray(cov)

    # Eigendecomposition for ellipse
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    # Ellipse parameters
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigenvalues)

    from matplotlib.patches import Ellipse

    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, **kwargs)
    ax.add_patch(ellipse)

    # Plot mean
    ax.plot(mean[0], mean[1], "k+", markersize=10)

    return ax


def plot_confidence_ellipse(X, ax=None, confidence=0.95, **kwargs):
    """
    Plot confidence ellipse for bivariate data.

    Parameters
    ----------
    X : array-like of shape (n_samples, 2)
        Bivariate data.
    ax : matplotlib axes, optional
        Axes to plot on.
    confidence : float, default=0.95
        Confidence level.

    Returns
    -------
    ax : matplotlib axes
    """
    X = np.asarray(X)
    mean = np.mean(X, axis=0)
    cov = np.cov(X, rowvar=False)

    # Chi-squared critical value
    chi2_val = chi2.ppf(confidence, df=2)
    n_std = np.sqrt(chi2_val)

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(X[:, 0], X[:, 1], alpha=0.5, s=20)
    plot_bivariate_normal(mean, cov, ax=ax, n_std=n_std, **kwargs)

    return ax


def plot_pca_variance(pca_result, ax=None, cumulative=True):
    """
    Plot PCA variance explained (scree plot).

    Parameters
    ----------
    pca_result : dict
        Output from pca().
    ax : matplotlib axes, optional
        Axes to plot on.
    cumulative : bool, default=True
        Also plot cumulative variance.

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    var_ratio = pca_result["variance_ratio"]
    n_comp = len(var_ratio)
    x = np.arange(1, n_comp + 1)

    # Individual variance
    ax.bar(x, var_ratio, alpha=0.7, label="Individual")

    if cumulative:
        cum_var = pca_result["cumulative_variance"]
        ax.plot(x, cum_var, "ro-", label="Cumulative")
        ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5)

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Proportion of Variance Explained")
    ax.set_title("PCA Scree Plot")
    ax.set_xticks(x)
    ax.legend()

    return ax


def plot_pca_loadings(pca_result, components=(0, 1), var_names=None, ax=None):
    """
    Plot PCA loadings (biplot arrows).

    Parameters
    ----------
    pca_result : dict
        Output from pca().
    components : tuple, default=(0, 1)
        Which components to plot.
    var_names : list, optional
        Variable names.
    ax : matplotlib axes, optional
        Axes to plot on.

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    loadings = pca_result["loadings"]
    p = loadings.shape[0]

    if var_names is None:
        var_names = [f"X{i + 1}" for i in range(p)]

    pc1, pc2 = components

    for i, name in enumerate(var_names):
        ax.arrow(
            0,
            0,
            loadings[i, pc1],
            loadings[i, pc2],
            head_width=0.05,
            head_length=0.02,
            fc="blue",
            ec="blue",
        )
        ax.text(loadings[i, pc1] * 1.1, loadings[i, pc2] * 1.1, name)

    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel(f"PC{pc1 + 1}")
    ax.set_ylabel(f"PC{pc2 + 1}")
    ax.set_title("PCA Loadings Plot")

    # Equal aspect ratio
    ax.set_aspect("equal")

    return ax


def plot_lda_projection(lda_result, X, y, components=(0, 1), ax=None):
    """
    Plot LDA projection of data.

    Parameters
    ----------
    lda_result : dict
        Output from lda().
    X : array-like
        Original data.
    y : array-like
        Class labels.
    components : tuple, default=(0, 1)
        Which discriminant functions to plot.
    ax : matplotlib axes, optional
        Axes to plot on.

    Returns
    -------
    ax : matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    scores = lda_result["scores"]
    classes = np.unique(y)

    ld1, ld2 = components

    for c in classes:
        mask = y == c
        if scores.shape[1] > 1:
            ax.scatter(scores[mask, ld1], scores[mask, ld2], label=f"Class {c}", alpha=0.7)
        else:
            ax.scatter(scores[mask, 0], np.zeros(np.sum(mask)), label=f"Class {c}", alpha=0.7)

    ax.set_xlabel(f"LD{ld1 + 1}")
    if scores.shape[1] > 1:
        ax.set_ylabel(f"LD{ld2 + 1}")
    ax.set_title("LDA Projection")
    ax.legend()

    return ax


def plot_correlation_heatmap(X, var_names=None, ax=None, cmap="RdBu_r", **kwargs):
    """
    Plot correlation matrix as heatmap.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    var_names : list, optional
        Variable names.
    ax : matplotlib axes, optional
        Axes to plot on.
    cmap : str, default='RdBu_r'
        Colormap.

    Returns
    -------
    ax : matplotlib axes
    """
    X = np.asarray(X)
    p = X.shape[1]

    if var_names is None:
        var_names = [f"X{i + 1}" for i in range(p)]

    corr = np.corrcoef(X, rowvar=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, **kwargs)

    ax.set_xticks(range(p))
    ax.set_yticks(range(p))
    ax.set_xticklabels(var_names, rotation=45, ha="right")
    ax.set_yticklabels(var_names)

    # Add correlation values
    for i in range(p):
        for j in range(p):
            color = "white" if abs(corr[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", color=color)

    plt.colorbar(im, ax=ax, label="Correlation")
    ax.set_title("Correlation Matrix")

    return ax


def plot_residuals(result, ax=None):
    """
    Diagnostic plots for regression residuals.

    Parameters
    ----------
    result : dict
        Output from linear_regression().
    ax : matplotlib axes or None
        If None, creates 2x2 subplot.

    Returns
    -------
    fig, axes : matplotlib figure and axes
    """
    residuals = result["residuals"]
    fitted = result["fitted"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Residuals vs Fitted
    axes[0, 0].scatter(fitted, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color="red", linestyle="--")
    axes[0, 0].set_xlabel("Fitted values")
    axes[0, 0].set_ylabel("Residuals")
    axes[0, 0].set_title("Residuals vs Fitted")

    # Q-Q plot
    from scipy import stats

    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Normal Q-Q")

    # Scale-Location
    std_resid = residuals / np.std(residuals)
    axes[1, 0].scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.5)
    axes[1, 0].set_xlabel("Fitted values")
    axes[1, 0].set_ylabel("√|Standardized Residuals|")
    axes[1, 0].set_title("Scale-Location")

    # Histogram of residuals
    axes[1, 1].hist(residuals, bins=20, edgecolor="black", alpha=0.7)
    axes[1, 1].set_xlabel("Residuals")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Histogram of Residuals")

    plt.tight_layout()
    return fig, axes
