"""
Descriptive Statistics for Multivariate Data
=============================================

Functions for computing covariance matrices, correlation matrices,
and summary statistics for multivariate datasets.
"""

import numpy as np
import pandas as pd


def multivariate_mean(X):
    """
    Compute the multivariate mean (column-wise).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix where rows are observations and columns are variables.

    Returns
    -------
    mean : ndarray of shape (n_features,)
        Mean vector.

    Examples
    --------
    >>> X = np.array([[1, 0], [2, 4], [3, 5]])
    >>> multivariate_mean(X)
    array([2., 3.])
    """
    X = np.asarray(X)
    return np.mean(X, axis=0)


def covariance_matrix(X, ddof=1):
    """
    Compute the sample covariance matrix.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    ddof : int, default=1
        Delta degrees of freedom. Use ddof=1 for sample covariance (default),
        ddof=0 for population covariance.

    Returns
    -------
    S : ndarray of shape (n_features, n_features)
        Covariance matrix.

    Examples
    --------
    >>> X = np.array([[1, 0, 2], [3, 4, 5], [1, 5, 9]])
    >>> covariance_matrix(X.T)  # Variables as columns
    """
    X = np.asarray(X)
    return np.cov(X, rowvar=False, ddof=ddof)


def correlation_matrix(X):
    """
    Compute the sample correlation matrix.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.

    Returns
    -------
    R : ndarray of shape (n_features, n_features)
        Correlation matrix.

    Examples
    --------
    >>> X = np.array([[1, 0, 2], [3, 4, 5], [1, 5, 9]])
    >>> correlation_matrix(X.T)
    """
    X = np.asarray(X)
    return np.corrcoef(X, rowvar=False)


def pooled_covariance(X1, X2):
    """
    Compute the pooled covariance matrix for two groups.

    Parameters
    ----------
    X1 : array-like of shape (n1, p)
        Data from group 1.
    X2 : array-like of shape (n2, p)
        Data from group 2.

    Returns
    -------
    S_pooled : ndarray of shape (p, p)
        Pooled covariance matrix.

    Notes
    -----
    The pooled covariance is:
        S_pooled = ((n1-1)*S1 + (n2-1)*S2) / (n1 + n2 - 2)
    """
    X1, X2 = np.asarray(X1), np.asarray(X2)
    n1, n2 = len(X1), len(X2)
    S1 = covariance_matrix(X1)
    S2 = covariance_matrix(X2)
    return ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)


def summary_stats(X, var_names=None):
    """
    Compute comprehensive summary statistics.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    var_names : list of str, optional
        Variable names for output.

    Returns
    -------
    summary : DataFrame
        Summary statistics including mean, std, min, max, etc.

    Examples
    --------
    >>> X = np.random.randn(100, 3)
    >>> summary_stats(X, var_names=['X1', 'X2', 'X3'])
    """
    X = np.asarray(X)
    n, p = X.shape

    if var_names is None:
        var_names = [f"X{i + 1}" for i in range(p)]

    summary = pd.DataFrame(
        {
            "n": n,
            "mean": np.mean(X, axis=0),
            "std": np.std(X, axis=0, ddof=1),
            "min": np.min(X, axis=0),
            "Q1": np.percentile(X, 25, axis=0),
            "median": np.median(X, axis=0),
            "Q3": np.percentile(X, 75, axis=0),
            "max": np.max(X, axis=0),
        },
        index=var_names,
    )

    return summary


def standardize(X):
    """
    Standardize data to zero mean and unit variance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.

    Returns
    -------
    Z : ndarray of shape (n_samples, n_features)
        Standardized data.
    mean : ndarray
        Original means.
    std : ndarray
        Original standard deviations.
    """
    X = np.asarray(X)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0, ddof=1)
    Z = (X - mean) / std
    return Z, mean, std


def mahalanobis_distance(x, mean, cov):
    """
    Compute the Mahalanobis distance.

    Parameters
    ----------
    x : array-like of shape (n_features,) or (n_samples, n_features)
        Point(s) to compute distance for.
    mean : array-like of shape (n_features,)
        Mean vector.
    cov : array-like of shape (n_features, n_features)
        Covariance matrix.

    Returns
    -------
    d : float or ndarray
        Mahalanobis distance(s).
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    cov_inv = np.linalg.inv(cov)

    if x.ndim == 1:
        diff = x - mean
        return np.sqrt(diff @ cov_inv @ diff)
    else:
        diff = x - mean
        return np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))


# ============================================================================
# LINEAR TRANSFORMATIONS
# ============================================================================


def linear_transform_mean(A, mu, b=None):
    """
    Calculate the mean of a linear transformation.

    E[AX + b] = A × μ + b

    Parameters
    ----------
    A : array-like of shape (m, n)
        Transformation matrix.
    mu : array-like of shape (n,)
        Mean vector of X.
    b : array-like of shape (m,), optional
        Constant vector to add.

    Returns
    -------
    mean : ndarray
        Mean of the transformed variable.

    Examples
    --------
    >>> mu = np.array([1, 2, 3])  # E[X, Y, Z]
    >>> A = np.array([[1, -1, 0], [0, 1, -1]])  # S = X-Y, T = Y-Z
    >>> linear_transform_mean(A, mu)
    array([-1, -1])
    """
    A = np.asarray(A)
    mu = np.asarray(mu)

    result = A @ mu
    if b is not None:
        result = result + np.asarray(b)

    return result


def linear_transform_cov(A, Sigma):
    """
    Calculate the covariance matrix of a linear transformation.

    D[AX] = A × Σ × A'

    Parameters
    ----------
    A : array-like of shape (m, n)
        Transformation matrix.
    Sigma : array-like of shape (n, n)
        Covariance matrix of X.

    Returns
    -------
    cov : ndarray of shape (m, m)
        Covariance matrix of the transformed variable.

    Examples
    --------
    >>> Sigma = np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])
    >>> A = np.array([[1, -1, 0], [0, 1, -1]])  # S = X-Y, T = Y-Z
    >>> linear_transform_cov(A, Sigma)
    """
    A = np.asarray(A)
    Sigma = np.asarray(Sigma)

    return A @ Sigma @ A.T


def linear_transform(A, mu, Sigma, b=None):
    """
    Calculate both mean and covariance of a linear transformation.

    For Y = AX + b:
    - E[Y] = A × μ + b
    - D[Y] = A × Σ × A'

    Parameters
    ----------
    A : array-like of shape (m, n)
        Transformation matrix.
    mu : array-like of shape (n,)
        Mean vector of X.
    Sigma : array-like of shape (n, n)
        Covariance matrix of X.
    b : array-like of shape (m,), optional
        Constant vector.

    Returns
    -------
    result : dict
        - mean: Mean of transformed variable
        - cov: Covariance matrix of transformed variable

    Examples
    --------
    >>> mu = np.array([1, 2, 3])
    >>> Sigma = np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])
    >>> A = np.array([[1, -1, 0], [0, 1, -1]])
    >>> result = linear_transform(A, mu, Sigma)
    >>> print(f"E[S, T] = {result['mean']}")
    >>> print(f"D[S, T] = \\n{result['cov']}")
    """
    return {
        "mean": linear_transform_mean(A, mu, b),
        "cov": linear_transform_cov(A, Sigma),
    }


# ============================================================================
# CONDITIONAL DISTRIBUTIONS
# ============================================================================


def conditional_mean(mu, Sigma, idx_given, x_given, idx_target=None):
    """
    Calculate the conditional mean E(X_1 | X_2 = x_2).

    For multivariate normal:
    E(X_1 | X_2) = μ_1 + Σ_12 × Σ_22^(-1) × (X_2 - μ_2)

    Parameters
    ----------
    mu : array-like
        Mean vector of the full distribution.
    Sigma : array-like
        Covariance matrix of the full distribution.
    idx_given : int or list of int
        Index/indices of the conditioning variables (X_2).
    x_given : float or array-like
        Observed values of the conditioning variables.
    idx_target : int or list of int, optional
        Index/indices of target variables (X_1). If None, uses all non-given indices.

    Returns
    -------
    result : dict
        - conditional_mean: E(X_1 | X_2 = x_2)
        - formula_coefficients: The coefficients for the formula

    Examples
    --------
    >>> mu = np.array([1, 2, 3])
    >>> Sigma = np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])
    >>> # E(X | Y=y) where Y is index 1
    >>> result = conditional_mean(mu, Sigma, idx_given=1, x_given=2.5, idx_target=0)
    >>> print(f"E(X|Y=2.5) = {result['conditional_mean']}")
    """
    mu = np.asarray(mu)
    Sigma = np.asarray(Sigma)

    # Convert to lists for indexing
    if isinstance(idx_given, int):
        idx_given = [idx_given]
    idx_given = list(idx_given)

    if idx_target is None:
        idx_target = [i for i in range(len(mu)) if i not in idx_given]
    elif isinstance(idx_target, int):
        idx_target = [idx_target]
    idx_target = list(idx_target)

    x_given = np.atleast_1d(x_given)

    # Extract submatrices
    mu_1 = mu[idx_target]
    mu_2 = mu[idx_given]

    # Sigma_11 not needed for conditional mean, only for variance
    Sigma_12 = Sigma[np.ix_(idx_target, idx_given)]
    Sigma_22 = Sigma[np.ix_(idx_given, idx_given)]

    # Calculate conditional mean
    Sigma_22_inv = np.linalg.inv(Sigma_22) if Sigma_22.size > 1 else 1.0 / Sigma_22

    if Sigma_22.size == 1:
        coef = Sigma_12.flatten() / Sigma_22.flatten()[0]
        cond_mean = mu_1 + coef * (x_given[0] - mu_2[0])
    else:
        coef = Sigma_12 @ Sigma_22_inv
        cond_mean = mu_1 + coef @ (x_given - mu_2)

    return {
        "conditional_mean": cond_mean.flatten()
        if len(idx_target) > 1
        else float(cond_mean.flatten()[0]),
        "formula_coefficients": coef,
        "intercept": mu_1 - coef @ mu_2 if Sigma_22.size > 1 else mu_1 - coef * mu_2,
    }


def conditional_variance(Sigma, idx_given, idx_target=None):
    """
    Calculate the conditional covariance matrix D(X_1 | X_2).

    For multivariate normal:
    D(X_1 | X_2) = Σ_11 - Σ_12 × Σ_22^(-1) × Σ_21

    Note: This does not depend on the value of X_2!

    Parameters
    ----------
    Sigma : array-like
        Covariance matrix of the full distribution.
    idx_given : int or list of int
        Index/indices of the conditioning variables (X_2).
    idx_target : int or list of int, optional
        Index/indices of target variables (X_1).

    Returns
    -------
    result : dict
        - conditional_cov: D(X_1 | X_2)
        - conditional_var: Diagonal (variances) if multiple targets

    Examples
    --------
    >>> Sigma = np.array([[1, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 1]])
    >>> # D(X, Z | Y)
    >>> result = conditional_variance(Sigma, idx_given=1, idx_target=[0, 2])
    >>> print(f"D(X,Z|Y) = \\n{result['conditional_cov']}")
    """
    Sigma = np.asarray(Sigma)

    # Convert to lists for indexing
    if isinstance(idx_given, int):
        idx_given = [idx_given]
    idx_given = list(idx_given)

    if idx_target is None:
        idx_target = [i for i in range(len(Sigma)) if i not in idx_given]
    elif isinstance(idx_target, int):
        idx_target = [idx_target]
    idx_target = list(idx_target)

    # Extract submatrices
    Sigma_11 = Sigma[np.ix_(idx_target, idx_target)]
    Sigma_12 = Sigma[np.ix_(idx_target, idx_given)]
    Sigma_21 = Sigma[np.ix_(idx_given, idx_target)]
    Sigma_22 = Sigma[np.ix_(idx_given, idx_given)]

    # Calculate conditional covariance
    if Sigma_22.size == 1:
        Sigma_22_inv = 1.0 / Sigma_22.flatten()[0]
        cond_cov = Sigma_11 - Sigma_12 @ Sigma_21 * Sigma_22_inv
    else:
        Sigma_22_inv = np.linalg.inv(Sigma_22)
        cond_cov = Sigma_11 - Sigma_12 @ Sigma_22_inv @ Sigma_21

    return {
        "conditional_cov": cond_cov,
        "conditional_var": np.diag(cond_cov) if cond_cov.ndim > 1 else cond_cov,
    }


def conditional_distribution(mu, Sigma, idx_given, x_given, idx_target=None):
    """
    Calculate both conditional mean and covariance.

    For X = [X_1, X_2]' ~ N(μ, Σ):
    X_1 | X_2 = x_2 ~ N(μ_cond, Σ_cond)

    where:
    - μ_cond = μ_1 + Σ_12 Σ_22^(-1) (x_2 - μ_2)
    - Σ_cond = Σ_11 - Σ_12 Σ_22^(-1) Σ_21

    Parameters
    ----------
    mu : array-like
        Mean vector.
    Sigma : array-like
        Covariance matrix.
    idx_given : int or list of int
        Index/indices of conditioning variables.
    x_given : float or array-like
        Observed values.
    idx_target : int or list of int, optional
        Target variable indices.

    Returns
    -------
    result : dict
        - mean: Conditional mean
        - cov: Conditional covariance
        - var: Conditional variance(s)

    Examples
    --------
    >>> mu = np.array([1, 2, 3])
    >>> rho = 0.5
    >>> Sigma = np.array([[1, rho, rho**2],
    ...                   [rho, 1, rho],
    ...                   [rho**2, rho, 1]])
    >>> # Distribution of (X, Z) given Y = 2
    >>> result = conditional_distribution(mu, Sigma, idx_given=1, x_given=2, idx_target=[0, 2])
    """
    mean_result = conditional_mean(mu, Sigma, idx_given, x_given, idx_target)
    var_result = conditional_variance(Sigma, idx_given, idx_target)

    return {
        "mean": mean_result["conditional_mean"],
        "cov": var_result["conditional_cov"],
        "var": var_result["conditional_var"],
        "formula_coefficients": mean_result["formula_coefficients"],
    }


# ============================================================================
# HELPER FUNCTIONS FOR COMMON EXAM PATTERNS
# ============================================================================


def covariance_xy(x_var, y_var, cov_xy, a=1, b=1, c=0, d=0):
    """
    Calculate Cov(aX + c, bY + d) = ab × Cov(X, Y).

    Useful for exam questions like "What is Cov(X, X-Y)?"

    Parameters
    ----------
    x_var : float
        Var(X)
    y_var : float
        Var(Y)
    cov_xy : float
        Cov(X, Y)
    a, b : float
        Coefficients
    c, d : float
        Constants (don't affect covariance)

    Returns
    -------
    float
        The covariance

    Examples
    --------
    >>> # Cov(X, X-Y) = Cov(X, X) - Cov(X, Y) = Var(X) - Cov(X, Y)
    >>> covariance_xy(x_var=1, y_var=1, cov_xy=0.5, a=1, b=1)  # Cov(X, X)
    1.0
    """
    # This is a simplified version - for full cases use linear_transform_cov
    return a * b * cov_xy


def variance_difference(var_x, var_y, cov_xy):
    """
    Calculate Var(X - Y) = Var(X) + Var(Y) - 2×Cov(X,Y).

    Parameters
    ----------
    var_x : float
        Variance of X.
    var_y : float
        Variance of Y.
    cov_xy : float
        Covariance of X and Y.

    Returns
    -------
    float
        Var(X - Y)

    Examples
    --------
    >>> variance_difference(var_x=1, var_y=1, cov_xy=0.5)
    1.0
    """
    return var_x + var_y - 2 * cov_xy


def variance_sum(var_x, var_y, cov_xy):
    """
    Calculate Var(X + Y) = Var(X) + Var(Y) + 2×Cov(X,Y).

    Parameters
    ----------
    var_x : float
        Variance of X.
    var_y : float
        Variance of Y.
    cov_xy : float
        Covariance of X and Y.

    Returns
    -------
    float
        Var(X + Y)
    """
    return var_x + var_y + 2 * cov_xy
