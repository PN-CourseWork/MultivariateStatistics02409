"""
Hypothesis Testing for Multivariate Data
=========================================

Statistical tests including Hotelling's T², MANOVA, Box's M test,
and various likelihood ratio tests.
"""

import numpy as np
from scipy import stats
from scipy.stats import chi2
from scipy.stats import f as f_dist


def hotellings_t2(X, mu0):
    """
    Hotelling's T² test for one-sample multivariate mean.

    Test H0: μ = μ0 vs H1: μ ≠ μ0

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Sample data.
    mu0 : array-like of shape (n_features,)
        Hypothesized mean vector.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - 'T2': Hotelling's T² statistic
        - 'F': F-statistic
        - 'df1', 'df2': Degrees of freedom
        - 'p_value': p-value
        - 'mean': Sample mean

    Examples
    --------
    >>> X = np.random.randn(50, 3)
    >>> result = hotellings_t2(X, [0, 0, 0])
    >>> print(f"T² = {result['T2']:.3f}, p = {result['p_value']:.4f}")
    """
    X = np.asarray(X)
    mu0 = np.asarray(mu0)
    n, p = X.shape

    x_bar = np.mean(X, axis=0)
    S = np.cov(X, rowvar=False, ddof=1)
    S_inv = np.linalg.inv(S)

    diff = x_bar - mu0
    T2 = n * diff @ S_inv @ diff

    # Convert to F-statistic
    F = T2 * (n - p) / (p * (n - 1))
    df1, df2 = p, n - p
    p_value = 1 - f_dist.cdf(F, df1, df2)

    return {
        "T2": T2,
        "F": F,
        "df1": df1,
        "df2": df2,
        "p_value": p_value,
        "mean": x_bar,
    }


def two_sample_t2(X1, X2, equal_cov=True):
    """
    Two-sample Hotelling's T² test.

    Test H0: μ1 = μ2 vs H1: μ1 ≠ μ2

    Parameters
    ----------
    X1, X2 : array-like
        Sample data from two groups.
    equal_cov : bool, default=True
        Assume equal covariance matrices.

    Returns
    -------
    result : dict
        Dictionary with T², F-statistic, degrees of freedom, and p-value.

    Examples
    --------
    >>> X1 = np.random.randn(30, 3)
    >>> X2 = np.random.randn(30, 3) + 0.5
    >>> result = two_sample_t2(X1, X2)
    """
    X1, X2 = np.asarray(X1), np.asarray(X2)
    n1, p = X1.shape
    n2 = len(X2)

    x1_bar = np.mean(X1, axis=0)
    x2_bar = np.mean(X2, axis=0)
    diff = x1_bar - x2_bar

    if equal_cov:
        # Pooled covariance
        S1 = np.cov(X1, rowvar=False, ddof=1)
        S2 = np.cov(X2, rowvar=False, ddof=1)
        S_pooled = ((n1 - 1) * S1 + (n2 - 1) * S2) / (n1 + n2 - 2)
        S_inv = np.linalg.inv(S_pooled)

        T2 = (n1 * n2) / (n1 + n2) * diff @ S_inv @ diff
        F = T2 * (n1 + n2 - p - 1) / (p * (n1 + n2 - 2))
        df1, df2 = p, n1 + n2 - p - 1
    else:
        # Separate covariances (approximate)
        S1 = np.cov(X1, rowvar=False, ddof=1)
        S2 = np.cov(X2, rowvar=False, ddof=1)
        S_combined = S1 / n1 + S2 / n2
        S_inv = np.linalg.inv(S_combined)

        T2 = diff @ S_inv @ diff
        # Approximate df using Nel-Van der Merwe
        F = T2
        df1, df2 = p, min(n1, n2) - p

    p_value = 1 - f_dist.cdf(F, df1, df2)

    return {
        "T2": T2,
        "F": F,
        "df1": df1,
        "df2": df2,
        "p_value": p_value,
        "mean_diff": diff,
    }


def manova(X, groups):
    """
    One-way MANOVA (Multivariate Analysis of Variance).

    Test H0: μ1 = μ2 = ... = μk vs H1: at least one differs

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.
    groups : array-like of shape (n_samples,)
        Group labels.

    Returns
    -------
    result : dict
        Dictionary with Wilks' lambda, F-approximation, and p-value.

    Examples
    --------
    >>> X = np.vstack([np.random.randn(30, 3),
    ...                np.random.randn(30, 3) + 1])
    >>> groups = np.repeat([0, 1], 30)
    >>> result = manova(X, groups)
    """
    X = np.asarray(X)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    g = len(unique_groups)
    n, p = X.shape

    # Overall mean
    x_bar = np.mean(X, axis=0)

    # Between-groups sum of squares (H)
    H = np.zeros((p, p))
    for group in unique_groups:
        X_g = X[groups == group]
        n_g = len(X_g)
        x_g_bar = np.mean(X_g, axis=0)
        diff = x_g_bar - x_bar
        H += n_g * np.outer(diff, diff)

    # Within-groups sum of squares (E)
    E = np.zeros((p, p))
    for group in unique_groups:
        X_g = X[groups == group]
        x_g_bar = np.mean(X_g, axis=0)
        for row in X_g:
            diff = row - x_g_bar
            E += np.outer(diff, diff)

    # Wilks' lambda
    det_E = np.linalg.det(E)
    det_EH = np.linalg.det(E + H)
    wilks_lambda = det_E / det_EH

    # F-approximation (Rao's)
    df_h = g - 1  # Hypothesis df
    df_e = n - g  # Error df

    # Parameters for F approximation
    t = np.sqrt((p**2 * df_h**2 - 4) / (p**2 + df_h**2 - 5)) if p**2 + df_h**2 - 5 > 0 else 1
    df1 = p * df_h
    df2 = (df_e + df_h - (p + df_h + 1) / 2) * t - (p * df_h - 2) / 2

    lambda_power = wilks_lambda ** (1 / t)
    F = (1 - lambda_power) / lambda_power * df2 / df1
    p_value = 1 - f_dist.cdf(F, df1, df2)

    return {
        "wilks_lambda": wilks_lambda,
        "F": F,
        "df1": df1,
        "df2": df2,
        "p_value": p_value,
        "H": H,
        "E": E,
    }


def box_m_test(groups_data):
    """
    Box's M test for equality of covariance matrices.

    Test H0: Σ1 = Σ2 = ... = Σk

    Parameters
    ----------
    groups_data : list of arrays
        List of data matrices, one per group.

    Returns
    -------
    result : dict
        Dictionary with M statistic, chi-squared approximation, and p-value.

    Examples
    --------
    >>> g1 = np.random.randn(30, 3)
    >>> g2 = np.random.randn(30, 3)
    >>> result = box_m_test([g1, g2])
    """
    k = len(groups_data)
    n_list = [len(g) for g in groups_data]
    n_total = sum(n_list)
    p = groups_data[0].shape[1]

    # Pooled covariance
    S_list = [np.cov(g, rowvar=False, ddof=1) for g in groups_data]
    S_pooled = sum((n - 1) * S for n, S in zip(n_list, S_list, strict=True)) / (n_total - k)

    # Box's M
    M = (n_total - k) * np.log(np.linalg.det(S_pooled))
    for n, S in zip(n_list, S_list, strict=True):
        M -= (n - 1) * np.log(np.linalg.det(S))

    # Correction factor
    sum_inv = sum(1 / (n - 1) for n in n_list) - 1 / (n_total - k)
    c1 = (2 * p**2 + 3 * p - 1) / (6 * (p + 1) * (k - 1)) * sum_inv
    # c2 would be used for F-approximation (not implemented)
    # c2 = ((p - 1) * (p + 2) / (6 * (k - 1))) * (
    #     sum(1 / (n - 1) ** 2 for n in n_list) - 1 / (n_total - k) ** 2
    # )

    df = p * (p + 1) * (k - 1) / 2
    chi2_stat = M * (1 - c1)
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return {
        "M": M,
        "chi2": chi2_stat,
        "df": df,
        "p_value": p_value,
    }


def bartlett_sphericity(X):
    """
    Bartlett's test of sphericity.

    Test H0: Σ = σ²I (correlation matrix is identity)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Data matrix.

    Returns
    -------
    result : dict
        Dictionary with chi-squared statistic, df, and p-value.

    Notes
    -----
    Used to check if data is suitable for factor analysis.
    """
    X = np.asarray(X)
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)

    det_R = np.linalg.det(R)
    chi2_stat = -(n - 1 - (2 * p + 5) / 6) * np.log(det_R)
    df = p * (p - 1) / 2
    p_value = 1 - chi2.cdf(chi2_stat, df)

    return {
        "chi2": chi2_stat,
        "df": df,
        "p_value": p_value,
    }


def likelihood_ratio_test(L_full, L_reduced, df):
    """
    General likelihood ratio test.

    Test H0: reduced model vs H1: full model

    Parameters
    ----------
    L_full : float
        Log-likelihood of full model.
    L_reduced : float
        Log-likelihood of reduced model.
    df : int
        Degrees of freedom (difference in parameters).

    Returns
    -------
    result : dict
        Dictionary with test statistic and p-value.
    """
    lambda_stat = -2 * (L_reduced - L_full)
    p_value = 1 - chi2.cdf(lambda_stat, df)

    return {
        "lambda": lambda_stat,
        "df": df,
        "p_value": p_value,
    }


def correlation_test(r, n):
    """
    Test if correlation is significantly different from zero.

    Parameters
    ----------
    r : float
        Sample correlation coefficient.
    n : int
        Sample size.

    Returns
    -------
    result : dict
        Dictionary with t-statistic and p-value.
    """
    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
    df = n - 2
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    return {
        "t": t_stat,
        "df": df,
        "p_value": p_value,
    }


# ============================================================================
# NESTED MODEL TESTS
# ============================================================================


def nested_f_test(SSE_full, SSE_reduced, df_full, df_reduced):
    """
    F-test for comparing nested regression models.

    Tests whether the additional parameters in the full model
    significantly improve the fit compared to the reduced model.

    H0: Reduced model is adequate
    H1: Full model is better

    Parameters
    ----------
    SSE_full : float
        Sum of squared errors for the full model.
    SSE_reduced : float
        Sum of squared errors for the reduced model.
    df_full : int
        Degrees of freedom for error in full model (n - p_full).
    df_reduced : int
        Degrees of freedom for error in reduced model (n - p_reduced).

    Returns
    -------
    result : dict
        - F: F-statistic
        - df1, df2: Degrees of freedom
        - p_value: p-value
        - distribution: String description of distribution

    Examples
    --------
    >>> # Compare model with 6 params vs model with 3 params (n=29)
    >>> result = nested_f_test(SSE_full=124382, SSE_reduced=137360, df_full=23, df_reduced=26)
    >>> print(f"F = {result['F']:.3f}, p = {result['p_value']:.4f}")

    Notes
    -----
    F = [(SSE_reduced - SSE_full) / (df_reduced - df_full)] / [SSE_full / df_full]
    F ~ F(df_reduced - df_full, df_full)
    """
    # Numerator df = difference in parameters
    df1 = df_reduced - df_full  # Number of additional parameters in full model

    # Denominator df = error df in full model
    df2 = df_full

    # F-statistic
    F = ((SSE_reduced - SSE_full) / df1) / (SSE_full / df2)

    # p-value
    p_value = 1 - f_dist.cdf(F, df1, df2)

    return {
        "F": F,
        "df1": df1,
        "df2": df2,
        "p_value": p_value,
        "distribution": f"F({df1}, {df2})",
        "SSE_reduction": SSE_reduced - SSE_full,
        "MSE_full": SSE_full / df_full,
    }


def nested_f_test_from_r2(R2_full, R2_reduced, n, p_full, p_reduced):
    """
    F-test for nested models using R² values.

    Parameters
    ----------
    R2_full : float
        R² of the full model.
    R2_reduced : float
        R² of the reduced model.
    n : int
        Sample size.
    p_full : int
        Number of parameters in full model (including intercept).
    p_reduced : int
        Number of parameters in reduced model.

    Returns
    -------
    result : dict
        F-statistic, degrees of freedom, and p-value.

    Examples
    --------
    >>> result = nested_f_test_from_r2(R2_full=0.6151, R2_reduced=0.5750, n=29, p_full=6, p_reduced=3)
    """
    df1 = p_full - p_reduced
    df2 = n - p_full

    F = ((R2_full - R2_reduced) / df1) / ((1 - R2_full) / df2)
    p_value = 1 - f_dist.cdf(F, df1, df2)

    return {
        "F": F,
        "df1": df1,
        "df2": df2,
        "p_value": p_value,
        "R2_reduction": R2_full - R2_reduced,
        "distribution": f"F({df1}, {df2})",
    }


# ============================================================================
# MULTIVARIATE GENERAL LINEAR MODEL (MGLM) TESTS
# ============================================================================


def mglm_test(n, p, q, wilks_lambda=None, hypothesis_matrix_rank=None):
    """
    Test hypotheses in Multivariate General Linear Model.

    For testing H0: ABB' = C in the model Y = XB + E

    The test statistic follows a U-distribution: U(s, m, n)
    where typically:
    - s = min(q, r) where q = number of responses, r = rank of A
    - m = (|q - r| - 1) / 2
    - n = (error df - q - 1) / 2

    Parameters
    ----------
    n : int
        Number of observations.
    p : int
        Number of parameters (columns in X).
    q : int
        Number of response variables.
    wilks_lambda : float, optional
        Wilks' Lambda statistic if already computed.
    hypothesis_matrix_rank : int, optional
        Rank of the hypothesis matrix A. Default is 1.

    Returns
    -------
    result : dict
        - distribution: Description of distribution U(s, m, n)
        - s, m, n_param: Parameters of U-distribution
        - F_approx: F approximation if applicable
        - df1, df2: Degrees of freedom for F

    Examples
    --------
    >>> # Test if β = 0 in model with 5 obs, 3 params, 4 responses
    >>> result = mglm_test(n=5, p=3, q=4, hypothesis_matrix_rank=1)
    >>> print(f"Distribution: {result['distribution']}")
    """
    r = hypothesis_matrix_rank if hypothesis_matrix_rank is not None else 1

    # Parameters for U-distribution
    s = min(q, r)
    error_df = n - p
    m = (abs(q - r) - 1) / 2
    n_param = (error_df - q - 1) / 2

    result = {
        "distribution": f"U({s}, {m:.1f}, {n_param:.1f})",
        "s": s,
        "m": m,
        "n_param": n_param,
        "error_df": error_df,
        "hypothesis_df": r,
    }

    # F approximation for Wilks' Lambda
    if wilks_lambda is not None:
        # Rao's F approximation
        if s == 1:
            # Simple case: F = (1-λ)/λ × (n-p-q+1)/q
            F = (1 - wilks_lambda) / wilks_lambda * (error_df - q + 1) / q
            df1 = q
            df2 = error_df - q + 1
        elif s == 2:
            # Two-term case
            sqrt_lambda = np.sqrt(wilks_lambda)
            df1 = 2 * q
            df2 = 2 * (error_df - q) - 2
            F = (1 - sqrt_lambda) / sqrt_lambda * df2 / df1
        else:
            # General case (approximate)
            t = np.sqrt((q**2 * r**2 - 4) / (q**2 + r**2 - 5)) if q**2 + r**2 > 5 else 1
            df1 = q * r
            df2 = (error_df - (q - r + 1) / 2) * t - (q * r - 2) / 2
            F = (1 - wilks_lambda ** (1 / t)) / (wilks_lambda ** (1 / t)) * df2 / df1

        result["F"] = F
        result["df1"] = df1
        result["df2"] = df2
        result["p_value"] = 1 - f_dist.cdf(F, df1, df2)

    return result


def discriminant_subset_test(n1, n2, D2_full, D2_reduced, p_full, p_reduced):
    """
    Test if a subset of variables contributes to discrimination.

    Tests whether the additional variables in the full model
    significantly improve discrimination between two groups.

    Parameters
    ----------
    n1, n2 : int
        Sample sizes for the two groups.
    D2_full : float
        Generalized squared distance (Mahalanobis D²) for full model.
    D2_reduced : float
        Generalized squared distance for reduced model.
    p_full : int
        Number of variables in full model.
    p_reduced : int
        Number of variables in reduced model.

    Returns
    -------
    result : dict
        - F: F-statistic
        - df1, df2: Degrees of freedom
        - p_value: p-value
        - distribution: String description

    Examples
    --------
    >>> # Test if variables 4-8 add to discrimination beyond variables 1-3
    >>> result = discriminant_subset_test(n1=21, n2=12, D2_full=3.62, D2_reduced=1.52, p_full=8, p_reduced=3)

    Notes
    -----
    F = [(n1+n2-p_full-1)/(p_full-p_reduced)] × [D²_full - D²_reduced] / [(n1+n2)(n1+n2-2)/(n1×n2) + D²_reduced]
    """
    n = n1 + n2

    # Degrees of freedom
    df1 = p_full - p_reduced
    df2 = n - p_full - 1

    # F-statistic
    numerator = (D2_full - D2_reduced) / df1
    denominator = (n * (n - 2) / (n1 * n2) + D2_reduced) / df2

    F = numerator / denominator

    # p-value
    p_value = 1 - f_dist.cdf(F, df1, df2)

    return {
        "F": F,
        "df1": df1,
        "df2": df2,
        "p_value": p_value,
        "distribution": f"F({df1}, {df2})",
        "D2_improvement": D2_full - D2_reduced,
    }


def hotellings_from_D2(n1, n2, D2):
    """
    Calculate Hotelling's T² from generalized squared distance D².

    T² = (n1 × n2)/(n1 + n2) × D²

    Parameters
    ----------
    n1, n2 : int
        Sample sizes for the two groups.
    D2 : float
        Generalized squared distance (from SAS output).

    Returns
    -------
    result : dict
        - T2: Hotelling's T²
        - F: F-statistic
        - df1, df2: Degrees of freedom (approximation)
        - p_value: p-value

    Examples
    --------
    >>> result = hotellings_from_D2(n1=21, n2=12, D2=3.62001)
    >>> print(f"T² = {result['T2']:.4f}")
    """
    T2 = (n1 * n2) / (n1 + n2) * D2

    return {
        "T2": T2,
        "D2": D2,
        "n1": n1,
        "n2": n2,
    }
