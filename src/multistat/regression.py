"""
Regression Analysis
===================

Linear and logistic regression with comprehensive output
matching R-style summaries for exam purposes.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f as f_dist
from scipy.stats import t as t_dist


def linear_regression(X, y, add_intercept=True):
    """
    Simple or multiple linear regression using OLS.

    Parameters
    ----------
    X : array-like of shape (n_samples,) or (n_samples, n_features)
        Predictor variables.
    y : array-like of shape (n_samples,)
        Response variable.
    add_intercept : bool, default=True
        Add intercept term.

    Returns
    -------
    result : dict
        Comprehensive regression results including:
        - coefficients, se, t_values, p_values
        - R², adjusted R², F-statistic
        - residuals, fitted values

    Examples
    --------
    >>> X = np.random.randn(100)
    >>> y = 2 + 3*X + np.random.randn(100)*0.5
    >>> result = linear_regression(X, y)
    >>> print(result['summary'])
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape

    if add_intercept:
        X = np.column_stack([np.ones(n), X])
        p += 1
        var_names = ["Intercept"] + [f"X{i}" for i in range(1, p)]
    else:
        var_names = [f"X{i}" for i in range(1, p + 1)]

    # OLS estimation: β = (X'X)^(-1) X'y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta = XtX_inv @ X.T @ y

    # Fitted values and residuals
    y_hat = X @ beta
    residuals = y - y_hat

    # Sum of squares
    SS_res = np.sum(residuals**2)
    SS_tot = np.sum((y - np.mean(y)) ** 2)
    SS_reg = SS_tot - SS_res

    # Degrees of freedom
    df_reg = p - 1 if add_intercept else p
    df_res = n - p
    df_tot = n - 1

    # Mean squares
    MS_reg = SS_reg / df_reg
    MS_res = SS_res / df_res

    # R² and adjusted R²
    R2 = 1 - SS_res / SS_tot
    R2_adj = 1 - (SS_res / df_res) / (SS_tot / df_tot)

    # F-statistic
    F_stat = MS_reg / MS_res
    F_pvalue = 1 - f_dist.cdf(F_stat, df_reg, df_res)

    # Standard errors of coefficients
    sigma2 = MS_res
    se = np.sqrt(np.diag(XtX_inv) * sigma2)

    # t-statistics and p-values for coefficients
    t_values = beta / se
    p_values = 2 * (1 - t_dist.cdf(np.abs(t_values), df_res))

    # Create summary DataFrame
    summary = pd.DataFrame(
        {
            "Coefficient": beta,
            "Std.Error": se,
            "t-value": t_values,
            "p-value": p_values,
        },
        index=var_names,
    )

    # Significance stars
    def sig_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.1:
            return "."
        return ""

    summary["Sig"] = summary["p-value"].apply(sig_stars)

    return {
        "coefficients": beta,
        "se": se,
        "t_values": t_values,
        "p_values": p_values,
        "R2": R2,
        "R2_adj": R2_adj,
        "F_stat": F_stat,
        "F_pvalue": F_pvalue,
        "df_reg": df_reg,
        "df_res": df_res,
        "sigma": np.sqrt(sigma2),
        "residuals": residuals,
        "fitted": y_hat,
        "summary": summary,
        "SS_reg": SS_reg,
        "SS_res": SS_res,
        "SS_tot": SS_tot,
    }


def multiple_regression(X, y, var_names=None):
    """
    Multiple regression with named variables.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Predictor matrix.
    y : array-like of shape (n_samples,)
        Response variable.
    var_names : list of str, optional
        Names for predictor variables.

    Returns
    -------
    result : dict
        Same as linear_regression but with named variables.
    """
    result = linear_regression(X, y, add_intercept=True)

    if var_names is not None:
        new_names = ["Intercept"] + list(var_names)
        result["summary"].index = new_names

    return result


def logistic_regression(X, y, add_intercept=True, max_iter=100, tol=1e-6):
    """
    Logistic regression using iteratively reweighted least squares (IRLS).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Predictor variables.
    y : array-like of shape (n_samples,)
        Binary response (0 or 1).
    add_intercept : bool, default=True
        Add intercept term.
    max_iter : int, default=100
        Maximum iterations.
    tol : float, default=1e-6
        Convergence tolerance.

    Returns
    -------
    result : dict
        Logistic regression results including coefficients, standard errors,
        z-values, p-values, deviance, and AIC.

    Examples
    --------
    >>> X = np.random.randn(100)
    >>> p = 1 / (1 + np.exp(-(0.5 + 2*X)))
    >>> y = (np.random.rand(100) < p).astype(int)
    >>> result = logistic_regression(X, y)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape

    if add_intercept:
        X = np.column_stack([np.ones(n), X])
        p += 1
        var_names = ["Intercept"] + [f"X{i}" for i in range(1, p)]
    else:
        var_names = [f"X{i}" for i in range(1, p + 1)]

    def logistic(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    # Initialize
    beta = np.zeros(p)

    # IRLS algorithm
    for _ in range(max_iter):
        eta = X @ beta
        mu = logistic(eta)
        mu = np.clip(mu, 1e-10, 1 - 1e-10)

        # Weights
        W = mu * (1 - mu)

        # Working response
        z = eta + (y - mu) / W

        # Weighted least squares update
        XtWX = X.T @ (W[:, None] * X)
        XtWz = X.T @ (W * z)

        beta_new = np.linalg.solve(XtWX, XtWz)

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    # Final predictions
    eta = X @ beta
    mu = logistic(eta)

    # Covariance matrix of coefficients
    W = mu * (1 - mu)
    XtWX = X.T @ (W[:, None] * X)
    cov_beta = np.linalg.inv(XtWX)
    se = np.sqrt(np.diag(cov_beta))

    # z-statistics and p-values
    z_values = beta / se
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_values)))

    # Deviance
    eps = 1e-10
    deviance = -2 * np.sum(y * np.log(mu + eps) + (1 - y) * np.log(1 - mu + eps))

    # Null deviance (intercept only)
    p_null = np.mean(y)
    null_deviance = -2 * np.sum(y * np.log(p_null + eps) + (1 - y) * np.log(1 - p_null + eps))

    # AIC
    AIC = deviance + 2 * p

    # Pseudo R²
    pseudo_R2 = 1 - deviance / null_deviance

    # Summary
    summary = pd.DataFrame(
        {
            "Coefficient": beta,
            "Std.Error": se,
            "z-value": z_values,
            "p-value": p_values,
        },
        index=var_names,
    )

    def sig_stars(pval):
        if pval < 0.001:
            return "***"
        elif pval < 0.01:
            return "**"
        elif pval < 0.05:
            return "*"
        elif pval < 0.1:
            return "."
        return ""

    summary["Sig"] = summary["p-value"].apply(sig_stars)

    return {
        "coefficients": beta,
        "se": se,
        "z_values": z_values,
        "p_values": p_values,
        "deviance": deviance,
        "null_deviance": null_deviance,
        "AIC": AIC,
        "pseudo_R2": pseudo_R2,
        "fitted": mu,
        "summary": summary,
    }


def regression_summary(result):
    """
    Print a formatted regression summary similar to R.

    Parameters
    ----------
    result : dict
        Output from linear_regression or logistic_regression.
    """
    print("=" * 60)
    print("REGRESSION SUMMARY")
    print("=" * 60)
    print()
    print("Coefficients:")
    print(result["summary"].to_string())
    print()
    print("---")
    print("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")
    print()

    if "R2" in result:
        print(f"R-squared:          {result['R2']:.4f}")
        print(f"Adjusted R-squared: {result['R2_adj']:.4f}")
        print(
            f"F-statistic: {result['F_stat']:.2f} on {result['df_reg']} and {result['df_res']} DF"
        )
        print(f"p-value: {result['F_pvalue']:.4e}")
        print(f"Residual std error: {result['sigma']:.4f}")
    else:
        print(f"Null deviance:     {result['null_deviance']:.2f}")
        print(f"Residual deviance: {result['deviance']:.2f}")
        print(f"AIC: {result['AIC']:.2f}")
        print(f"Pseudo R²: {result['pseudo_R2']:.4f}")


def anova_regression(result):
    """
    ANOVA table for regression.

    Parameters
    ----------
    result : dict
        Output from linear_regression.

    Returns
    -------
    anova : DataFrame
        ANOVA table.
    """
    df_reg = result["df_reg"]
    df_res = result["df_res"]

    MS_reg = result["SS_reg"] / df_reg
    MS_res = result["SS_res"] / df_res

    anova = pd.DataFrame(
        {
            "Df": [df_reg, df_res],
            "Sum Sq": [result["SS_reg"], result["SS_res"]],
            "Mean Sq": [MS_reg, MS_res],
            "F value": [result["F_stat"], np.nan],
            "Pr(>F)": [result["F_pvalue"], np.nan],
        },
        index=["Regression", "Residual"],
    )

    return anova


# ============================================================================
# LEVERAGE AND DIAGNOSTIC FUNCTIONS
# ============================================================================


def leverage(X, add_intercept=True):
    """
    Calculate leverage values (hat matrix diagonal) for observations.

    The leverage h_ii measures how far observation i is from the center
    of the predictor space. High leverage observations can be influential.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix (predictors).
    add_intercept : bool, default=True
        Add intercept column to X.

    Returns
    -------
    result : dict
        - leverage: Array of leverage values h_ii
        - hat_matrix: Full hat matrix H = X(X'X)^(-1)X'
        - trace: Sum of leverage values (equals p)
        - mean_leverage: Average leverage (p/n)
        - high_leverage_threshold: 2p/n threshold
        - high_leverage_obs: Observations exceeding threshold (1-indexed)

    Examples
    --------
    >>> X = np.array([[1, -2], [1, -1], [1, 0], [1, 1], [1, 2]])
    >>> result = leverage(X, add_intercept=False)
    >>> print(f"Lowest leverage: obs {np.argmin(result['leverage']) + 1}")
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape

    if add_intercept:
        X = np.column_stack([np.ones(n), X])
        p += 1

    # Hat matrix: H = X(X'X)^(-1)X'
    XtX_inv = np.linalg.inv(X.T @ X)
    H = X @ XtX_inv @ X.T
    h = np.diag(H)

    # Threshold for high leverage: 2p/n
    threshold = 2 * p / n
    high_leverage = np.where(h > threshold)[0] + 1  # 1-indexed

    return {
        "leverage": h,
        "hat_matrix": H,
        "XtX_inv": XtX_inv,
        "trace": np.sum(h),
        "mean_leverage": p / n,
        "high_leverage_threshold": threshold,
        "high_leverage_obs": high_leverage,
        "lowest_leverage_obs": np.argmin(h) + 1,
        "highest_leverage_obs": np.argmax(h) + 1,
    }


def regression_diagnostics(X, y, add_intercept=True):
    """
    Comprehensive regression diagnostics including leverage, residuals,
    DFBETAS, DFFITS, and Cook's distance.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Predictor matrix.
    y : array-like of shape (n_samples,)
        Response variable.
    add_intercept : bool, default=True
        Add intercept column.

    Returns
    -------
    result : dict
        Comprehensive diagnostics including:
        - leverage: Hat diagonal values
        - residuals: Raw residuals
        - standardized_residuals: Residuals / (σ√(1-h))
        - studentized_residuals: Leave-one-out studentized (RStudent)
        - cooks_distance: Cook's D
        - dffits: DFFITS values
        - dfbetas: DFBETAS matrix (impact on each coefficient)
        - sigma_deleted: σ̂_(i) for each observation
        - influential_obs: Dictionary of influential observations

    Examples
    --------
    >>> X = np.random.randn(30, 3)
    >>> y = X @ [1, 2, 3] + np.random.randn(30) * 0.5
    >>> diag = regression_diagnostics(X, y)
    >>> print(f"Highest leverage: obs {diag['highest_leverage_obs']}")
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p_orig = X.shape

    if add_intercept:
        X = np.column_stack([np.ones(n), X])

    n, p = X.shape

    # Basic regression
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    residuals = y - y_hat

    # MSE and sigma
    SSE = np.sum(residuals**2)
    MSE = SSE / (n - p)
    sigma = np.sqrt(MSE)

    # Leverage
    H = X @ XtX_inv @ X.T
    h = np.diag(H)

    # Standardized residuals
    std_residuals = residuals / (sigma * np.sqrt(1 - h))

    # Studentized residuals (RStudent) - leave-one-out
    # σ̂_(i)² = [(n-p)σ̂² - e_i²/(1-h_i)] / (n-p-1)
    sigma_deleted_sq = ((n - p) * MSE - residuals**2 / (1 - h)) / (n - p - 1)
    sigma_deleted = np.sqrt(np.maximum(sigma_deleted_sq, 1e-10))
    rstudent = residuals / (sigma_deleted * np.sqrt(1 - h))

    # Cook's distance
    cooks_d = (std_residuals**2 / p) * (h / (1 - h))

    # DFFITS
    dffits = rstudent * np.sqrt(h / (1 - h))

    # DFBETAS - impact on each coefficient
    # DFBETAS_j(i) = (β̂_j - β̂_j(i)) / (σ̂_(i) * √((X'X)^(-1))_jj)
    dfbetas = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            dfbetas[i, j] = (residuals[i] * XtX_inv[j, :] @ X[i, :]) / (
                sigma_deleted[i] * np.sqrt(XtX_inv[j, j]) * (1 - h[i])
            )

    # Thresholds for influential observations
    leverage_threshold = 2 * p / n
    dfbetas_threshold = 2 / np.sqrt(n)
    dffits_threshold = 2 * np.sqrt(p / n)
    cooks_threshold = 4 / n

    # Find influential observations
    influential = {
        "high_leverage": np.where(h > leverage_threshold)[0] + 1,
        "high_cooks": np.where(cooks_d > cooks_threshold)[0] + 1,
        "high_dffits": np.where(np.abs(dffits) > dffits_threshold)[0] + 1,
        "high_rstudent": np.where(np.abs(rstudent) > 2)[0] + 1,
    }

    # Create summary DataFrame
    diag_df = pd.DataFrame(
        {
            "Obs": np.arange(1, n + 1),
            "Leverage": h,
            "Residual": residuals,
            "Std_Resid": std_residuals,
            "RStudent": rstudent,
            "Cooks_D": cooks_d,
            "DFFITS": dffits,
        }
    )

    # Add DFBETAS columns
    for j in range(p):
        col_name = f"DFBETAS_{j}" if j > 0 else "DFBETAS_Intercept"
        diag_df[col_name] = dfbetas[:, j]

    return {
        "leverage": h,
        "residuals": residuals,
        "standardized_residuals": std_residuals,
        "studentized_residuals": rstudent,
        "cooks_distance": cooks_d,
        "dffits": dffits,
        "dfbetas": dfbetas,
        "sigma": sigma,
        "sigma_deleted": sigma_deleted,
        "MSE": MSE,
        "XtX_inv": XtX_inv,
        "coefficients": beta,
        "fitted": y_hat,
        "summary": diag_df,
        "influential_obs": influential,
        "lowest_leverage_obs": np.argmin(h) + 1,
        "highest_leverage_obs": np.argmax(h) + 1,
        "highest_cooks_obs": np.argmax(cooks_d) + 1,
        "highest_dfbetas_intercept_obs": np.argmax(np.abs(dfbetas[:, 0])) + 1,
        "thresholds": {
            "leverage": leverage_threshold,
            "dfbetas": dfbetas_threshold,
            "dffits": dffits_threshold,
            "cooks": cooks_threshold,
        },
    }


def covariance_of_estimates(X, MSE=None, sigma_sq=None, add_intercept=True):
    """
    Calculate the covariance matrix of regression parameter estimates.

    Cov(β̂) = σ² × (X'X)^(-1)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Design matrix.
    MSE : float, optional
        Mean squared error (estimate of σ²). Provide either MSE or sigma_sq.
    sigma_sq : float, optional
        Known or estimated error variance σ².
    add_intercept : bool, default=True
        Add intercept column.

    Returns
    -------
    result : dict
        - cov_matrix: Full covariance matrix of β̂
        - var_estimates: Diagonal (variances of each β̂)
        - se_estimates: Standard errors of each β̂
        - XtX_inv: (X'X)^(-1) matrix
        - Individual covariances accessible via cov_matrix[i,j]

    Examples
    --------
    >>> X = np.array([[1, -2, 4], [1, -1, 1], [1, 0, 0], [1, 1, 1], [1, 2, 4]])
    >>> result = covariance_of_estimates(X, MSE=0.15, add_intercept=False)
    >>> print(f"Var(α̂) = {result['var_estimates'][0]}")
    >>> print(f"Cov(β̂, γ̂) = {result['cov_matrix'][1, 2]}")
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])

    XtX_inv = np.linalg.inv(X.T @ X)

    # Use provided variance estimate
    if sigma_sq is not None:
        var = sigma_sq
    elif MSE is not None:
        var = MSE
    else:
        raise ValueError("Must provide either MSE or sigma_sq")

    cov_matrix = var * XtX_inv
    var_estimates = np.diag(cov_matrix)
    se_estimates = np.sqrt(var_estimates)

    return {
        "cov_matrix": cov_matrix,
        "var_estimates": var_estimates,
        "se_estimates": se_estimates,
        "XtX_inv": XtX_inv,
        "sigma_sq": var,
    }


def prediction_interval(X_new, X, y, confidence=0.95, add_intercept=True):
    """
    Calculate confidence and prediction intervals for new observations.

    Parameters
    ----------
    X_new : array-like
        New predictor values (single observation or multiple).
    X : array-like
        Original design matrix.
    y : array-like
        Original response values.
    confidence : float, default=0.95
        Confidence level (e.g., 0.95 for 95%).
    add_intercept : bool, default=True
        Add intercept column.

    Returns
    -------
    result : dict
        - predicted: Predicted values ŷ
        - se_mean: Standard error for mean response
        - se_pred: Standard error for prediction
        - ci_lower, ci_upper: Confidence interval for E[Y|X]
        - pi_lower, pi_upper: Prediction interval for new Y

    Examples
    --------
    >>> X = np.random.randn(30, 2)
    >>> y = X @ [1, 2] + np.random.randn(30) * 0.5
    >>> result = prediction_interval([0.5, 1.0], X, y)
    >>> print(f"95% CI: [{result['ci_lower']:.2f}, {result['ci_upper']:.2f}]")
    """
    X = np.asarray(X)
    y = np.asarray(y)
    X_new = np.asarray(X_new)

    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    n = len(y)

    if add_intercept:
        X = np.column_stack([np.ones(n), X])
        X_new = np.column_stack([np.ones(len(X_new)), X_new])

    p = X.shape[1]

    # Fit model
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    residuals = y - y_hat
    MSE = np.sum(residuals**2) / (n - p)
    RMSE = np.sqrt(MSE)

    # Predictions
    y_pred = X_new @ beta

    # Leverage for new points
    h_new = np.array([x @ XtX_inv @ x for x in X_new])

    # Standard errors
    se_mean = RMSE * np.sqrt(h_new)  # For confidence interval
    se_pred = RMSE * np.sqrt(1 + h_new)  # For prediction interval

    # t critical value
    alpha = 1 - confidence
    t_crit = t_dist.ppf(1 - alpha / 2, n - p)

    # Intervals
    ci_lower = y_pred - t_crit * se_mean
    ci_upper = y_pred + t_crit * se_mean
    pi_lower = y_pred - t_crit * se_pred
    pi_upper = y_pred + t_crit * se_pred

    return {
        "predicted": y_pred.flatten() if len(y_pred) > 1 else y_pred[0],
        "se_mean": se_mean.flatten() if len(se_mean) > 1 else se_mean[0],
        "se_pred": se_pred.flatten() if len(se_pred) > 1 else se_pred[0],
        "ci_lower": ci_lower.flatten() if len(ci_lower) > 1 else ci_lower[0],
        "ci_upper": ci_upper.flatten() if len(ci_upper) > 1 else ci_upper[0],
        "pi_lower": pi_lower.flatten() if len(pi_lower) > 1 else pi_lower[0],
        "pi_upper": pi_upper.flatten() if len(pi_upper) > 1 else pi_upper[0],
        "leverage_new": h_new.flatten() if len(h_new) > 1 else h_new[0],
        "t_critical": t_crit,
        "df": n - p,
        "MSE": MSE,
        "RMSE": RMSE,
        "confidence": confidence,
    }


def leave_one_out_variance(residual, rstudent, leverage):
    """
    Calculate the variance estimate when an observation is deleted.

    This is commonly asked: "What is the variance if observation i is deleted?"

    Parameters
    ----------
    residual : float
        Raw residual e_i for observation i.
    rstudent : float
        Studentized residual (RStudent) for observation i.
    leverage : float
        Leverage h_ii for observation i.

    Returns
    -------
    result : dict
        - sigma_deleted: σ̂_(i)
        - variance_deleted: σ̂_(i)²

    Formula
    -------
    σ̂_(i) = e_i / (RStudent_i × √(1 - h_ii))

    Examples
    --------
    >>> result = leave_one_out_variance(residual=-2.0847, rstudent=-0.1112, leverage=0.936)
    >>> print(f"Variance if deleted: {result['variance_deleted']:.2f}")
    """
    sigma_deleted = residual / (rstudent * np.sqrt(1 - leverage))
    variance_deleted = sigma_deleted**2

    return {
        "sigma_deleted": sigma_deleted,
        "variance_deleted": variance_deleted,
    }


def multivariate_regression(X, Y, add_intercept=True):
    """
    Multivariate multiple regression (multiple response variables).

    Y = XB + E where Y has multiple columns.

    Parameters
    ----------
    X : array-like of shape (n, p)
        Design matrix.
    Y : array-like of shape (n, q)
        Response matrix with q response variables.
    add_intercept : bool, default=True
        Add intercept column.

    Returns
    -------
    result : dict
        - B_hat: Parameter matrix (p × q)
        - fitted: Predicted values
        - residuals: Residual matrix
        - MSE: MSE for each response
        - XtX_inv: (X'X)^(-1)
        - cov_params: Covariance structure for parameters

    Examples
    --------
    >>> X = np.array([[1, -2, 4], [1, -1, 1], [1, 0, 0], [1, 1, 1], [1, 2, 4]])
    >>> Y = np.array([[1, 8], [0, 9], [2, 4], [1, 5], [1, 2]])
    >>> result = multivariate_regression(X, Y, add_intercept=False)
    >>> print(f"B_hat[:, 0] = {result['B_hat'][:, 0]}")  # Params for first response
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n = len(Y)

    if add_intercept:
        X = np.column_stack([np.ones(n), X])

    n, p = X.shape
    q = Y.shape[1]

    # ML estimates: B_hat = (X'X)^(-1) X' Y
    XtX_inv = np.linalg.inv(X.T @ X)
    B_hat = XtX_inv @ X.T @ Y

    # Fitted values and residuals
    Y_hat = X @ B_hat
    residuals = Y - Y_hat

    # MSE for each response
    SSE = np.sum(residuals**2, axis=0)
    MSE = SSE / (n - p)

    # Residual covariance matrix
    Sigma_hat = (residuals.T @ residuals) / (n - p)

    return {
        "B_hat": B_hat,
        "fitted": Y_hat,
        "residuals": residuals,
        "SSE": SSE,
        "MSE": MSE,
        "XtX_inv": XtX_inv,
        "Sigma_hat": Sigma_hat,
        "n": n,
        "p": p,
        "q": q,
    }
