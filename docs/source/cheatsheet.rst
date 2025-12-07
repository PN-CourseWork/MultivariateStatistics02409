Exam Cheatsheet
===============

Quick reference for common exam operations.

Descriptive Statistics
----------------------

.. code-block:: python

    import numpy as np
    from multistat import covariance_matrix, correlation_matrix, summary_stats

    # Sample covariance matrix (n-1 denominator)
    S = covariance_matrix(X)

    # Correlation matrix
    R = correlation_matrix(X)

    # Summary statistics
    stats = summary_stats(X, var_names=['Weight', 'Height', 'Age'])

    # Pooled covariance for two groups
    from multistat import pooled_covariance
    S_pooled = pooled_covariance(X1, X2)


Linear Transformations (VERY COMMON!)
-------------------------------------

Given :math:`X \sim N(\mu, \Sigma)` and :math:`Y = AX + b`:

.. code-block:: python

    from multistat import linear_transform

    # Example: S = X - Y, T = Y - Z for [X, Y, Z]'
    A = np.array([[1, -1, 0],    # S = X - Y
                  [0, 1, -1]])   # T = Y - Z

    mu = np.array([1, 2, 3])
    Sigma = np.array([[1, 0.5, 0.25],
                      [0.5, 1, 0.5],
                      [0.25, 0.5, 1]])

    result = linear_transform(A, mu, Sigma)
    print(f"E[S, T] = {result['mean']}")      # Aμ
    print(f"D[S, T] =\n{result['cov']}")      # AΣA'

**Key formulas:**

.. math::

    E[AX + b] = A\mu + b

.. math::

    D[AX + b] = A \Sigma A^T

**Common cases:**

.. code-block:: python

    # Var(X - Y) = Var(X) + Var(Y) - 2Cov(X,Y)
    from multistat import variance_difference
    var_diff = variance_difference(Sigma, i=0, j=1)

    # Var(X + Y) = Var(X) + Var(Y) + 2Cov(X,Y)
    from multistat import variance_sum
    var_sum = variance_sum(Sigma, i=0, j=1)


Conditional Distributions (VERY COMMON!)
----------------------------------------

Given :math:`X = [X_1, X_2]' \sim N(\mu, \Sigma)`, find :math:`X_1 | X_2 = x_2`:

.. code-block:: python

    from multistat import conditional_distribution

    mu = np.array([1, 2, 3])
    Sigma = np.array([[1, 0.5, 0.25],
                      [0.5, 1, 0.5],
                      [0.25, 0.5, 1]])

    # E(X | Y=2) and D(X | Y) where Y is variable at index 1
    result = conditional_distribution(mu, Sigma, idx_given=1, x_given=2)
    print(f"E(X|Y=2) = {result['mean']}")
    print(f"D(X|Y) =\n{result['cov']}")

**For single variable conditioning:**

.. code-block:: python

    from multistat import conditional_mean, conditional_variance

    # E(X | Y = y)
    cond_mean = conditional_mean(mu, Sigma, idx_given=1, x_given=2)

    # D(X | Y) - doesn't depend on the value of Y!
    cond_var = conditional_variance(Sigma, idx_given=1)

**Key formulas:**

.. math::

    E(X_1 | X_2 = x_2) = \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (x_2 - \mu_2)

.. math::

    D(X_1 | X_2) = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}


Hypothesis Tests
----------------

T-tests
^^^^^^^

.. code-block:: python

    from scipy import stats

    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(x, popmean=0)

    # Two-sample t-test (equal variance)
    t_stat, p_value = stats.ttest_ind(x1, x2, equal_var=True)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(x1, x2)


Hotelling's T²
^^^^^^^^^^^^^^

.. code-block:: python

    from multistat import hotellings_t2, two_sample_t2

    # One-sample
    result = hotellings_t2(X, mu0=[0, 0, 0])
    print(f"T² = {result['T2']:.3f}, p = {result['p_value']:.4f}")

    # Two-sample
    result = two_sample_t2(X1, X2)

    # From generalized squared distance D² (discriminant analysis output)
    from multistat import hotellings_from_D2
    T2 = hotellings_from_D2(D2=3.62, n1=21, n2=12)  # T² = (n1*n2)/(n1+n2) * D²


MANOVA
^^^^^^

.. code-block:: python

    from multistat import manova

    result = manova(X, groups)
    print(f"Wilks' Lambda = {result['wilks_lambda']:.4f}")
    print(f"F = {result['F']:.2f}, p = {result['p_value']:.4f}")


Box's M Test
^^^^^^^^^^^^

.. code-block:: python

    from multistat import box_m_test

    # Test equality of covariance matrices
    result = box_m_test([X1, X2, X3])
    print(f"Chi² = {result['chi2']:.2f}, p = {result['p_value']:.4f}")


Nested F-test (Model Comparison)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from multistat import nested_f_test

    # Compare full model vs reduced model
    result = nested_f_test(
        SSE_full=124382,     # SSE from full model
        SSE_reduced=137360,  # SSE from reduced model
        df_full=23,          # df for full model (n - p_full)
        df_reduced=26        # df for reduced model (n - p_reduced)
    )
    print(f"F = {result['F']:.4f}")
    print(f"p-value = {result['p_value']:.4f}")

**Formula:**

.. math::

    F = \frac{(SSE_{reduced} - SSE_{full}) / (df_{reduced} - df_{full})}{SSE_{full} / df_{full}}


Regression
----------

Linear Regression
^^^^^^^^^^^^^^^^^

.. code-block:: python

    from multistat import linear_regression, regression_summary

    result = linear_regression(X, y)

    # Print summary (R-style)
    regression_summary(result)

    # Access specific values
    print(f"R² = {result['R2']:.4f}")
    print(f"Coefficients: {result['coefficients']}")
    print(f"MSE = {result['MSE']:.4f}")


Leverage (VERY COMMON!)
^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from multistat import leverage

    # X should include intercept column if model has intercept
    X = np.column_stack([np.ones(n), x1, x2])  # Design matrix with intercept

    result = leverage(X, add_intercept=False)  # Already has intercept

    print(f"Leverage values: {result['leverage']}")
    print(f"Lowest leverage observation: {result['lowest_leverage_obs']}")
    print(f"Highest leverage observation: {result['highest_leverage_obs']}")

**Manual calculation:**

.. code-block:: python

    # Hat matrix H = X(X'X)^(-1)X'
    XtX_inv = np.linalg.inv(X.T @ X)
    H = X @ XtX_inv @ X.T
    leverage = np.diag(H)

    # Find observation with lowest leverage
    lowest_obs = np.argmin(leverage) + 1  # +1 for 1-indexed

**Formula:**

.. math::

    h_{ii} = x_i^T (X^T X)^{-1} x_i


Covariance of Parameter Estimates (VERY COMMON!)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from multistat import covariance_of_estimates

    # Get Cov(β̂) = σ²(X'X)^(-1)
    result = covariance_of_estimates(X, MSE=0.75, add_intercept=False)

    print(f"Cov(β̂) =\n{result['cov_matrix']}")

    # Specific covariance, e.g., Cov(β̂, γ̂) at indices [1,2]
    print(f"Cov(β̂, γ̂) = {result['cov_matrix'][1, 2]}")

    # Variance of specific parameter
    print(f"Var(α̂) = {result['cov_matrix'][0, 0]}")

**Manual calculation:**

.. code-block:: python

    XtX_inv = np.linalg.inv(X.T @ X)
    cov_beta = MSE * XtX_inv

    # Cov(β̂, γ̂) where β is param 1 and γ is param 2
    cov_beta_gamma = MSE * XtX_inv[1, 2]

**Formula:**

.. math::

    Cov(\hat{\beta}) = \sigma^2 (X^T X)^{-1}

**Estimated version** (replace σ² with MSE):

.. math::

    \widehat{Cov}(\hat{\beta}) = MSE \cdot (X^T X)^{-1}


Regression Diagnostics
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from multistat import regression_diagnostics

    result = regression_diagnostics(X, y)

    # Full summary table
    print(result['summary'])  # DataFrame with all diagnostics

    # Individual measures
    print(f"Leverage: {result['leverage']}")
    print(f"RStudent: {result['rstudent']}")
    print(f"DFFITS: {result['dffits']}")
    print(f"Cook's D: {result['cooks_d']}")


Logistic Regression
^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from multistat import logistic_regression

    result = logistic_regression(X, y)

    # Coefficients and their significance
    print(result['summary'])

    # Predicted probabilities
    p_hat = result['fitted']


Using statsmodels (Alternative)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    import statsmodels.api as sm

    # Linear regression
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    print(model.summary())

    # Logistic regression
    model = sm.Logit(y, X_with_const).fit()
    print(model.summary())


Principal Component Analysis
----------------------------

.. code-block:: python

    from multistat import pca, screeplot_elbow

    result = pca(X, n_components=3, standardize=True)

    # Eigenvalues (variance explained)
    print(f"Eigenvalues: {result['eigenvalues']}")
    print(f"Variance ratio: {result['variance_ratio']}")
    print(f"Cumulative: {result['cumulative_variance']}")

    # Loadings
    print(result['loadings_df'])

    # Scores (transformed data)
    scores = result['scores']

    # Determine number of components (screeplot elbow)
    elbow = screeplot_elbow(result['eigenvalues'])
    print(f"Suggested components: {elbow['n_components']}")


Discriminant Analysis
---------------------

LDA (Linear Discriminant Analysis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from multistat import lda, predict_lda

    result = lda(X, y)

    # Discriminant coefficients
    print(f"Coefficients:\n{result['coefficients']}")

    # Eigenvalues (importance)
    print(f"Eigenvalues: {result['eigenvalues']}")

    # Classify new observations
    predictions = predict_lda(result, X_new)


QDA (Quadratic Discriminant Analysis)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use when covariance matrices differ between groups (Box's M test significant).

.. code-block:: python

    from multistat import qda, predict_qda, box_m_test

    # First check if QDA is needed
    box_result = box_m_test([X1, X2])
    if box_result['p_value'] < 0.05:
        print("Covariances differ - use QDA")

    # Fit QDA
    result = qda(X, y)

    # Classify
    predictions = predict_qda(result, X_new)


Comparing LDA vs QDA
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from multistat import confusion_matrix, compare_classifiers

    # Get predictions from both
    lda_pred = predict_lda(lda_result, X)
    qda_pred = predict_qda(qda_result, X)

    # Confusion matrices
    lda_cm = confusion_matrix(y, lda_pred)
    qda_cm = confusion_matrix(y, qda_pred)

    print(f"LDA misclassification: {lda_cm['misclassification_rate']:.2%}")
    print(f"QDA misclassification: {qda_cm['misclassification_rate']:.2%}")

    # Reduction in misclassifications
    reduction = lda_cm['n_misclassified'] - qda_cm['n_misclassified']
    print(f"QDA reduces misclassifications by: {reduction}")


Testing Variable Subsets in Discriminant Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    from multistat import discriminant_subset_test

    # Test if variables can be removed from discriminant function
    result = discriminant_subset_test(
        D2_full=3.62,      # D² with all variables
        D2_reduced=1.52,   # D² with subset
        p_full=8,          # Number of variables in full model
        p_reduced=3,       # Number of variables in reduced model
        n1=21, n2=12       # Sample sizes
    )
    print(f"F = {result['F']:.4f}, p = {result['p_value']:.4f}")


Factor Analysis
---------------

.. code-block:: python

    from multistat import factor_analysis, bartlett_sphericity, screeplot_elbow

    # Check if factor analysis is appropriate
    result = bartlett_sphericity(X)
    print(f"Bartlett's test: chi² = {result['chi2']:.2f}, p = {result['p_value']:.4f}")

    # Determine number of factors
    eigenvalues = np.linalg.eigvalsh(correlation_matrix(X))[::-1]
    elbow = screeplot_elbow(eigenvalues)
    print(f"Kaiser criterion (eigenvalue > 1): {sum(eigenvalues > 1)} factors")

    # Factor analysis
    result = factor_analysis(X, n_factors=2, rotation='varimax')

    # Loadings
    print(result['loadings_df'])

    # Communalities (variance explained by factors) and uniquenesses
    print(f"Communalities: {result['communalities']}")
    print(f"Uniquenesses: {result['uniquenesses']}")

    # Variable with lowest communality = worst explained by factors
    worst_var = np.argmin(result['communalities'])


Canonical Correlation Analysis
------------------------------

.. code-block:: python

    from multistat import canonical_correlation

    # X and Y are two sets of variables
    result = canonical_correlation(X, Y)

    print(f"Canonical correlations: {result['correlations']}")
    print(f"X coefficients:\n{result['x_coefficients']}")
    print(f"Y coefficients:\n{result['y_coefficients']}")

    # Test significance
    print(f"Wilks' Lambda = {result['wilks_lambda']:.4f}")
    print(f"p-value = {result['p_value']:.4f}")


Correlation Tests
-----------------

.. code-block:: python

    from scipy import stats

    # Test if correlation differs from zero
    r, p = stats.pearsonr(x, y)
    print(f"r = {r:.3f}, p = {p:.4f}")


Key Formulas
------------

Covariance Matrix
^^^^^^^^^^^^^^^^^

.. math::

    S = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(x_i - \bar{x})^T

Correlation from Covariance
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

    r_{ij} = \frac{s_{ij}}{\sqrt{s_{ii} \cdot s_{jj}}}

Hotelling's T²
^^^^^^^^^^^^^^

.. math::

    T^2 = n(\bar{x} - \mu_0)^T S^{-1} (\bar{x} - \mu_0)

F-approximation:

.. math::

    F = \frac{n-p}{p(n-1)} T^2 \sim F_{p, n-p}

Two-sample from D²:

.. math::

    T^2 = \frac{n_1 n_2}{n_1 + n_2} D^2

Wilks' Lambda
^^^^^^^^^^^^^

.. math::

    \Lambda = \frac{|E|}{|E + H|}

where E is within-groups SS matrix and H is between-groups SS matrix.

Leverage (Hat Matrix)
^^^^^^^^^^^^^^^^^^^^^

.. math::

    H = X(X^T X)^{-1} X^T

.. math::

    h_{ii} = x_i^T (X^T X)^{-1} x_i

Covariance of Estimates
^^^^^^^^^^^^^^^^^^^^^^^

.. math::

    Cov(\hat{\beta}) = \sigma^2 (X^T X)^{-1}

Estimated:

.. math::

    \widehat{Cov}(\hat{\beta}) = MSE \cdot (X^T X)^{-1}

Conditional Distribution
^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

    E(X_1 | X_2) = \mu_1 + \Sigma_{12} \Sigma_{22}^{-1} (X_2 - \mu_2)

.. math::

    D(X_1 | X_2) = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1} \Sigma_{21}

Linear Transformation
^^^^^^^^^^^^^^^^^^^^^

.. math::

    E[AX + b] = A\mu + b

.. math::

    D[AX + b] = A \Sigma A^T

Variance of Differences
^^^^^^^^^^^^^^^^^^^^^^^

.. math::

    Var(X - Y) = Var(X) + Var(Y) - 2Cov(X, Y)

.. math::

    Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)

Nested F-test
^^^^^^^^^^^^^

.. math::

    F = \frac{(SSE_{red} - SSE_{full}) / (df_{red} - df_{full})}{SSE_{full} / df_{full}}
