"""
Multistat - Tools for 02409 Multivariate Statistics Exam
========================================================

A Python toolkit for multivariate statistical analysis,
designed as a practical alternative to R for the DTU exam.

Modules
-------
descriptive : Covariance, correlation, and summary statistics
hypothesis : Statistical tests (t-test, F-test, MANOVA, etc.)
regression : Linear and logistic regression
multivariate : PCA, discriminant analysis, factor analysis
plotting : Visualization utilities
"""

from multistat.descriptive import (
    conditional_distribution,
    # Conditional distributions
    conditional_mean,
    conditional_variance,
    correlation_matrix,
    covariance_matrix,
    # Linear transformations
    linear_transform,
    linear_transform_cov,
    linear_transform_mean,
    mahalanobis_distance,
    multivariate_mean,
    pooled_covariance,
    standardize,
    summary_stats,
    # Helper functions
    variance_difference,
    variance_sum,
)
from multistat.hypothesis import (
    bartlett_sphericity,
    box_m_test,
    correlation_test,
    discriminant_subset_test,
    hotellings_from_D2,
    hotellings_t2,
    likelihood_ratio_test,
    manova,
    # MGLM tests
    mglm_test,
    # Nested model tests
    nested_f_test,
    nested_f_test_from_r2,
    two_sample_t2,
)
from multistat.multivariate import (
    # Selection helpers
    backward_selection_order,
    canonical_correlation,
    classification_summary,
    compare_classifiers,
    # Classification helpers
    confusion_matrix,
    factor_analysis,
    lda,
    pca,
    predict_lda,
    predict_qda,
    qda,
    screeplot_elbow,
)
from multistat.plotting import (
    plot_bivariate_normal,
    plot_correlation_heatmap,
    plot_lda_projection,
    plot_pca_variance,
    plot_scatter_matrix,
)
from multistat.regression import (
    anova_regression,
    covariance_of_estimates,
    leave_one_out_variance,
    # Leverage and diagnostics
    leverage,
    linear_regression,
    logistic_regression,
    multiple_regression,
    multivariate_regression,
    prediction_interval,
    regression_diagnostics,
    regression_summary,
)

__version__ = "0.2.0"
__all__ = [
    # Descriptive
    "covariance_matrix",
    "correlation_matrix",
    "multivariate_mean",
    "pooled_covariance",
    "summary_stats",
    "standardize",
    "mahalanobis_distance",
    # Linear transformations
    "linear_transform",
    "linear_transform_mean",
    "linear_transform_cov",
    # Conditional distributions
    "conditional_mean",
    "conditional_variance",
    "conditional_distribution",
    # Helper functions
    "variance_difference",
    "variance_sum",
    # Hypothesis testing
    "hotellings_t2",
    "two_sample_t2",
    "manova",
    "box_m_test",
    "bartlett_sphericity",
    "likelihood_ratio_test",
    "correlation_test",
    # Nested model tests
    "nested_f_test",
    "nested_f_test_from_r2",
    # MGLM tests
    "mglm_test",
    "discriminant_subset_test",
    "hotellings_from_D2",
    # Regression
    "linear_regression",
    "multiple_regression",
    "logistic_regression",
    "regression_summary",
    "anova_regression",
    # Leverage and diagnostics
    "leverage",
    "regression_diagnostics",
    "covariance_of_estimates",
    "prediction_interval",
    "leave_one_out_variance",
    "multivariate_regression",
    # Multivariate
    "pca",
    "lda",
    "qda",
    "factor_analysis",
    "canonical_correlation",
    "predict_lda",
    "predict_qda",
    # Classification helpers
    "confusion_matrix",
    "classification_summary",
    "compare_classifiers",
    # Selection helpers
    "backward_selection_order",
    "screeplot_elbow",
    # Plotting
    "plot_scatter_matrix",
    "plot_bivariate_normal",
    "plot_pca_variance",
    "plot_lda_projection",
    "plot_correlation_heatmap",
]
