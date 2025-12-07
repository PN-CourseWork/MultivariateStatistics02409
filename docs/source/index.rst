02409 Multivariate Statistics - Exam Tools
==========================================

Python toolkit for the DTU 02409 Multivariate Statistics exam.

**Author:** Philip Korsager Nickel

This documentation provides:

- **Exam Gallery**: Solved old exams with Python implementations
- **API Reference**: Complete reference for the multistat module
- **Quick Reference**: Common operations for the exam

Quick Start
-----------

.. code-block:: python

    import numpy as np
    from multistat import (
        covariance_matrix, correlation_matrix,
        linear_regression, logistic_regression,
        pca, lda, manova, hotellings_t2
    )

    # Load data
    X = np.random.randn(100, 3)
    y = np.random.randn(100)

    # Descriptive statistics
    S = covariance_matrix(X)
    R = correlation_matrix(X)

    # Regression
    result = linear_regression(X, y)
    print(result['summary'])

    # PCA
    pca_result = pca(X, n_components=2)
    print(f"Variance explained: {pca_result['cumulative_variance']}")


Contents
--------

:doc:`exam_gallery/2019/index`
   Solved old exams using Python (sphinx-gallery format).

:doc:`api_reference`
   Complete API reference for the multistat module.

:doc:`cheatsheet`
   Quick reference for common exam operations.

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Exam Solutions

   exam_gallery/2019/index

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Reference

   api_reference
   cheatsheet


R to Python Translation
-----------------------

Quick reference for translating R code to Python:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - R
     - Python (multistat)
   * - ``cov(X)``
     - ``covariance_matrix(X)``
   * - ``cor(X)``
     - ``correlation_matrix(X)``
   * - ``t.test(x, y)``
     - ``scipy.stats.ttest_ind(x, y)``
   * - ``lm(y ~ X)``
     - ``linear_regression(X, y)``
   * - ``glm(..., family=binomial)``
     - ``logistic_regression(X, y)``
   * - ``prcomp(X)``
     - ``pca(X)``
   * - ``lda(y ~ ., data)``
     - ``lda(X, y)``
   * - ``manova(...)``
     - ``manova(X, groups)``


Installation
------------

.. code-block:: bash

    cd MultivariateStatistics02409
    uv sync  # or: pip install -e .

