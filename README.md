# 02409 Multivariate Statistics - Python Exam Tools

Python toolkit for the DTU 02409 Multivariate Statistics exam.

## Quick Start

```bash
# Install dependencies
uv sync

# Run a solved exam problem
uv run python exams/2019/plot_p1_regression.py

# Build documentation
uv run sphinx-build -b html docs/source docs/build/html

# Open docs
open docs/build/html/index.html
```

## Project Structure

```
MultivariateStatistics02409/
├── src/multistat/              # Main Python module (45+ functions)
│   ├── descriptive.py          # Covariance, correlation, transforms
│   ├── hypothesis.py           # T², MANOVA, F-tests
│   ├── regression.py           # Regression, leverage, diagnostics
│   ├── multivariate.py         # PCA, LDA, QDA, Factor Analysis
│   └── plotting.py             # Visualization utilities
├── exams/
│   ├── 2019/                   # Solved exam (self-contained)
│   │   ├── paper/              # Exam problem PDF
│   │   ├── solution/           # Official detailed solution PDFs
│   │   ├── data/               # CSV data files
│   │   └── plot_p*.py          # Python solutions (sphinx-gallery)
│   └── unsolved/               # Other exam years (PDFs only)
│       ├── papers/             # Exam PDFs (2001-2024)
│       └── solutions/          # Official solutions (2014-2023)
├── docs/                       # Sphinx documentation
└── pyproject.toml              # Dependencies
```

## Module Overview

```python
from multistat import (
    # Descriptive & Transformations
    covariance_matrix, correlation_matrix, mahalanobis_distance,
    linear_transform, conditional_distribution,

    # Hypothesis tests
    hotellings_t2, two_sample_t2, manova, box_m_test,
    nested_f_test, mglm_test,

    # Regression & Diagnostics
    linear_regression, leverage, regression_diagnostics,
    covariance_of_estimates, prediction_interval,

    # Multivariate methods
    pca, lda, qda, factor_analysis, canonical_correlation,
    confusion_matrix, backward_selection_order,
)
```

## Verified Against Official Solutions (2019 Exam)

```
✓ Q1.1: [α_y, β_y, γ_y] = [1, 1, -0.5]
✓ Q1.2: Cov(α̂, β̂) = 0
✓ Q1.3: Est. Cov(β̂, γ̂) = -0.84375
✓ Q1.4: Est. Var(α̂) = 0.15
✓ Q1.5: Lowest leverage = obs 3
✓ Q1.6: Lowest MSE = Y
ALL 30/30 ANSWERS VERIFIED CORRECT!
```

## Key Functions by Exam Pattern

### Leverage (Very Common!)
```python
from multistat import leverage
result = leverage(X, add_intercept=False)
print(f"Lowest leverage: obs {result['lowest_leverage_obs']}")
```

### Covariance of Parameter Estimates
```python
from multistat import covariance_of_estimates
result = covariance_of_estimates(X, MSE=0.75)
print(f"Cov(β̂, γ̂) = {result['cov_matrix'][1, 2]}")
```

### Conditional Distribution (Theory Section)
```python
from multistat import conditional_distribution
result = conditional_distribution(mu, Sigma, idx_given=1, x_given=2)
print(f"E(X|Y=2) = {result['mean']}")
print(f"D(X|Y) = {result['cov']}")
```

### Linear Transformation
```python
from multistat import linear_transform
A = np.array([[1, -1, 0], [0, 1, -1]])  # S=X-Y, T=Y-Z
result = linear_transform(A, mu, Sigma)
print(f"E[S,T] = {result['mean']}")
print(f"D[S,T] = {result['cov']}")
```

### Nested F-test
```python
from multistat import nested_f_test
result = nested_f_test(SSE_full=124382, SSE_reduced=137360, df_full=23, df_reduced=26)
print(f"F = {result['F']:.4f}, p = {result['p_value']:.4f}")
```

### Full Regression Diagnostics
```python
from multistat import regression_diagnostics
diag = regression_diagnostics(X, y)
print(diag['summary'])  # DataFrame with leverage, RStudent, DFBETAS, etc.
```

## R to Python Quick Reference

| R | Python |
|---|--------|
| `cov(X)` | `covariance_matrix(X)` |
| `cor(X)` | `correlation_matrix(X)` |
| `lm(y ~ X)` | `linear_regression(X, y)` |
| `prcomp(X)` | `pca(X)` |
| `lda(y ~ .)` | `lda(X, y)` |
| `manova(...)` | `manova(X, groups)` |
| `solve(A)` | `np.linalg.inv(A)` |
| `t(X)` | `X.T` |
| `X %*% Y` | `X @ Y` |

## All Functions (v0.2.0)

**Descriptive**: `covariance_matrix`, `correlation_matrix`, `pooled_covariance`, `mahalanobis_distance`, `linear_transform`, `conditional_mean`, `conditional_variance`, `conditional_distribution`

**Hypothesis**: `hotellings_t2`, `two_sample_t2`, `manova`, `box_m_test`, `bartlett_sphericity`, `nested_f_test`, `mglm_test`, `discriminant_subset_test`

**Regression**: `linear_regression`, `logistic_regression`, `leverage`, `regression_diagnostics`, `covariance_of_estimates`, `prediction_interval`, `leave_one_out_variance`, `multivariate_regression`

**Multivariate**: `pca`, `lda`, `qda`, `factor_analysis`, `canonical_correlation`, `predict_lda`, `predict_qda`, `confusion_matrix`, `backward_selection_order`, `screeplot_elbow`

## Commands

```bash
uv sync                                                  # Install deps
uv run python exams/2019/plot_p1_regression.py           # Run exam problem
uv run sphinx-build -b html docs/source docs/build/html  # Build docs
uv run pytest                                            # Run tests
```
