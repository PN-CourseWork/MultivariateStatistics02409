"""
Problem 1: Multivariate Linear Regression
==========================================

8 questions on multivariate regression: parameter estimation,
covariance of estimates, leverage, and hypothesis testing.

**Answers:** 2, 1, 5, 3, 3, 2, 1, 4
"""

import numpy as np

# %%
# Data Setup
# ----------

# Design matrix X (5 observations, 3 parameters: intercept, x, x²)
X = np.array([[1, -2, -4], [1, -1, -1], [1, 0, 0], [1, 1, 1], [1, 2, 4]])

# Response matrix Y (columns: Y, Z, V, W)
Y = np.array([[1, 8, 2, 9], [0, 9, 4, 6], [2, 4, 4, 2], [1, 5, 9, 5], [1, 2, 8, 7]])

# Given: (X'X)^(-1)
XtX_inv = np.array([[0.2, 0, 0], [0, 2.125, -1.125], [0, -1.125, 0.625]])

n, p = X.shape
var_names = ["Y", "Z", "V", "W"]

print("Design matrix X:")
print(X)
print("\nResponse matrix Y (columns: Y, Z, V, W):")
print(Y)

# %%
# Q1.1: ML Estimates for [α_y, β_y, γ_y]
# --------------------------------------
# Formula: B_hat = (X'X)^(-1) X' Y

B_hat = XtX_inv @ X.T @ Y
alpha_y, beta_y, gamma_y = B_hat[:, 0]

print("Q1.1: ML estimates for Y")
print(f"[α_y, β_y, γ_y] = [{alpha_y}, {beta_y}, {gamma_y}]")
print("✓ Answer 2: [1, 1, -0.5]")

# %%
# Q1.2: Cov(α̂, β̂)
# ----------------
# Cov(α̂, β̂) = σ² × (X'X)^(-1)[0,1]

print("\nQ1.2: Cov(α̂, β̂)")
print(f"(X'X)^(-1)[0,1] = {XtX_inv[0, 1]}")
print("Since this is 0, Cov(α̂, β̂) = 0 regardless of σ²")
print("✓ Answer 1: 0")

# %%
# Q1.3: Estimated Cov(β̂, γ̂) for Y
# --------------------------------
# First estimate σ² from residuals

Y_hat = X @ B_hat
residuals = Y - Y_hat
SSE_y = np.sum(residuals[:, 0] ** 2)
MSE_y = SSE_y / (n - p)  # df = 5 - 3 = 2

est_cov_beta_gamma = MSE_y * XtX_inv[1, 2]

print("\nQ1.3: Estimated Cov(β̂, γ̂)")
print(f"MSE for Y: {MSE_y}")
print(f"Est. Cov(β̂, γ̂) = {MSE_y} × {XtX_inv[1, 2]} = {est_cov_beta_gamma}")
print("✓ Answer 5: -0.84375")

# %%
# Q1.4: Estimated Var(α̂) for Y
# ----------------------------

est_var_alpha = MSE_y * XtX_inv[0, 0]

print("\nQ1.4: Estimated Var(α̂)")
print(f"Est. Var(α̂) = {MSE_y} × {XtX_inv[0, 0]} = {est_var_alpha}")
print("✓ Answer 3: 0.15")

# %%
# Q1.5: Observation with Lowest Leverage
# --------------------------------------
# Leverage = diagonal of H = X(X'X)^(-1)X'

H = X @ XtX_inv @ X.T
leverage = np.diag(H)

print("\nQ1.5: Leverage values")
for i, h in enumerate(leverage, 1):
    print(f"  Obs {i}: {h:.4f}")
print(f"Lowest leverage: observation {np.argmin(leverage) + 1}")
print("✓ Answer 3: observation 3")

# %%
# Q1.6: Variable with Lowest MSE
# ------------------------------

MSE_all = np.sum(residuals**2, axis=0) / (n - p)

print("\nQ1.6: MSE per variable")
for name, mse in zip(var_names, MSE_all, strict=True):
    print(f"  {name}: {mse:.4f}")
print(f"Lowest MSE: {var_names[np.argmin(MSE_all)]}")
print("✓ Answer 2: Y")

# %%
# Q1.7 & Q1.8: Hypothesis Testing
# -------------------------------

print("\nQ1.7: Matrix A for testing β = 0")
print("A = [0, 1, 0] selects the β row")
print("✓ Answer 1: [0, 1, 0]")

print("\nQ1.8: Distribution of test statistic")
print("U(s, r, n-k) where s=4, r=1, n-k=2")
print("✓ Answer 4: U(4, 1, 2)")
