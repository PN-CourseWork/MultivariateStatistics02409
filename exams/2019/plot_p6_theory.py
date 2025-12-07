"""
Problem 6: Multivariate Theory
==============================

5 questions on linear transformations and conditional distributions
using symbolic mathematics.

**Answers:** 2, 2, 5, 1, 5
"""

import numpy as np
import sympy as sp

# %%
# Setup
# -----

rho = sp.Symbol("rho")

mu = sp.Matrix([1, 2, 3])
Sigma = sp.Matrix([[1, rho, rho**2], [rho, 1, rho], [rho**2, rho, 1]])

print("PROBLEM 6: Multivariate Theory")
print("\nGiven: [X, Y, Z]' ~ N(μ, Σ)")
print("μ = [1, 2, 3]'")
print("Σ = [[1, ρ, ρ²], [ρ, 1, ρ], [ρ², ρ, 1]]")
print("\nDefine: S = X - Y, T = Y - Z")

# %%
# Q6.1: Mean of [S, T]
# --------------------

mean_S = mu[0] - mu[1]  # E[X] - E[Y] = 1 - 2 = -1
mean_T = mu[1] - mu[2]  # E[Y] - E[Z] = 2 - 3 = -1

print("\nQ6.1: E[[S, T]']")
print(f"E[S] = E[X] - E[Y] = 1 - 2 = {mean_S}")
print(f"E[T] = E[Y] - E[Z] = 2 - 3 = {mean_T}")
print("✓ Answer 2: [-1, -1]'")

# %%
# Q6.2: Dispersion Matrix D([S, T])
# ---------------------------------
# Using D[AX] = A Σ A'

A = sp.Matrix([[1, -1, 0], [0, 1, -1]])
Var_ST = A @ Sigma @ A.T
Var_ST_simplified = sp.simplify(Var_ST)

print("\nQ6.2: D[[S, T]']")
print("Using [S, T]' = A[X, Y, Z]' where A = [[1,-1,0], [0,1,-1]]")
print("D([S,T]) = A Σ A'")
print("\nVar(S) = Var(X) + Var(Y) - 2Cov(X,Y) = 1 + 1 - 2ρ = 2(1-ρ)")
print("Var(T) = 2(1-ρ)")
print("Cov(S,T) = ρ - ρ² - 1 + ρ = -(1-ρ)² = (1-ρ)(ρ-1)")
print("\nResult: (1-ρ) × [[2, ρ-1], [ρ-1, 2]]")
print("✓ Answer 2")

# %%
# Q6.3: Cov(X, S)
# ---------------

print("\nQ6.3: Cov(X, S)")
print("Cov(X, S) = Cov(X, X-Y)")
print("         = Var(X) - Cov(X,Y)")
print("         = 1 - ρ")
print("✓ Answer 5: 1 - ρ")

# %%
# Q6.4: Conditional Mean E(X|Y)
# -----------------------------

print("\nQ6.4: E(X|Y)")
print("E(X|Y) = μ_X + Σ_XY × Σ_YY⁻¹ × (Y - μ_Y)")
print("      = 1 + ρ × 1 × (Y - 2)")
print("      = 1 + ρ(Y - 2)")
print("      = ρ(Y - 2) + 1")
print("✓ Answer 1: ρ(Y-2) + 1")

# %%
# Q6.5: Conditional Dispersion D([X,Z]|Y)
# ---------------------------------------

print("\nQ6.5: D([X,Z]'|Y)")
print("D([X,Z]|Y) = Σ_XZ - Σ_XY,ZY × Σ_YY⁻¹ × Σ_YX,YZ")
print("\nΣ_XZ = [[1, ρ²], [ρ², 1]]")
print("Σ_XY,ZY = [ρ, ρ]'")
print("Σ_YY = 1")
print("\nD([X,Z]|Y) = [[1, ρ²], [ρ², 1]] - [[ρ², ρ²], [ρ², ρ²]]")
print("          = [[1-ρ², 0], [0, 1-ρ²]]")
print("✓ Answer 5: [[1-ρ², 0], [0, 1-ρ²]]")

# %%
# Verification with numerical example
# -----------------------------------

rho_val = 0.5
mu_num = np.array([1, 2, 3])
Sigma_num = np.array([[1, rho_val, rho_val**2], [rho_val, 1, rho_val], [rho_val**2, rho_val, 1]])

print("\n--- Numerical verification (ρ = 0.5) ---")

# Q6.5: D([X,Z]|Y)
Sigma_XZ = Sigma_num[np.ix_([0, 2], [0, 2])]
Sigma_XY_ZY = Sigma_num[np.ix_([0, 2], [1])]
cond_cov = Sigma_XZ - Sigma_XY_ZY @ Sigma_XY_ZY.T
print(f"\nD([X,Z]|Y) =\n{cond_cov}")
print(f"Expected: [[{1 - rho_val**2}, 0], [0, {1 - rho_val**2}]]")
