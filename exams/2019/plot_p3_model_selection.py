"""
Problem 3: Regression Model Selection
=====================================

8 questions on model comparison: R² reduction, backward elimination,
F-tests, leverage, DFBETAS, and prediction intervals.

**Answers:** 1, 1, 2, 4, 4, 3, 2, 4
"""

import numpy as np

# %%
# SAS Output Values
# -----------------

R2_M1 = 0.6151  # Full model (5 variables)
R2_M2 = 0.5750  # Reduced model (2 variables)
SSE_M1 = 124382
SSE_M2 = 137360
MSE_M1 = 5407.93
MSE_M2 = 5283.08
n_obs = 29
k_M1 = 6  # 5 variables + intercept
k_M2 = 3  # 2 variables + intercept

print("Model comparison: Full (M1) vs Reduced (M2)")
print(f"M1: R² = {R2_M1}, SSE = {SSE_M1}, params = {k_M1}")
print(f"M2: R² = {R2_M2}, SSE = {SSE_M2}, params = {k_M2}")

# %%
# Q3.1: Reduction in R²
# ---------------------

r2_reduction = R2_M1 - R2_M2

print(f"\nQ3.1: R² reduction = {R2_M1} - {R2_M2} = {r2_reduction:.4f}")
print("✓ Answer 1: 0.0401")

# %%
# Q3.2: First Variable to Exclude
# -------------------------------

print("\nQ3.2: Backward elimination - first to exclude")
print("P-values from M1 output:")
print("  F1: 0.3876 (highest)")
print("  F2: 0.3557")
print("  F3: 0.3736")
print("  F4: 0.3620")
print("  F5: 0.3741")
print("✓ Answer 1: F1")

# %%
# Q3.3 & Q3.4: Nested F-test
# --------------------------

df_M1 = n_obs - k_M1  # 23
df_M2 = n_obs - k_M2  # 26
num_df = df_M2 - df_M1  # 3
den_df = df_M1  # 23

print("\nQ3.3: F-test formula")
print("F = [(SSE_M2 - SSE_M1)/(df_M2 - df_M1)] / [SSE_M1/df_M1]")
print("✓ Answer 2: F formula with SSE values")

print(f"\nQ3.4: Distribution F({num_df}, {den_df})")
print("✓ Answer 4: F(3, 23)")

# %%
# Q3.5: Highest Leverage
# ----------------------

print("\nQ3.5: Observation with highest leverage")
print("From SAS 'Hat Diag H' column:")
print("  Obs 8: h = 0.9360 (HIGHEST)")
print("✓ Answer 4: Observation 8")

# %%
# Q3.6: Highest Impact on Intercept
# ---------------------------------

print("\nQ3.6: Highest |DFBETAS| for intercept")
print("From SAS output:")
print("  Obs 5: |DFBETAS Intercept| = 0.7064 (highest)")
print("✓ Answer 3: Observation 5")

# %%
# Q3.7: Prediction Interval for Obs 3
# -----------------------------------

pred_3 = 245.8493
h_33 = 0.0979
RMSE = np.sqrt(MSE_M2)
se_pred = RMSE * np.sqrt(h_33)

print("\nQ3.7: 95% CI for observation 3")
print(f"Predicted: {pred_3}")
print(f"SE = RMSE × √h = {RMSE:.4f} × √{h_33} = {se_pred:.4f}")
print(f"CI: {pred_3} ± t(26)_0.975 × {se_pred:.4f}")
print("✓ Answer 2: 245.8493 ± t(26) × 22.7423")

# %%
# Q3.8: Variance if Obs 8 Deleted
# -------------------------------

r_8 = -2.0847
RStudent_8 = -0.1112
h_88 = 0.9360

sigma_hat_8 = r_8 / (RStudent_8 * np.sqrt(1 - h_88))
var_without_8 = sigma_hat_8**2

print("\nQ3.8: Variance if observation 8 deleted")
print("σ̂_(8) = residual / (RStudent × √(1-h))")
print(f"σ̂_(8) = {r_8} / ({RStudent_8} × √{1 - h_88:.4f})")
print(f"σ̂²_(8) = {var_without_8:.2f}")
print("✓ Answer 4: 5491.58")
