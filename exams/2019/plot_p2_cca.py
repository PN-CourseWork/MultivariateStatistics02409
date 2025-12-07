"""
Problem 2: Canonical Correlation Analysis
=========================================

3 questions on CCA: squared correlations, variance explained,
and interpretation of canonical variates.

**Answers:** 5, 2, 4
"""


# %%
# Overview
# --------
# This problem uses SAS output from library use vs education data.

print("PROBLEM 2: Canonical Correlation Analysis")
print("Based on SAS output (Enclosure A)")

# %%
# Q2.1: First Canonical Correlation Squared
# -----------------------------------------

squared_can_corr_1 = 0.5849

print("\nQ2.1: Fraction of variation between V1 and W1")
print(f"From SAS: Squared canonical correlation = {squared_can_corr_1}")
print("✓ Answer 5: 0.5849")

# %%
# Q2.2: Variance in U1 Explained by V1
# ------------------------------------

corr_U1_V1 = -0.7532
var_explained = corr_U1_V1**2

print("\nQ2.2: Variance in U1 explained by V1")
print(f"From SAS: Corr(U1, V1) = {corr_U1_V1}")
print(f"Variance explained = r² = {var_explained:.4f}")
print("✓ Answer 2: 0.5673")

# %%
# Q2.3: Interpretation of W1
# --------------------------

print("\nQ2.3: Interpretation of first canonical variate W1")
print("Correlations between education variables and W1:")
print("  H1 (Primary):    +0.8011")
print("  H2 (High school):-0.5932")
print("  H3 (Vocational): +0.8678")
print("  H4 (Short):      -0.4394")
print("  H5 (Medium):     -0.7879")
print("  H6 (Bachelor):   -0.5541")
print("  H7 (Master):     -0.8623")
print("  H8 (Ph.D.):      -0.8735")
print("\nW1 contrasts primary+vocational (positive)")
print("vs higher education (negative)")
print("✓ Answer 4: Contrast primary/vocational vs rest")
