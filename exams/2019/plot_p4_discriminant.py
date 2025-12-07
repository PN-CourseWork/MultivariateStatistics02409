"""
Problem 4: Discriminant Analysis
================================

3 questions on LDA vs QDA: misclassification comparison,
Hotelling's T², and testing variable subsets.

**Answers:** 4, 2, 3
"""


# %%
# Data from SAS Output
# --------------------

n_high = 21  # High crime municipalities
n_low = 12  # Low crime municipalities
n_total = 33

# Misclassifications
misclass_qda = 1  # QDA: 1 misclassified
misclass_lda = 5  # LDA: 5 misclassified

# Generalized squared distances
D2_full = 3.62001  # Full model (8 variables)
D2_reduced = 1.52415  # Reduced model (3 variables)

print("PROBLEM 4: Discriminant Analysis")
print(f"Groups: High crime (n={n_high}), Low crime (n={n_low})")

# %%
# Q4.1: Reduction in Misclassifications
# -------------------------------------

reduction = misclass_lda - misclass_qda

print("\nQ4.1: Misclassification reduction (LDA → QDA)")
print(f"LDA: {misclass_lda} misclassified")
print(f"QDA: {misclass_qda} misclassified")
print(f"Reduction: {misclass_lda} - {misclass_qda} = {reduction}")
print("✓ Answer 4: 4")

# %%
# Q4.2: Hotelling's T²
# --------------------
# T² = (n₁ × n₂)/(n₁ + n₂) × D²

T2 = (n_high * n_low) / (n_high + n_low) * D2_full

print("\nQ4.2: Hotelling's T²")
print("T² = (n₁ × n₂)/(n₁ + n₂) × D²")
print(f"T² = ({n_high} × {n_low})/({n_total}) × {D2_full}")
print(f"T² = {T2:.4f}")
print("✓ Answer 2: 27.6437")

# %%
# Q4.3: Test for Variable Subset
# ------------------------------

p_full = 8  # U1-U8
p_reduced = 3  # U4, U6, U7

print("\nQ4.3: Testing if U1,U2,U3,U5,U8 contribute")
print("F-test formula:")
print("F = [(n+m-p_full-1)/(p_full-p_reduced)] ×")
print("    [D²_full - D²_reduced] /")
print("    [(n+m)(n+m-2)/(n×m) + D²_reduced]")
print(f"\nWith D²_full={D2_full}, D²_reduced={D2_reduced}")
print("✓ Answer 3: F formula with D² values")
