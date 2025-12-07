"""
Problem 5: Factor Analysis
==========================

3 questions: number of factors (screeplot), communalities,
and score plot interpretation.

**Answers:** 4, 2, 2
"""

import matplotlib.pyplot as plt

# %%
# Eigenvalues from SAS
# --------------------

eigenvalues = [2.806, 2.190, 0.789, 0.469, 0.381, 0.253, 0.112]

# Communalities with 3 factors
communalities = {
    "B1": 0.91159983,
    "B2": 0.68926825,  # Lowest
    "B3": 0.99103354,
    "B4": 0.82877011,
    "B5": 0.77181742,
    "B6": 0.75630004,
    "B7": 0.83613014,
}

print("PROBLEM 5: Factor Analysis")
print("Library contents across municipalities")

# %%
# Q5.1: Screeplot - Number of Factors
# -----------------------------------

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, "bo-", markersize=10)
ax.axhline(y=1, color="r", linestyle="--", label="Kaiser criterion")
ax.set_xlabel("Factor")
ax.set_ylabel("Eigenvalue")
ax.set_title("Screeplot - Exam 2019 Problem 5")
ax.legend()
ax.set_xticks(range(1, len(eigenvalues) + 1))
plt.tight_layout()
plt.show()

print("\nQ5.1: Number of factors to retain")
print("Eigenvalues:", [f"{e:.3f}" for e in eigenvalues])
print("Clear elbow after factor 2")
print("✓ Answer 4: 2 factors")

# %%
# Q5.2: Lowest Communality
# ------------------------

print("\nQ5.2: Variable with lowest communality")
for var, comm in sorted(communalities.items(), key=lambda x: x[1]):
    marker = " ← LOWEST" if var == "B2" else ""
    print(f"  {var}: {comm:.4f}{marker}")
print("✓ Answer 2: B2")

# %%
# Q5.3: Score Plot Interpretation
# -------------------------------

print("\nQ5.3: Score plot interpretation")
print("Rotated factor pattern:")
print("  Factor 1: B1, B6, B7 (Books, Other, Electronic)")
print("  Factor 2: B2, B4, B5 (Audio, Movies, Multimedia)")
print("  Factor 3: B3 (Music)")
print("\nNotable observations:")
print("  - Læsø, Samsø: high on Factor 1")
print("  - København: low on Factor 2")
print("  - Gentofte: high on Factor 2")
print("  - Odense, Herlev: high on Factor 3")
print("✓ Answer 2: Correct interpretation")
