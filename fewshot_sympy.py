# Reimport sympy and redo the computation due to session reset
import sympy as sp

# Define variables again
B1, B2, B_star = sp.symbols('B1 B2 B_star')

# First Loss
p = 0.5 * (B1 + B2)
L1 = B_star * sp.log(p) + (1 - B_star) * sp.log(1 - p)

# Compute all second-order derivatives for the first loss
L1_Hessian = sp.Matrix([
    [sp.diff(sp.diff(L1, B1), B1), sp.diff(sp.diff(L1, B1), B2)],
    [sp.diff(sp.diff(L1, B2), B1), sp.diff(sp.diff(L1, B2), B2)]
])

# Substitute B1 = B2 = B_star
L1_Hessian_substituted = L1_Hessian.subs({B1: B_star, B2: B_star})

# Second Loss
L2 = B_star * sp.log(B1) + (1 - B_star) * sp.log(1 - B1) + B_star * sp.log(B2) + (1 - B_star) * sp.log(1 - B2)
L2 = 0.5 * L2

# Compute all second-order derivatives for the second loss
L2_Hessian = sp.Matrix([
    [sp.diff(sp.diff(L2, B1), B1), sp.diff(sp.diff(L2, B1), B2)],
    [sp.diff(sp.diff(L2, B2), B1), sp.diff(sp.diff(L2, B2), B2)]
])

# Substitute B1 = B2 = B_star
L2_Hessian_substituted = L2_Hessian.subs({B1: B_star, B2: B_star})

# Convert to LaTeX
L1_Hessian_latex = sp.latex(L1_Hessian_substituted)
L2_Hessian_latex = sp.latex(L2_Hessian_substituted)

print(L1_Hessian_latex)
print(L2_Hessian_latex)