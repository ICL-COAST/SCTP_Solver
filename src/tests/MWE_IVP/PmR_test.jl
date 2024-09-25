#= Test derivation of the system matrix P - R =#
# include("SpectralBases_MWE.jl")
# using .SpectralBases: build_quadrature, create_basis_set
include("SpectralSolver_MWE.jl")
using .SpectralSolver: PRMatrix
using Revise
using Test

N = 1; M = 5;

PmR, P, R = PRMatrix(N,M)

# Output from Wolframalpha
P_true = [0 0 0 0 0 0
2 0 -2/3 0 -2/15 0
0 8/3 0 -8/5 0 -8/21
2 0 18/5 0 -18/7 0
0 32/15 0 32/7 0 -32/9
2 0 50/21 0 50/9 0]
@test isapprox(P, P_true)

# # Integrate ∫(dT_m/dτ T_l) dτ over [-1, 1]. Pick m = l = M + 1.


# # Gauss-Lobatto integration. Pick m = l = M + 1.
# const T_GL, dT_GL = create_basis_set(τ_GL, M, "Chebyshev");
