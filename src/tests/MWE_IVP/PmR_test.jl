#= Test derivation of the system matrix P - R =#
include("SpectralBases_MWE.jl")
using .SpectralBases: build_quadrature, create_basis_set
using QuadGK

M = 3

const τ_GL, w_GL = build_quadrature(M, "Legendre");
const τ_GC, w_GC = build_quadrature(M, "Chebyshev");

# Test integration
for i 




# Integrate ∫(dT_m/dτ T_l) dτ over [-1, 1]. Pick m = l = M + 1.


# Gauss-Lobatto integration. Pick m = l = M + 1.
const T_GL, dT_GL = create_basis_set(τ_GL, M, "Chebyshev");
