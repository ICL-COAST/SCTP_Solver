#= Test script for an individual orbit propagation with the GCN method =#

include("Inputs_IVP_MWE.jl")
include("Constants_MWE.jl")
include("Perts_Geopot_MWE.jl")
include("SpectralBases_MWE.jl")

using .Geopotential
using .SpectralBases

# Set up ICs and timespan
x0 = [a / DU, e, i * π / 180, RAAN * π / 180, omega * π / 180, theta * π / 180]
t_span = time_span_IVP ./ TU
N = 6  # Dimension of the state vector.

# Load geopot coefs
const C_gp, S_gp = Geopotential.import_coefficients(coefs_path, geopotential_order)

# # Create Gauss-Lobatto and Gauss-Chebyshev nodes (abscissae) and weights.
const τ_GL, w_GL = build_quadrature(M, "Legendre");
const τ_GC, w_GC = build_quadrature(M, "Chebyshev");

# Evaluate chosen spectral basis on Gauss-Lobatto and Gauss-Chebyshev nodes.
const T_GL, T_GL′, T_GL′′ = create_basis_set(τ_GL, M, "Chebyshev");
const T_GC, T_GC′, T_GC′′ = create_basis_set(τ_GC, M, "Chebyshev");

# Evaluate system matrix
# const sys_matrix_IVP = build_system_matrix_IVP(N, M);
# const sys_matrix_IVP_inv = inv(sys_matrix_IVP);