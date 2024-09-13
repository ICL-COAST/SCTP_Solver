#= Test script for an individual orbit propagation with the GCN method =#

include("Inputs_IVP_test.jl")
include("Constants_test.jl")

# include("../../SpectralBases.jl")
# include("../../SpectralSolver.jl")
# include("../../CowellPropagator.jl")
# include("../../IVP_Solvers.jl")

# include("../../Perturbations_Geopotential.jl")

# include("../../DataHandling.jl")

# Set up ICs and timespan
x0 = [a / DU, e, i * π / 180, RAAN * π / 180, omega * π / 180, theta * π / 180]
t_span = time_span_IVP ./ TU
sys_size = 6  # = N in the GCN paper. Size of the state.

# Load geopot coefs
const C_gp, S_gp = import_coefficients(coefs_path, geopotential_order)

# # Create Gauss-Lobatto and Gauss-Chebyshev nodes (abscissae) and weights.
# const gl_nodes_IVP, gl_weights_IVP = build_quadrature(s_order_IVP, "Legendre");
# const gc_nodes_IVP, gc_weights_IVP = build_quadrature(s_order_IVP, "Chebyshev");

# # Evaluate chosen spectral basis on Gauss-Lobatto and Gauss-Chebyshev nodes.
# const spectral_basis_gl_IVP, spectral_basis_deriv_gl_IVP, spectral_basis_deriv2_gl_IVP = create_basis_set(gl_nodes_IVP, s_order_IVP, "Chebyshev");
# const spectral_basis_gc_IVP, spectral_basis_deriv_gc_IVP, spectral_basis_deriv2_gc_IVP = create_basis_set(gc_nodes_IVP, s_order_IVP, "Chebyshev");

# # Evaluate system matrix 
# const sys_matrix_IVP = build_system_matrix_IVP(6);
# const sys_matrix_IVP_inv = inv(sys_matrix_IVP);