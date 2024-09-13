#= Test script for an individual orbit propagation with the GCN method =#

include("Inputs_IVP_test.jl")
include("Constants_test.jl")

include("../../SpectralBases.jl")
include("../../SpectralSolver.jl")
include("../../CowellPropagator.jl")
include("../../IVP_Solvers.jl")

include("../../Perturbations_Geopotential.jl")

include("../../DataHandling.jl")


# Set up ICs and timespan
x0 = [a / DU, e, i * π / 180, RAAN * π / 180, omega * π / 180, theta * π / 180]
t_span = time_span_IVP ./ TU
sys_size = size(x0)[1]  # = N in the GCN paper. Size of the state.

# Store const's
const C_gp, S_gp = import_coefficients(coefs_path, geopotential_order) # import EGM2008 coefficients
