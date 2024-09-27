#= Test script for an individual orbit propagation with the GCN method =#

include("SpectralBases_MWE.jl")
include("SpectralSolver_MWE.jl")

using Revise
using .SpectralBases
using .SpectralSolver

# Define test RHS
function pendulum!(du, u, p, t)
    θ  = u[1]
    dθ = u[2]

    du[1] = dθ
    du[2] = -sin(θ)

    nothing
end


#= INITIALIZATIONS: ICs AND PARAMETERS =#
# Set up ICs and timespan
x0 = [3.13, 0]
tspan = (0.0, 2500.0)
N = 6                           # Dimension of the state vector.
M = 50                          # Order of the Chebyshev expansion.
Δt = (tspan[2] - tspan[1])/25.0


#= GCN INITIALIZATIONS =#
# Create Gauss-Lobatto and Gauss-Chebyshev nodes (abscissae) and weights.
τ_GL, w_GL = build_quadrature(M, "Legendre");
τ_GC, w_GC = build_quadrature(M, "Chebyshev");

# Evaluate chosen spectral basis on Gauss-Lobatto and Gauss-Chebyshev nodes.
T_GL, _, _ = create_basis_set(τ_GL, M, "Chebyshev");
T_GC, _, _ = create_basis_set(τ_GC, M, "Chebyshev");

# Evaluate coefficient matrices
PmR, _, _ = PRMatrix(N,M)

#= SOLVER CALL =#