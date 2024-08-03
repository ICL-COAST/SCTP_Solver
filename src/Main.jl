#This is the code written by Egor Kaygorodov
#work started 1st July 2024 as part of a UROP
#with dr Davide amato
#this code implements a version of a modified
#galerkin method derived by Davide Amato and Anton Sabin
#for fast IVP solution

#Julia modules
using LinearAlgebra
using LaTeXStrings, PyPlot

using Logging: global_logger
using TerminalLoggers: TerminalLogger
using BenchmarkTools, ProfileView, Distributions, LinearAlgebra


#Allows access to subroutines in other files

include("StateConversions.jl")
include("Perturbations_Geopotential.jl")
include("Perturbations_Aerodynamic.jl")
#include("Perturbations_SRP.jl")
include("SpectralBases.jl")
include("SpectralSolver.jl")
#include("LeastSquaresSolver.jl")
include("CowellPropagator.jl")
include("EquinoctialPropagator.jl")
#include("PLambert.jl")
include("Constants.jl")
include("IVP_Solvers.jl")
#include("DataHandling.jl")
#include("Plotter.jl")
#include("Analysis_Tools.jl")
#include("MCPISolver.jl")
include("Quadrature.jl")

#local useful functions
function meshgrid(x, y)
    X = repeat(x', length(y), 1)
    Y = repeat(y, 1, length(x))
    return X, Y
end

#Define Initial Conditions
include("Inputs_IVP_3.jl")

#Define Simulation Parameters separately:

global time_span_IVP       =       (0.0 , 0.64 * 86164.0905)           # [seconds, multiples of solar days] - total integration time time_span
global s_interval_IVP      =       time_span_IVP[2] / 1000                   # [seconds, multiples of solar days] - the spectral time interval
global s_order_IVP         =       500                             # [-] - spectral order of the polynomial approximation
global m_order_IVP         =       500                             # [-] - MCPI order of the polynomial approximation
global rel_tol             =       1e-13                           # [-] - relative tolerance of the RK45 integrator
global abs_tol             =       1e-13                           # [ER] - absolute tolerance of the RK45 integrator
global vern8_rel_tol             =       1e-13                           # [-] - relative tolerance of the RK45 integrator
global vern8_abs_tol             =       1e-13                           # [ER] - absolute tolerance of the RK45 integrator

#spectral integrator parameters
spectral_tol_IVP    =       1e-9                          # [-] - relative tolerance of the spectral integrator
relative_tol_IVP    =       1e-9                          # [-] - relative tolerance of the adaptive spectral integrator
spectral_iter_IVP   =       10000                           # [-] - maximum number of iterations of the spectral propagator
h_adaptive      =       false

#coordinate conversion for initial conditions
x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
t_span = time_span_IVP ./ norm_time_E

sys_size = size(x0)[1]

const C_gp, S_gp = import_coefficients("physical_data/EGM2008_to2190_TideFree", geopotential_order) # import EGM2008 coefficients

#create legendere and chebyshev based quadrature nodes and weights
const gl_nodes_IVP, gl_weights_IVP = build_quadrature(s_order_IVP, "Legendre");
const gc_nodes_IVP, gc_weights_IVP = build_quadrature(s_order_IVP, "Chebyshev");

#create and evaluate quadrature polynomial functions
const spectral_basis_gl_IVP, spectral_basis_deriv_gl_IVP, spectral_basis_deriv2_gl_IVP = create_basis_set(gl_nodes_IVP, s_order_IVP, "Chebyshev");
const spectral_basis_gc_IVP, spectral_basis_deriv_gc_IVP, spectral_basis_deriv2_gc_IVP = create_basis_set(gc_nodes_IVP, s_order_IVP, "Chebyshev");

const sys_matrix_IVP = build_system_matrix_IVP(sys_size); # P - Br returned
const sys_matrix_IVP_inv = inv(sys_matrix_IVP); # pre-compute P - Br inverse

t_nums, sol_nums, num_b_times = [], [], []
t_specs, c_specs, e_mats, i_mats, spec_b_times = [], [], [], [], []

formulation_type = "Cowell"

#counter for function calls, check destats.nf is accurate
global function_calls = 0

#run reference solution used to calculate error
println("Running Benchmark Solution")
t_ref, sol_ref = Vern8_solve(x0, t_span, vern8_rel_tol, vern8_abs_tol, formulation_type);
println("Benchmarking finished")

#=
#RUN GCN
solver = Spectral_solve
println("begin")


#orbital_period = 2 * pi * sqrt((a^3)/3.986004418e5) # Kepler's law for orbital period, inpt in km, output in seconds
#global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period);

#orbital_period_multiplier = 10
#orbital_period = 2 * pi * sqrt((a^3)/3.986004418e5) # Kepler's law for orbital period, inpt in km, output in seconds
#global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds

t_spec, c_spec, e_mat, i_mat = solver(x0, time_span_IVP ./ norm_time_E, s_interval_IVP / norm_time_E, relative_tol_IVP, spectral_tol_IVP, spectral_iter_IVP, formulation_type);


#=
#post process results to convert to appropriate coordinates
spectral_time, spectral_solution = post_process_IVP(t_spec, c_spec, sys_size, 10000, formulation_type)
#spectral_solution = kepler_to_cartesian_array(spectral_solution) #convert coordinates to cartesian from kepler

time_samps, error_samps = compute_error(t_ref,sol_ref,spectral_time,spectral_solution,20,"IVP")

#convert to log and correct time units
time_samps = (time_samps.* norm_time_E)./60 # to minutes

#run and plot test case

# Create the plot
PyPlot.plot(time_samps, error_samps, label="GCN",marker="d")

# Add labels and title
PyPlot.yscale("log")
PyPlot.xlabel("t /min")
PyPlot.ylabel("log10 error")
PyPlot.title("Plot of Error")
PyPlot.legend()
=#

=#