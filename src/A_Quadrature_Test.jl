#This is the code written by Egor Kaygorodov
#work started 1st July 2024 as part of a UROP
#with dr Davide amato
#this code implements a version of a modified
#galerkin method derived by Davide Amato and Anton Sabin
#for fast IVP solution

#this is the main file for the testing of how the order of the quadrature impacts accuracy

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



#Define Simulation Parameters separately:

global time_span_IVP       =       (0.0 , 2.0 * 86164.0905)           # [seconds, multiples of solar days] - total integration time time_span
global s_interval_IVP      =       0.1 * 86164.0905                   # [seconds, multiples of solar days] - the spectral time interval
global s_order_IVP         =       100                                # [-] - spectral order of the polynomial approximation
global s_order_quad        =       s_order_IVP#Int(ceil((s_order_IVP+1)/2))       # [-] - order of quadrature  
s_order_quad_og = s_order_quad
#additional solver quadrature orders
global s_order_quad_1        =       s_order_quad+1#Int(ceil((s_order_IVP+1)/2))       # [-] - order of quadrature  
global s_order_quad_2        =       s_order_quad+10#Int(ceil((s_order_IVP+1)/2))       # [-] - order of quadrature  
global s_order_quad_3        =       s_order_quad+15#Int(ceil((s_order_IVP+1)/2))       # [-] - order of quadrature  

global m_order_IVP         =       200                                # [-] - MCPI order of the polynomial approximation
global rel_tol             =       1e-13                              # [-] - relative tolerance of the RK45 integrator
global abs_tol             =       1e-13                              # [ER] - absolute tolerance of the RK45 integrator
global vern8_rel_tol             =       1e-15                       # [-] - relative tolerance of the RK45 integrator
global vern8_abs_tol             =       1e-15                        # [ER] - absolute tolerance of the RK45 integrator

#spectral integrator parameters
spectral_tol_IVP    =       1e-6                                      # [-] - relative tolerance of the spectral integrator
relative_tol_IVP    =       1e-6                                      # [-] - relative tolerance of the adaptive spectral integrator
spectral_iter_IVP   =       10000                                     # [-] - maximum number of iterations of the spectral propagator
h_adaptive      =       false

#coordinate conversion for initial conditions

#Define Initial Conditions
include("Inputs_IVP_1.jl")
x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
t_span = time_span_IVP ./ norm_time_E

sys_size = size(x0)[1]

const C_gp, S_gp = import_coefficients("physical_data/EGM2008_to2190_TideFree", geopotential_order) # import EGM2008 coefficients

include("Initialise_Constant_Arrays.jl")

formulation_type = "Cowell"

#benchmark

#run reference solution used to calculate error for case 1
println("Running Benchmark Solution for case 1")
include("Inputs_IVP_1.jl")
x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global vern8_rel_tol             =       1e-10                           # [-] - relative tolerance of the VERN8 integrator used as a benchmark 
global vern8_abs_tol             =       1e-10                           # [ER] - absolute tolerance of the VERN8 integrator used as a benchmark
t_ref, sol_ref = Vern8_solve(x0, t_span, vern8_rel_tol, vern8_abs_tol, formulation_type);


#RUN GCN
solver = Spectral_solve
println("begin")

#call solver with above basis
t_spec, c_spec, e_mat, i_mat = solver(x0, time_span_IVP ./ norm_time_E, s_interval_IVP / norm_time_E, relative_tol_IVP, spectral_tol_IVP, spectral_iter_IVP, formulation_type);

#call solver with basis 1
global s_order_quad = s_order_quad_1
include("Initialise_Constant_Arrays.jl")
t_spec1, c_spec1, e_mat1, i_mat1 = solver(x0, time_span_IVP ./ norm_time_E, s_interval_IVP / norm_time_E, relative_tol_IVP, spectral_tol_IVP, spectral_iter_IVP, formulation_type);

#call solver with basis 2
global s_order_quad = s_order_quad_2
include("Initialise_Constant_Arrays.jl")
t_spec2, c_spec2, e_mat2, i_mat2 = solver(x0, time_span_IVP ./ norm_time_E, s_interval_IVP / norm_time_E, relative_tol_IVP, spectral_tol_IVP, spectral_iter_IVP, formulation_type);

#call solver with basis 3
global s_order_quad = s_order_quad_3
include("Initialise_Constant_Arrays.jl")
t_spec3, c_spec3, e_mat3, i_mat3 = solver(x0, time_span_IVP ./ norm_time_E, s_interval_IVP / norm_time_E, relative_tol_IVP, spectral_tol_IVP, spectral_iter_IVP, formulation_type);



#post process results to convert to appropriate coordinates
spectral_time, spectral_solution = post_process_IVP(t_spec, c_spec, sys_size, 10000, formulation_type)
spectral_time1, spectral_solution1 = post_process_IVP(t_spec1, c_spec1, sys_size, 10000, formulation_type)
spectral_time2, spectral_solution2 = post_process_IVP(t_spec2, c_spec2, sys_size, 10000, formulation_type)
spectral_time3, spectral_solution3 = post_process_IVP(t_spec3, c_spec3, sys_size, 10000, formulation_type)


#spectral_solution = kepler_to_cartesian_array(spectral_solution) #convert coordinates to cartesian from kepler

time_samps, error_samps = compute_error(t_ref,sol_ref,spectral_time,spectral_solution,20,"IVP")
time_samps1, error_samps1 = compute_error(t_ref,sol_ref,spectral_time1,spectral_solution1,20,"IVP")
time_samps2, error_samps2 = compute_error(t_ref,sol_ref,spectral_time2,spectral_solution2,20,"IVP")
time_samps3, error_samps3 = compute_error(t_ref,sol_ref,spectral_time3,spectral_solution3,20,"IVP")

#convert to log and correct time units
time_samps = (time_samps.* norm_time_E)./60 # to minutes

time_samps1 = (time_samps1.* norm_time_E)./60 # to minutes
time_samps2 = (time_samps2.* norm_time_E)./60 # to minutes
time_samps3 = (time_samps3.* norm_time_E)./60 # to minutes

#run and plot test case

# Create the plot
PyPlot.plot(time_samps, error_samps, label="Quadrature "*string(s_order_quad_og),marker="d")
PyPlot.plot(time_samps1, error_samps1, label="Quadrature "*string(s_order_quad_1),marker="d")
PyPlot.plot(time_samps2, error_samps2, label="Quadrature "*string(s_order_quad_2),marker="d")
PyPlot.plot(time_samps3, error_samps3, label="Quadrature "*string(s_order_quad_3),marker="d")

# Add labels and title
PyPlot.yscale("log")
PyPlot.xlabel("t /min")
PyPlot.ylabel("log10 error")
PyPlot.title("Plot of Error")
PyPlot.legend()
