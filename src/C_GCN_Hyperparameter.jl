#This is the code written by Egor Kaygorodov
#work started 1st July 2024 as part of a UROP
#with dr Davide amato
#this code implements a version of a modified
#galerkin method derived by Davide Amato and Anton Sabin
#for fast IVP solution


#this code analyses the landscape of the GCN method hyper-parameters
#for a given level of error
#a range of spectral intervals and method orders is analysed
#then two surfaces are plotted: one of deviation from the benchmark produced by Vern8
#and another of the number of function calls to obtain this deviation

#cpu times would take too long and so are ignored for now

#Julia modules
using LinearAlgebra
using LaTeXStrings, PyPlot

using Logging: global_logger
using TerminalLoggers: TerminalLogger
using BenchmarkTools, ProfileView, Distributions, LinearAlgebra
using DataFrames

#Allows access to subroutines in other files

include("StateConversions.jl")
include("Perturbations_Geopotential.jl")
include("Perturbations_Aerodynamic.jl")
include("SpectralBases.jl")
include("SpectralSolver.jl")
include("CowellPropagator.jl")
include("EquinoctialPropagator.jl")
include("Constants.jl")
include("IVP_Solvers.jl")
include("MCPISolver.jl")
include("Quadrature.jl")
include("Inputs_IVP_1.jl")

#local useful functions
function meshgrid(x, y)
    X = repeat(x', length(y), 1)
    Y = repeat(y, 1, length(x))
    return X, Y
end

global function_calls = 0
const orbital_period_multiplier = 20
formulation_type = "Cowell"
const C_gp, S_gp = import_coefficients("physical_data/EGM2008_to2190_TideFree", geopotential_order) # import EGM2008 coefficients

#EXPLORE GCN HYPER-PARAMETERS IN CASE 1
include("Inputs_IVP_1.jl")
x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds
t_span = time_span_IVP ./ norm_time_E
sys_size = size(x0)[1]


#initialise mesh for GCN

order_length = 11
interval_length = 11


s_orders = Int.(collect(range(20,stop = 50, length = order_length))) #broad sweep here too, check orders
s_intervals = Int.(collect(range(20,stop = 50, length = interval_length))) 

order_grid, interval_grid = meshgrid(s_orders,s_intervals) 

func_c_grid = zeros(interval_length,order_length)
error_grid = zeros(interval_length,order_length)


#VERN8 benchmark
println("Running Benchmark Solution for case 1")
global vern8_rel_tol             =       1e-12                          # [-] - relative tolerance of the VERN8 integrator used as a benchmark 
global vern8_abs_tol             =       1e-12                           # [ER] - absolute tolerance of the VERN8 integrator used as a benchmark

#generate benchmark file name
#bench_name = 

global t_ref_1 = []
global sol_ref_1 = []

try
    data = readdlm(string(vern8_rel_tol)*"_"*string(orbital_period_multiplier)*"_test_case_1_benchmark.txt")
    # Extract the columns into separate arrays
    global t_ref_1 = data[:, 1]
    global sol_ref_1 = data[:, 2:7]

catch
    t_ref_1, sol_ref_1 = Vern8_solve(x0, t_span, vern8_rel_tol, vern8_abs_tol, formulation_type);
    data = [t_ref_1 sol_ref_1]
    writedlm(string(vern8_rel_tol)*"_"*string(orbital_period_multiplier)*"_test_case_1_benchmark.txt", data, ' ')
end


#generate mesh for hyperparameter testing
#sub interval mesh

# this is into how many sections the IVP interval should be split into
#hypothesis that the absolute time doesnt matter, what matters is ratio of segments its split into
#broad sweep here, 10 vs 11 segments shouldn't matter

#order_length = 6
#interval_length = 6


#s_orders = Int.(collect(range(10,stop = 200, length = order_length))) #broad sweep here too, check orders
#s_intervals = Int.(collect(range(5,stop = 105, length = interval_length))) 



s_spectral_iter_IVP        =      100000                               # [-] - maximum number of iterations of the GCN propagator

local_error_level = 1e-8

gcn_it_tol    =       local_error_level                             # [-] - relative tolerance of the spectral integrator
gcn_rel_tol   =       local_error_level                                      # [-] - relative tolerance of the adaptive spectral integrator

h_adaptive      =      false
global run_number = 0

total_numb = order_length*interval_length
for x = 1:order_length
    for y = 1:interval_length
        global s_order_IVP = order_grid[y,x]
        global s_interval_IVP = time_span_IVP[2]./interval_grid[y,x]

        global function_calls = 0

        #intialise basis functions
        #create legendere and chebyshev based quadrature nodes and weights
        global gl_nodes_IVP, gl_weights_IVP = build_quadrature(s_order_IVP, "Legendre");
        global gc_nodes_IVP, gc_weights_IVP = build_quadrature(s_order_IVP, "Chebyshev");

        #create and evaluate quadrature polynomial functions
        global spectral_basis_gl_IVP, spectral_basis_deriv_gl_IVP, spectral_basis_deriv2_gl_IVP = create_basis_set(gl_nodes_IVP, s_order_IVP, "Chebyshev");
        global spectral_basis_gc_IVP, spectral_basis_deriv_gc_IVP, spectral_basis_deriv2_gc_IVP = create_basis_set(gc_nodes_IVP, s_order_IVP, "Chebyshev");

        global sys_matrix_IVP = build_system_matrix_IVP(sys_size); # P - Br returned
        global sys_matrix_IVP_inv = inv(sys_matrix_IVP); # pre-compute P - Br inverse
        #println(gcn_it_tol)
        t_gcn_1, c_gcn_1, i_gcn_1, e_gcn_1 = Spectral_solve(x0, t_span, s_interval_IVP / norm_time_E, gcn_it_tol, gcn_rel_tol, s_spectral_iter_IVP, formulation_type)
    
        gcn_time_fin_1, gcn_sol_fin_1 = post_process_IVP(t_gcn_1, c_gcn_1, sys_size, 100, formulation_type)
        gcn_etime_1, gcn_e_1 = compute_error(t_ref_1, sol_ref_1, gcn_time_fin_1, gcn_sol_fin_1, 2, "IVP")

        func_c_grid[y,x] = function_calls
        error_grid[y,x] = abs(gcn_e_1[2].*norm_distance_E)
        global run_number += 1
        println("Progress percent "*string(round(run_number/total_numb*100, digits = 3)))
    end
end

#PLOT SURFACE PLOTS

# Example data (replace these with your actual data)
# Plot the func_c_grid surface
figure()
ax1 = subplot(121, projection="3d")


ax1.plot_surface(order_grid, interval_grid, func_c_grid, cmap="viridis")
ax1.set_title("Function C Grid")
ax1.set_xlabel("Order Grid")
ax1.set_ylabel("Interval Grid")
ax1.set_zlabel("Func C")

# Plot the error_grid surface
ax2 = subplot(122, projection="3d")

#PyPlot.yscale("log")
PyPlot.zscale("normal")
PyPlot.zlim([1,-4])
ax2.plot_surface(order_grid, interval_grid, log10.(error_grid), cmap="viridis")
ax2.set_title("Error Grid")
ax2.set_xlabel("Order Grid")
ax2.set_ylabel("Interval Grid")
ax2.set_zlabel("Error")

