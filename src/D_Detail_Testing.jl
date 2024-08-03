#This is the code written by Egor Kaygorodov
#work started 1st July 2024 as part of a UROP
#with dr Davide amato
#this code implements a version of a modified
#galerkin method derived by Davide Amato and Anton Sabin
#for fast IVP solution

#=
For test case 1 test the performance of DOPRI, VERN8 and RK45 for a range of error tolerances: e-2,e-3,e-4,e-5,e-6,e-7,e-8, e-9 vs benchmark e-12
Specifically for just one orbit
Plot the error and function call/ cpu time graphs
Even better do a error tolerance vs final deviation curve (i.e RK45 with a tolerance of e-5 leads to a 10m deviation over one orbit
Then for each of those error tolerances plot the function call and error surfaces of MCPI and GCN for a range of spectral intervals and spectral order
From these find the optimum interval/order combo for a given error tolerance run another test with it to plot function call/cpu time bar and an error grap
Plot/determine the variance of the error surface past the plateau
The idea is that past a certain point changing the params doesn't make a difference in error, it just raises computational complexity
Numerical determine that point and compare it to sim paramters
Plot for comparison based on the final deviation from vern8 e-12 benchmark (i.e RK45 with local tolerance e-4 comparative to GCN with local tolerance e-3 with the optimum interval/order
Repeat the same for a longer period (10 orbits)
Dynamics should be differen
Also should be done for test case 2
=#

#Julia modules
using LinearAlgebra
using LaTeXStrings, PyPlot

using Logging: global_logger
using TerminalLoggers: TerminalLogger
using BenchmarkTools, ProfileView, Distributions, LinearAlgebra
#using csv
#using DataFrames

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
const orbital_period_multiplier = 5
formulation_type = "Cowell"
const C_gp, S_gp = import_coefficients("physical_data/EGM2008_to2190_TideFree", geopotential_order) # import EGM2008 coefficients

#initialise simulation parameters for case 1
include("Inputs_IVP_1.jl")
x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds
t_span = time_span_IVP ./ norm_time_E
sys_size = size(x0)[1]


#create local error tolerance array
error_order_start = 1 # fixed, never change
error_order_stop = 6
offset = 4

e_tol_array = Int.(collect(range(error_order_start,stop = error_order_stop, length = error_order_stop)))

#initialise mesh for GCN

order_length = 11
interval_length = 11

s_orders = Int.(collect(range(10,stop = 100, length = order_length))) #broad sweep here too, check orders
s_intervals = Int.(collect(range(10,stop = 100, length = interval_length))) 

order_grid, interval_grid = meshgrid(s_orders,s_intervals) 

func_c_grid = zeros(order_length,interval_length)
error_grid = zeros(order_length,interval_length)

#VERN8 benchmark
println("Running Benchmark Solution for case 1")
global vern8_rel_tol             =       1e-15                           # [-] - relative tolerance of the VERN8 integrator used as a benchmark 
global vern8_abs_tol             =       1e-15                           # [ER] - absolute tolerance of the VERN8 integrator used as a benchmark
#t_ref_1, sol_ref_1 = Vern8_solve(x0, t_span, vern8_rel_tol, vern8_abs_tol, formulation_type);


global t_ref_1 = []
global sol_ref_1 = []

try
    data = readdlm(string(vern8_rel_tol)*"_"*string(orbital_period_multiplier)*"_test_case_1_benchmark.txt")
    # Extract the columns into separate arrays
    global t_ref_1 = data[:, 1]
    global sol_ref_1 = data[:, 2:7]

catch
    tref, solf = Vern8_solve(x0, t_span, vern8_rel_tol, vern8_abs_tol, formulation_type);
    global t_ref_1 = tref
    global sol_ref_1 = solf
    data = [t_ref_1 sol_ref_1]
    writedlm(string(vern8_rel_tol)*"_"*string(orbital_period_multiplier)*"_test_case_1_benchmark.txt", data, ' ')
end


#create arrays to store CPU times, Function calls and deviations from benchmark for 3 methods, dopri, rk and vern
dev_array = zeros(error_order_stop,4)
fc_array = zeros(error_order_stop,4)
cpu_array = zeros(error_order_stop,4)

#create finite difference method performance curves
#for DOPRI853, for RK45 with tsit, for VERN8

for i = e_tol_array
    global local_rel_tol = 10.0 ^(-i-offset)
    println(local_rel_tol)
    println("Starting Local error tolerance"*string(local_rel_tol))
    
    #DOPRI 853
    println("Running DOPRI853")
    global function_calls = 0
    t_dop_1, sol_dop_1 = DOPRI853_solve(x0, t_span, local_rel_tol, local_rel_tol, formulation_type);
    fc_array[i,1] = function_calls
    time, error = compute_error(t_ref_1, sol_ref_1, t_dop_1, sol_dop_1, 2, "IVP")
    dev_array[i,1] =  error[2].*norm_distance_E# deviation in km
    cpu_array[i,1] = @belapsed $DOPRI853_solve(x0, t_span, local_rel_tol, local_rel_tol, $formulation_type);
    println("done with belapsed")

    #VERN8
    println("Running VERN8")
    global function_calls = 0
    t_vern_1, sol_vern_1 = Vern8_solve(x0, t_span, local_rel_tol, local_rel_tol, formulation_type);
    fc_array[i,2] = function_calls
    time, error = compute_error(t_ref_1, sol_ref_1, t_vern_1, sol_vern_1, 2, "IVP")
    dev_array[i,2] =  error[2].*norm_distance_E# deviation in km
    cpu_array[i,2]  = @belapsed $Vern8_solve(x0, t_span, local_rel_tol, local_rel_tol, $formulation_type);

    #RK45 with TSIT
    println("Running RK45 with tsit")
    global function_calls = 0
    t_rk_1, sol_rk_1 = RK45_solve(x0, t_span, local_rel_tol, local_rel_tol, formulation_type);
    fc_array[i,3] = function_calls
    time, error = compute_error(t_ref_1, sol_ref_1, t_rk_1, sol_rk_1, 2, "IVP")
    dev_array[i,3] =  error[2].*norm_distance_E# deviation in km
    cpu_array[i,3] = @belapsed $RK45_solve(x0, t_span, local_rel_tol, local_rel_tol, $formulation_type);

    println("Finishing Local error tolerance for finite differences"*string(local_rel_tol))

    global s_order_IVP = 15
    global s_interval_IVP = time_span_IVP[2]./10
    global s_spectral_iter_IVP = 100000;
    global h_adaptive = true

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
   
    t_gcn_1, c_gcn_1, i_gcn_1, e_gcn_1 = Spectral_solve(x0, t_span, s_interval_IVP / norm_time_E, local_rel_tol, local_rel_tol, s_spectral_iter_IVP, formulation_type)
        
    gcn_time_fin_1, gcn_sol_fin_1 = post_process_IVP(t_gcn_1, c_gcn_1, sys_size, 100, formulation_type)
    gcn_etime_1, gcn_e_1 = compute_error(t_ref_1, sol_ref_1, gcn_time_fin_1, gcn_sol_fin_1, 2, "IVP")
    
    fc_array[i,4] = function_calls
    dev_array[i,4] = abs(gcn_e_1[2].*norm_distance_E)

    
    println("Finishing Local error tolerance for GCN"*string(local_rel_tol))
end
e_tol_array = Float64.(e_tol_array)
# Create a plot
PyPlot.figure()

PyPlot.plot(10 .^ (-e_tol_array.-offset), dev_array[:,1], marker="o", linestyle="-", color="b", label="DOPRI853")
PyPlot.plot(10 .^ (-e_tol_array.-offset), dev_array[:,2], marker="o", linestyle="-", color="r", label="VERN8")
PyPlot.plot(10 .^ (-e_tol_array.-offset), dev_array[:,3], marker="o", linestyle="-", color="g", label="RK45 - tsit")
PyPlot.plot(10 .^ (-e_tol_array.-offset), dev_array[:,4], marker="o", linestyle="-", color="k", label="GCN")
PyPlot.xscale("log")
PyPlot.yscale("log")

PyPlot.xlabel("Local error tolerance")
PyPlot.ylabel("Deviation from Benchmark /km")
PyPlot.title("Deviations")
PyPlot.legend()

# Create a plot
PyPlot.figure()

PyPlot.plot(10 .^ (-e_tol_array.-offset), fc_array[:,1], marker="o", linestyle="-", color="b", label="DOPRI853")
PyPlot.plot(10 .^ (-e_tol_array.-offset), fc_array[:,2], marker="o", linestyle="-", color="r", label="VERN8")
PyPlot.plot(10 .^ (-e_tol_array.-offset), fc_array[:,3], marker="o", linestyle="-", color="g", label="RK45 - tsit")
PyPlot.plot(10 .^ (-e_tol_array.-offset), fc_array[:,4], marker="o", linestyle="-", color="k", label="GCN")
PyPlot.xscale("log")
PyPlot.yscale("log")

PyPlot.xlabel("Local error tolerance")
PyPlot.ylabel("Number of Function Calls")
PyPlot.title("Function calls")
PyPlot.legend()

# Create a plot
PyPlot.figure()

PyPlot.plot(10 .^ (-e_tol_array.-offset), cpu_array[:,1], marker="o", linestyle="-", color="b", label="DOPRI853")
PyPlot.plot(10 .^ (-e_tol_array.-offset), cpu_array[:,2], marker="o", linestyle="-", color="r", label="VERN8")
PyPlot.plot(10 .^ (-e_tol_array.-offset), cpu_array[:,3], marker="o", linestyle="-", color="g", label="RK45 - tsit")
PyPlot.xscale("log")
PyPlot.yscale("log")

PyPlot.xlabel("Local error tolerance")
PyPlot.ylabel("CPU time")
PyPlot.title("CPU")
PyPlot.legend()


#Local error function plot

PyPlot.figure()

PyPlot.plot(dev_array[:,1], fc_array[:,1], marker="o", linestyle="-", color="b", label="DOPRI853")
PyPlot.plot(dev_array[:,2], fc_array[:,2], marker="o", linestyle="-", color="r", label="VERN8")
PyPlot.plot(dev_array[:,3], fc_array[:,3], marker="o", linestyle="-", color="g", label="RK45 - tsit")
PyPlot.plot(dev_array[:,4], fc_array[:,4], marker="o", linestyle="-", color="k", label="GCN")
PyPlot.xscale("log")
PyPlot.yscale("log")

PyPlot.xlabel("Deviation from error")
PyPlot.ylabel("Number of Function Calls")
PyPlot.title("Function calls")
PyPlot.legend()


#=
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

local_error_level = 1e-6

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

        func_c_grid[x,y] = function_calls
        error_grid[x,y] = abs(gcn_e_1[2].*norm_distance_E)
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
PyPlot.zscale("log")

ax2.plot_surface(order_grid, interval_grid, error_grid, cmap="viridis")
ax2.set_title("Error Grid")
ax2.set_xlabel("Order Grid")
ax2.set_ylabel("Interval Grid")
ax2.set_zlabel("Error")

=#