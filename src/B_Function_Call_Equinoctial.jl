#This is the code written by Egor Kaygorodov
#work started 1st July 2024 as part of a UROP
#with dr Davide amato
#this code implements a version of a modified
#galerkin method derived by Davide Amato and Anton Sabin
#for fast IVP solution

#this module runs the preliminary numerical testing campaign comparing different numerical solvers using a cowell formulation

#this testing campaign compares DOPRI853, MCPI and GCN
#for 3 separate test cases outlined in Inputs_IVP_1, Inputs_IVP_2, Inputs_IVP_3, which have 3 different eccentricities
#for each test case, the methods are calibrated until a similar order of magnitude of error is produced
#the graph of the error orders of magnitude is plotted
#and then the CPU time and function calls for each method are also plotted
#6 graphs are produced here

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
include("SpectralBases.jl")
include("SpectralSolver.jl")
include("CowellPropagator.jl")
include("EquinoctialPropagator.jl")
include("Constants.jl")
include("IVP_Solvers.jl")
include("MCPISolver.jl")
include("Quadrature.jl")
include("Inputs_IVP_1.jl")

#Define Simulation Parameters separately:
global rel_tol             =       1e-13                           # [-] - relative tolerance of the RK45 integrator
global abs_tol             =       1e-13                           # [ER] - absolute tolerance of the RK45 integrator  


#counter for function calls, check destats.nf is accurate
global function_calls = 0
const orbital_period_multiplier = 30
formulation_type = "Equinoctial"
const C_gp, S_gp = import_coefficients("physical_data/EGM2008_to2190_TideFree", geopotential_order) # import EGM2008 coefficients

h_adaptive      =     false

#SOLVE TEST CASE 1
include("Inputs_IVP_1.jl")
x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds
t_span = time_span_IVP ./ norm_time_E
sys_size = size(x0)[1]

#VERN8 benchmark
println("Running Benchmark Solution for case 1")
global vern8_rel_tol             =       1e-15                           # [-] - relative tolerance of the VERN8 integrator used as a benchmark 
global vern8_abs_tol             =       1e-15                           # [ER] - absolute tolerance of the VERN8 integrator used as a benchmark
#t_ref_1, sol_ref_1 = Vern8_solve(x0, t_span, vern8_rel_tol, vern8_abs_tol, formulation_type);


global t_ref_1 = []
global sol_ref_1 = []

try
    data = readdlm(string(vern8_rel_tol)*"_"*string(orbital_period_multiplier)*"E_test_case_1_benchmark.txt")
    # Extract the columns into separate arrays
    global t_ref_1 = data[:, 1]
    global sol_ref_1 = data[:, 2:7]

catch
    tref, solf = Vern8_solve(x0, t_span, vern8_rel_tol, vern8_abs_tol, formulation_type);
    global t_ref_1 = tref
    global sol_ref_1 = solf
    data = [t_ref_1 sol_ref_1]
    writedlm(string(vern8_rel_tol)*"_"*string(orbital_period_multiplier)*"E_test_case_1_benchmark.txt", data, ' ')
end



#MCPI
println("Running MCPI for case 1")
global m_order_IVP         =       50                            # [-] - MCPI order of the polynomial approximation
global m_interval_IVP      =       time_span_IVP[2]/50               # [seconds, multiples of solar days] - the MCPI interval
m_spectral_iter_IVP        =       100000                               # [-] - maximum number of iterations of the MPCI propagator
mcp_it_tol    =       1e-9                             # [-] - relative tolerance of the spectral integrator
mcp_rel_tol   =       1e-9                                      # [-] - relative tolerance of the adaptive spectral integrator
global function_calls = 0

t_mcp_1, c_mcp_1, i_mcp_1, e_mcp_1 = MCPI_solve(x0, t_span, m_interval_IVP / norm_time_E, mcp_it_tol, mcp_rel_tol, m_spectral_iter_IVP, formulation_type)
mcpi_f = function_calls
println(mcpi_f)
spec_mcpi_time = @belapsed $MCPI_solve(x0, t_span, m_interval_IVP / norm_time_E, mcp_it_tol, mcp_rel_tol, m_spectral_iter_IVP, $formulation_type)
println(mcpi_f)

#GCN
println("Running GCN for case 1")
global s_order_IVP         =      20                             # [-] - MCPI order of the polynomial approximation
global s_interval_IVP      =      time_span_IVP[2]/30               # [seconds, multiples of solar days] - the GCN interval
s_spectral_iter_IVP        =      100000                               # [-] - maximum number of iterations of the GCN propagator
gcn_it_tol    =       1e-8                            # [-] - relative tolerance of the spectral integrator
gcn_rel_tol   =       gcn_it_tol                                     # [-] - relative tolerance of the adaptive spectral integrator

global function_calls = 0
global call_structure = zeros(1,3)
#intialise basis functions
#create legendere and chebyshev based quadrature nodes and weights
const gl_nodes_IVP, gl_weights_IVP = build_quadrature(s_order_IVP, "Legendre");
const gc_nodes_IVP, gc_weights_IVP = build_quadrature(s_order_IVP, "Chebyshev");

#create and evaluate quadrature polynomial functions
const spectral_basis_gl_IVP, spectral_basis_deriv_gl_IVP, spectral_basis_deriv2_gl_IVP = create_basis_set(gl_nodes_IVP, s_order_IVP, "Chebyshev");
const spectral_basis_gc_IVP, spectral_basis_deriv_gc_IVP, spectral_basis_deriv2_gc_IVP = create_basis_set(gc_nodes_IVP, s_order_IVP, "Chebyshev");

const sys_matrix_IVP = build_system_matrix_IVP(sys_size); # P - Br returned
const sys_matrix_IVP_inv = inv(sys_matrix_IVP); # pre-compute P - Br inverse


t_gcn_1, c_gcn_1, i_gcn_1, e_gcn_1 = Spectral_solve(x0, t_span, s_interval_IVP / norm_time_E, gcn_it_tol, gcn_rel_tol, s_spectral_iter_IVP, formulation_type)
gcn_f = function_calls
spec_gcn_time = @belapsed $Spectral_solve(x0, t_span, s_interval_IVP / norm_time_E, gcn_it_tol, gcn_rel_tol, s_spectral_iter_IVP, $formulation_type)


println("Case 1 Calculations Complete")


#DOPRI 853
println("Running DOPRI853 for case 1")
global dop_rel_tol             =       1e-11                           # [-] - relative tolerance of the DOPRI853 integrator
global dop_abs_tol             =       1e-11                           # [ER] - absolute tolerance of the DOPRI853 integrator
global function_calls = 0

t_dop_1, sol_dop_1 = DOPRI853_solve(x0, t_span, dop_rel_tol, dop_abs_tol, formulation_type);
dop_f = function_calls
spec_dop_time = @belapsed $DOPRI853_solve(x0, t_span, dop_rel_tol, dop_abs_tol, $formulation_type);


#RK45 with TSIT
println("Running RK45 for case 1")
global rk45_rel_tol             =       1e-11                           # [-] - relative tolerance of the DOPRI853 integrator
global rk45_abs_tol             =       1e-11                           # [ER] - absolute tolerance of the DOPRI853 integrator
global function_calls = 0

t_rk_1, sol_rk_1 = RK45_solve(x0, t_span, rk45_rel_tol, rk45_abs_tol, formulation_type);
rk45_f = function_calls
spec_rk_time = @belapsed $RK45_solve(x0, t_span, rk45_rel_tol, rk45_abs_tol, $formulation_type);

#post process spectral results
println("Post Processing Case 1 Spectral results")
mcp_time_fin_1, mcp_sol_fin_1 = post_process_IVP(t_mcp_1, c_mcp_1, sys_size, 10000, formulation_type)
gcn_time_fin_1, gcn_sol_fin_1 = post_process_IVP(t_gcn_1, c_gcn_1, sys_size, 10000, formulation_type)
println("Post Processing Spectral results complete")

#calculate error wrt vern8
println("Calculating Error wrt benchmark")
dop_etime_1, dop_e_1 = compute_error(t_ref_1, sol_ref_1, t_dop_1, sol_dop_1, 20, "IVP")
rk_etime_1, rk_e_1 = compute_error(t_ref_1, sol_ref_1, t_rk_1, sol_rk_1, 20, "IVP")
mcp_etime_1, mcp_e_1 = compute_error(t_ref_1, sol_ref_1, mcp_time_fin_1, mcp_sol_fin_1, 20, "IVP")
gcn_etime_1, gcn_e_1 = compute_error(t_ref_1, sol_ref_1, gcn_time_fin_1, gcn_sol_fin_1, 20, "IVP")
println("Calculating Error complete")
#convert to log and correct time units

# Create the plot
PyPlot.figure(figsize=(7, 5))

PyPlot.plot(dop_etime_1.*(norm_time_E./3600), dop_e_1.*norm_distance_E, label="DOPRI853",marker="d")
PyPlot.plot(rk_etime_1.*(norm_time_E./3600), rk_e_1.*norm_distance_E, label="RK45",marker="d")  
PyPlot.plot(mcp_etime_1.*(norm_time_E./3600), mcp_e_1.*norm_distance_E, label="MCPI",marker="d") 
PyPlot.plot(gcn_etime_1.*(norm_time_E./3600), gcn_e_1.*norm_distance_E, label="GCN",marker="d")
#PyPlot.plot(dop_etime_1, dop_e_1, label="DOP",marker="d")

# Add labels and title
PyPlot.yscale("log")
ylim(10e-6,10e2)
PyPlot.xlabel("t /hours")
PyPlot.ylabel("Deviation /km")
PyPlot.title("Plot of Error for case 1, orbits:"*string(orbital_period_multiplier))
PyPlot.legend()

#make bar chart of cpu times and function calls

solvers = ["DOP853","RK45","MCPI","GCN"]

N = length(solvers)
ind = collect(1:N)
width = 0.35

fig, ax1 = subplots()
bars1 = ax1.bar(ind, [dop_f,rk45_f,mcpi_f,gcn_f], width, label="Function Calls", color="b")
PyPlot.yscale("log")

ax2 = ax1.twinx()
bars2 = ax2.bar(ind .+ width, [spec_dop_time, spec_rk_time, spec_mcpi_time, spec_gcn_time], width, label="CPU Times", color="r")
PyPlot.yscale("log")

# Add labels, title, and legend
ax1.set_xlabel("Solvers")
ax1.set_ylabel("Function Calls")
ax2.set_ylabel("CPU Time")
plt.title("")

# Set the x-axis tick labels
ax1.set_xticks(ind .+ width / 2)
ax1.set_xticklabels(solvers)

# Add legends
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Display the plot


#=

#run reference solution used to calculate error for case 1
println("Running Benchmark Solution for case 1")
include("Inputs_IVP_1.jl")
x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds
t_span = time_span_IVP ./ norm_time_E

global vern8_rel_tol             =       1e-8                           # [-] - relative tolerance of the VERN8 integrator used as a benchmark 
global vern8_abs_tol             =       1e-8                           # [ER] - absolute tolerance of the VERN8 integrator used as a benchmark
t_ref_1, sol_ref_1 = Vern8_solve(x0, t_span, vern8_rel_tol, vern8_abs_tol, formulation_type);
println("Benchmarking finished for case 1")


#run reference solution used to calculate error for case 2
println("Running Benchmark Solution for case 2")
include("Inputs_IVP_2.jl")

x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds
t_span = time_span_IVP ./ norm_time_E

global vern8_rel_tol             =       1e-8                           # [-] - relative tolerance of the VERN8 integrator used as a benchmark 
global vern8_abs_tol             =       1e-8                           # [ER] - absolute tolerance of the VERN8 integrator used as a benchmark
t_ref_2, sol_ref_2 = Vern8_solve(x0, t_span, vern8_rel_tol, vern8_abs_tol, formulation_type);
println("Benchmarking finished for case 2")


=#

#=
#run reference solution used to calculate error for case 3
println("Running Benchmark Solution for case 3")
include("Inputs_IVP_3.jl")

x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds
t_span = time_span_IVP ./ norm_time_E

global vern8_rel_tol             =       1e-8                           # [-] - relative tolerance of the VERN8 integrator used as a benchmark 
global vern8_abs_tol             =       1e-8                           # [ER] - absolute tolerance of the VERN8 integrator used as a benchmark
t_ref_3, sol_ref_3 = Vern8_solve(x0, t_span, vern8_rel_tol, vern8_abs_tol, formulation_type);
println("Benchmarking finished for case 3")

=#
#=

#=

#DOPRI853 for cases 1, 2 and 3

#case 1
println("Running DOPRI853 for case 1")
include("Inputs_IVP_1.jl")

x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds
t_span = time_span_IVP ./ norm_time_E
println("t span")
println(t_span)
global dop_rel_tol             =       1e-3                           # [-] - relative tolerance of the DOPRI853 integrator
global dop_abs_tol             =       1e-3                           # [ER] - absolute tolerance of the DOPRI853 integrator
t_dop_1, sol_dop_1 = DOPRI853_solve(x0, t_span, dop_rel_tol, dop_abs_tol, formulation_type);
println("DOPRI853 finished for case 1")


#case 2
println("Running DOPRI853 for case 2")
include("Inputs_IVP_2.jl")

x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds
t_span = time_span_IVP ./ norm_time_E

global dop_rel_tol             =       1e-3                           # [-] - relative tolerance of the DOPRI853 integrator
global dop_abs_tol             =       1e-3                           # [ER] - absolute tolerance of the DOPRI853 integrator
t_dop_2, sol_dop_2 = DOPRI853_solve(x0, t_span, dop_rel_tol, dop_abs_tol, formulation_type);
println("DOPRI853 finished for case 2")

=#
#case 3

println("Running DOPRI853 for case 3")
include("Inputs_IVP_3.jl")

x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds
t_span = time_span_IVP ./ norm_time_E

global dop_rel_tol             =       1e-3                           # [-] - relative tolerance of the DOPRI853 integrator
global dop_abs_tol             =       1e-3                           # [ER] - absolute tolerance of the DOPRI853 integrator
t_dop_3, sol_dop_3 = DOPRI853_solve(x0, t_span, dop_rel_tol, dop_abs_tol, formulation_type);
println("DOPRI853 finished for case 3")



#=
h_adaptive      =       false

#MCPI for cases 1, 2 and 3
println("Running MCPI")

#case 1
println("Running MCPI for case 1")
include("Inputs_IVP_1.jl")

x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds

global m_order_IVP         =       50                             # [-] - MCPI order of the polynomial approximation
global m_interval_IVP      =       time_span_IVP[2]/100               # [seconds, multiples of solar days] - the MCPI interval
m_spectral_iter_IVP        =       10000                               # [-] - maximum number of iterations of the MPCI propagator

mcp_it_tol    =       1e-6                             # [-] - relative tolerance of the spectral integrator
mcp_rel_tol   =       1e-6                                      # [-] - relative tolerance of the adaptive spectral integrator

t_mcp_1, c_mcp_1, i_mcp_1, e_mcp_1 = MCPI_solve(x0, t_span./ norm_time_E, m_interval_IVP / norm_time_E, mcp_it_tol, mcp_rel_tol, m_spectral_iter_IVP, formulation_type)

println("MCPI finished for case 1")

#case 2
println("Running MCPI for case 2")
include("Inputs_IVP_2.jl")

x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds

global m_order_IVP         =       50                             # [-] - MCPI order of the polynomial approximation
global m_interval_IVP      =       time_span_IVP[2]/100               # [seconds, multiples of solar days] - the MCPI interval

t_mcp_2, c_mcp_2, i_mcp_2, e_mcp_2 = MCPI_solve(x0, t_span./ norm_time_E, m_interval_IVP / norm_time_E, mcp_it_tol, mcp_rel_tol, m_spectral_iter_IVP, formulation_type)

println("MCPI finished for case 2")

#case 3
println("Running MCPI for case 3")
include("Inputs_IVP_3.jl")

x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds

global m_order_IVP         =       50                             # [-] - MCPI order of the polynomial approximation
global m_interval_IVP      =       time_span_IVP[2]/100               # [seconds, multiples of solar days] - the MCPI interval


t_mcp_3, c_mcp_3, i_mcp_3, e_mcp_3 = MCPI_solve(x0, t_span ./ norm_time_E, m_interval_IVP / norm_time_E, mcp_it_tol, mcp_rel_tol, m_spectral_iter_IVP, formulation_type)

println("MCPI finished for case 3")
=#
#GCN for cases 1, 2 and 3

#case 1

h_adaptive = false


println("Running GCN for case 1")
include("Inputs_IVP_1.jl")


x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds
t_span = time_span_IVP

global s_order_IVP         =      100                             # [-] - MCPI order of the polynomial approximation
global s_interval_IVP      =      time_span_IVP[2]/100               # [seconds, multiples of solar days] - the GCN interval
s_spectral_iter_IVP        =      10000                               # [-] - maximum number of iterations of the GCN propagator


gcn_it_tol    =       1e-2                             # [-] - relative tolerance of the spectral integrator
gcn_rel_tol   =       1e-2                                      # [-] - relative tolerance of the adaptive spectral integrator

#intialise basis functions

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


t_gcn_1, c_gcn_1, i_gcn_1, e_gcn_1 = Spectral_solve(x0, t_span ./ norm_time_E, s_interval_IVP / norm_time_E, gcn_it_tol, gcn_rel_tol, s_spectral_iter_IVP, formulation_type)

println("GCN finished for case 1")


#case 2
println("Running GCN for case 2")
include("Inputs_IVP_2.jl")

x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global time_span_IVP = (0.0 , orbital_period_multiplier * orbital_period); # in seconds
global s_interval_IVP      =      time_span_IVP[2]/100
t_span = time_span_IVP

gcn_it_tol    =       1e-2                             # [-] - relative tolerance of the spectral integrator
gcn_rel_tol   =       1e-2                                      # [-] - relative tolerance of the adaptive spectral integrator

#intialise basis functions
t_gcn_2, c_gcn_2, i_gcn_2, e_gcn_2 = Spectral_solve(x0, t_span ./ norm_time_E, s_interval_IVP / norm_time_E, gcn_it_tol, gcn_rel_tol, s_spectral_iter_IVP, formulation_type)

println("GCN finished for case 2")



#case 3
println("Running GCN for case 3")
include("Inputs_IVP_3.jl")

x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
sys_size = size(x0)[1]

global time_span_IVP = (0.0 , 0.01 * orbital_period);#(0.0 , orbital_period_multiplier * orbital_period); # in seconds
global s_interval_IVP      =      time_span_IVP[2]/100
t_span = time_span_IVP

gcn_it_tol    =       1e-6                             # [-] - relative tolerance of the spectral integrator
gcn_rel_tol   =       1e-6                                      # [-] - relative tolerance of the adaptive spectral integrator

#intialise basis functions
t_gcn_3, c_gcn_3, i_gcn_3, e_gcn_3 = Spectral_solve(x0, t_span ./ norm_time_E, s_interval_IVP / norm_time_E, gcn_it_tol, gcn_rel_tol, s_spectral_iter_IVP, formulation_type)

println("GCN finished for case 3")                                  # [-] - relative tolerance of the adaptive spectral integrator

test_t, test_sol = post_process_IVP(t_gcn_3, c_gcn_3, sys_size, 1000, formulation_type)
test_sol = kepler_to_cartesian(test_sol)

PyPlot.scatter(test_sol[:,1],test_sol[:,2],test_sol[:,3])

#plot results for case 1

#post process spectral results
println("Post Processing Spectral results")
mcp_time_fin_1, mcp_sol_fin_1 = post_process_IVP(t_mcp_1, c_mcp_1, sys_size, 10000, formulation_type)
gcn_time_fin_1, gcn_sol_fin_1 = post_process_IVP(t_gcn_1, c_gcn_1, sys_size, 10000, formulation_type)
println("Post Processing Spectral results complete")

#calculate error wrt vern8
println("Calculating Error")
mcp_etime_1, mcp_e_1 = compute_error(t_ref_1, sol_ref_1, mcp_time_fin_1, mcp_sol_fin_1, 20, "IVP")
gcn_etime_1, gcn_e_1 = compute_error(t_ref_1, sol_ref_1, gcn_time_fin_1, gcn_sol_fin_1, 20, "IVP")
#dop_etime_1, dop_e_1 = compute_error(t_ref_1, sol_ref_1, t_dop_1, t_ref_1, 20, "IVP")
println("Calculating Error complete")

#convert to log and correct time units

# Create the plot

PyPlot.plot(mcp_etime_1, mcp_e_1, label="MCPI",marker="d") 
PyPlot.plot(gcn_etime_1, gcn_e_1, label="GCN",marker="d")
#PyPlot.plot(dop_etime_1, dop_e_1, label="DOP",marker="d")

# Add labels and title
PyPlot.yscale("log")
PyPlot.xlabel("t /hours")
PyPlot.ylabel("Error")
PyPlot.title("Plot of Error for case 1")
PyPlot.legend()


=#