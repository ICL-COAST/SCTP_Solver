#This is the code written by Egor Kaygorodov
#work started 1st July 2024 as part of a UROP
#with dr Davide amato
#this code implements a version of a modified
#galerkin method derived by Davide Amato and Anton Sabin
#for fast IVP solution

#Julia modules
using LinearAlgebra
using LaTeXStrings, PyPlot

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


#Define Initial Conditions
include("Inputs_IVP.jl")
x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
t_span = time_span_IVP ./ norm_time_E

sys_size = size(x0)[1]

const C_gp, S_gp = import_coefficients("physical_data/EGM2008_to2190_TideFree", geopotential_order) # import EGM2008 coefficients

const gl_nodes_IVP, gl_weights_IVP = build_quadrature(s_order_IVP, "Legendre");
const gc_nodes_IVP, gc_weights_IVP = build_quadrature(s_order_IVP, "Chebyshev");

const spectral_basis_gl_IVP, spectral_basis_deriv_gl_IVP, spectral_basis_deriv2_gl_IVP = create_basis_set(gl_nodes_IVP, s_order_IVP, "Chebyshev");
const spectral_basis_gc_IVP, spectral_basis_deriv_gc_IVP, spectral_basis_deriv2_gc_IVP = create_basis_set(gc_nodes_IVP, s_order_IVP, "Chebyshev");

const sys_matrix_IVP = build_system_matrix_IVP(sys_size); # P - Br returned
const sys_matrix_IVP_inv = inv(sys_matrix_IVP); # pre-compute P - Br inverse


t_nums, sol_nums, num_b_times = [], [], []
t_specs, c_specs, e_mats, i_mats, spec_b_times = [], [], [], [], []



#RUN SCTP
solver = Spectral_solve
formulation_type = "Cowell"
t_spec, c_spec, e_mat, i_mat =  solver(x0, time_span_IVP ./ norm_time_E, s_interval_IVP / norm_time_E, relative_tol_IVP, spectral_tol_IVP, spectral_iter_IVP, formulation_type);
push!(t_specs, t_spec);
push!(c_specs, c_spec);
push!(e_mats, e_mat);
push!(i_mats, i_mat);

#RUN RK45
solver = RK45_solve
rk4_rel_tol = 1e-6
rk4_abs_tol = 1e-6
rk4_time, rk4_solution =  solver(x0, time_span_IVP ./ norm_time_E,  rk4_rel_tol, rk4_abs_tol, formulation_type);
#rk4_solution = kepler_to_cartesian_array(rk4_solution)

spectral_time, spectral_solution = post_process_IVP(t_spec, c_spec, sys_size, 1000, formulation_type)
#spectral_solution = kepler_to_cartesian_array(spectral_solution)
println("here")

#compare orbital elements

#PyPlot.figure(figsize=(7, 5))
#PyPlot.plot(spectral_time .* norm_time_E, spectral_solution[:, 2], label="");
#PyPlot.xlabel("Time [s]")
#PyPlot.ylabel("Eccentricity [-]")
#PyPlot.title("Eccentricity vs. time")
#PyPlot.close_figs()



PyPlot.figure(figsize=(7, 5))

PyPlot.subplot(2,3,1)
PyPlot.plot(spectral_time,spectral_solution[:,1],label="SCTP")
PyPlot.plot(rk4_time,rk4_solution[:,1],label="RK4")
PyPlot.xlabel("Time [s]")
PyPlot.ylabel("Semi-Major Axis")
PyPlot.legend()

PyPlot.subplot(2,3,2)
PyPlot.plot(spectral_time,spectral_solution[:,2],label="SCTP")
PyPlot.plot(rk4_time,rk4_solution[:,2],label="RK4")
PyPlot.xlabel("Time [s]")
PyPlot.ylabel("Eccentricity")
PyPlot.legend()

PyPlot.subplot(2,3,3)
PyPlot.plot(spectral_time,spectral_solution[:,3],label="SCTP")
PyPlot.plot(rk4_time,rk4_solution[:,3],label="RK4")
PyPlot.xlabel("Time [s]")
PyPlot.ylabel("Inclination")
PyPlot.legend()

PyPlot.subplot(2,3,4)
PyPlot.plot(spectral_time,spectral_solution[:,4],label="SCTP")
PyPlot.plot(rk4_time,rk4_solution[:,4],label="RK4")
PyPlot.xlabel("Time [s]")
PyPlot.ylabel("RAAN")
PyPlot.legend()

PyPlot.subplot(2,3,5)
PyPlot.plot(spectral_time,spectral_solution[:,5],label="SCTP")
PyPlot.plot(rk4_time,rk4_solution[:,5],label="RK4")
PyPlot.xlabel("Time [s]")
PyPlot.ylabel("Omega")
PyPlot.legend()

PyPlot.subplot(2,3,6)
PyPlot.plot(spectral_time,spectral_solution[:,6],label="SCTP")
PyPlot.plot(rk4_time,rk4_solution[:,6],label="RK4")
PyPlot.xlabel("Time [s]")
PyPlot.ylabel("Theta")
PyPlot.legend()


PyPlot.close_figs()

#quick and dirty orbit visual validation
#=PyPlot.figure(figsize=(7, 5))
PyPlot.plot3D(spectral_solution[:, 1], spectral_solution[:, 2], spectral_solution[:, 3], label="SCTP Orbit");
PyPlot.xlabel("X Coordinate [AU]")
PyPlot.ylabel("Y Coordinate [AU]")
PyPlot.zlabel("Z Coordinate [AU]")
PyPlot.plot3D(rk4_solution[:, 1], rk4_solution[:, 2], rk4_solution[:, 3], label="RK45 Orbit");
PyPlot.legend()

PyPlot.close_figs() =#

