
using Logging: global_logger
using TerminalLoggers: TerminalLogger
using BenchmarkTools, ProfileView, Distributions, LinearAlgebra


include("Inputs_IVP.jl")
include("Inputs_BVP.jl")
include("StateConversions.jl")
include("Perturbations_Geopotential.jl")
include("Perturbations_Aerodynamic.jl")
include("Perturbations_SRP.jl")
include("SpectralBases.jl")
include("SpectralSolver.jl")
include("LeastSquaresSolver.jl")
include("CowellPropagator.jl")
include("EquinoctialPropagator.jl")
include("PLambert.jl")
include("Constants.jl")
include("IVP_Solvers.jl")
include("BVP_Solvers.jl")
include("DataHandling.jl")
include("Plotter.jl")
include("Analysis_Tools.jl")
include("MCPISolver.jl")

BLAS.set_num_threads(8)


x0 = [a / norm_distance_E, e, i * pi / 180, RAAN * pi / 180, omega * pi / 180, theta * pi / 180]
t_span = time_span_IVP ./ norm_time_E

sys_size = size(x0)[1]

const C_gp, S_gp = import_coefficients("physical_data/EGM2008_to2190_TideFree", geopotential_order) # import EGM2008 coefficients
const gl_nodes_IVP, gl_weights_IVP = build_quadrature(s_order_IVP, "Legendre");
const gc_nodes_IVP, gc_weights_IVP = build_quadrature(s_order_IVP, "Chebyshev");
const spectral_basis_gl_IVP, spectral_basis_deriv_gl_IVP, spectral_basis_deriv2_gl_IVP = create_basis_set(gl_nodes_IVP, s_order_IVP, "Chebyshev");
const spectral_basis_gc_IVP, spectral_basis_deriv_gc_IVP, spectral_basis_deriv2_gc_IVP = create_basis_set(gc_nodes_IVP, s_order_IVP, "Chebyshev");
const gl_nodes_BVP, gl_weights_BVP = build_quadrature(s_order_BVP, "Legendre");
const gc_nodes_BVP, gc_weights_BVP = build_quadrature(s_order_BVP, "Chebyshev");
const spectral_basis_gl_BVP, spectral_basis_deriv_gl_BVP, spectral_basis_deriv2_gl_BVP = create_basis_set(gl_nodes_BVP, s_order_BVP, "Chebyshev");
const spectral_basis_gc_BVP, spectral_basis_deriv_gc_BVP, spectral_basis_deriv2_gc_BVP = create_basis_set(gc_nodes_BVP, s_order_BVP, "Chebyshev");
const sys_matrix_IVP = build_system_matrix_IVP(6);
const sys_matrix_IVP_inv = inv(sys_matrix_IVP);
const sys_matrix_BVP = build_system_matrix_BVP(3);
const sys_matrix_BVP_inv = inv(sys_matrix_BVP);

println("I am using ", Threads.nthreads(), " threads!")

# for s_int in 0.1:0.1:1.5
#     global time_span_IVP = (0.0, s_int * 5431)
#     global s_interval_IVP = s_int * 5431
#     save_name = "sec2_eq_o" * string(s_order_IVP) * "_s" * string(s_int)
#     compare_numerical_spectral_IVP(save_name, x0, "Equinoctial", [RK45_solve], ["RK45"], [Spectral_solve, MCPI_solve], ["BGS", "MCPI"], rel_tol, abs_tol, 100)
# end

# for s_int in 1:1:14
#     global rel_tol = 10.0 ^ (- s_int)
#     global abs_tol = 10.0 ^ (- s_int)
#     save_name = "sec2_eq_DP8_tol" * string(s_int)
#     compare_numerical_spectral_IVP(save_name, x0, "Equinoctial", [DP8_solve], ["DP8"], [Spectral_solve], ["BGS"], rel_tol, abs_tol, 100)
# end

# t_spec, c_spec, e_mat, i_mat =  Spectral_solve(x0, time_span_IVP ./ norm_time_E, s_interval_IVP / norm_time_E, relative_tol_IVP, spectral_tol_IVP, spectral_iter_IVP, "Cowell");
# plot_spectral_solution("Results", t_spec, c_spec, "test_solution", "Cowell", 1000, "IVP", size(x0)[1])

compare_numerical_spectral_IVP("sec2_cw_o30_s01", x0, "Cowell", [Vern8_solve], ["Vern8"], [Spectral_solve], ["BGS"], rel_tol, abs_tol, 1000)

# t_spec, c_spec, e_mat, i_mat =  Spectral_solve(x0, time_span_IVP ./ norm_time_E, s_interval_IVP / norm_time_E, relative_tol_IVP, spectral_tol_IVP, spectral_iter_IVP, "Cowell");
# save_solution_spectral("Results/" * "test_solution" * "_performance_analysis", "test_solution" * "_" * "BGS", "Cowell", t_spec, c_spec, e_mat, i_mat, size(x0)[1], "IVP")

x1 = [x_1, y_1, z_1]
x2 = [x_2, y_2, z_2]

# compare_numerical_spectral_BVP("sec1_Lambert_tof_40", x1, x2, [RK4_solve_BVP], ["RK4"], [Spectral_solve_BVP, MCPI_solve_BVP], ["BGS", "MCPI"], num_time_step, 100)

# for s_int in 0.5:0.5:5.0
#     global time_span_BVP = (0.0, s_int)
#     save_name = "sec2_lb_o" * string(s_order_BVP) * "_s" * string(s_int)
#     compare_numerical_spectral_BVP(save_name, x1, x2, [RK4_solve_BVP], ["RK4"], [Spectral_solve_BVP, MCPI_solve_BVP], ["BGS", "MCPI"], num_time_step, 100)

# end

# for s_int in 0:0.2:2
#     global num_time_step = 10.0 ^ (- s_int)
#     save_name = "sec2_lb_RK4_stp" * string(s_int)
#     compare_numerical_spectral_BVP(save_name, x1, x2, [RK4_solve_BVP], ["RK4"], [Spectral_solve_BVP, MCPI_solve_BVP], ["BGS", "MCPI"], num_time_step, 100)
# end