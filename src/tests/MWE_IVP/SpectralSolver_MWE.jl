module SpectralSolver

    using LinearAlgebra

    export build_system_matrix_IVP, PRMatrix

    function build_system_matrix_IVP(N::Int, M::Int, 
        τ_int::Vector{Float64}, T_GL::Matrix{Float64}, T_GL′::Matrix{Float64})

        #=
        This function builds the system matrix and optionally the Jacobian matrix
        based on spectral basis, its derivative, Gauss-Legendre weights, Jacobian array, 
        system size, and a boolean flag use_jacobian

        Inputs:
        - N: state vector dimension
        - M: basis order
        - τ_int: integration nodes
        - 
        =#

        # # Calculate the order of the spectral basis and the number of integration points
        # order, int_order = size(spectral_basis_gl_IVP)[1] - 1, size(spectral_basis_gl_IVP)[2]

        # TODO: Change from Gauss-Lobatto to Gauss-Chebyshev nodes.

        # Initialize the system matrix with zeros
        sys_matrix = zeros(Float64, (M + 1) * N, (M + 1) * N)

        # Calculate the system matrix
        for i in 1 : M + 1
            for j in 1 : M + 1
                for k in 1 : N
                    sys_matrix[(i - 1) * N + k, (j - 1) * N + k] = 
                        sum(τ_int .* T_GL[j, :] .* T_GL′[i, :]) - T_GL[i, end] .* T_GL[j, end]
                end
            end
        end

        # Return the system matrix
        return sys_matrix
    end

    function PRMatrix(N::Int, M::Int)
        #= Initializes the matrix P - R for the Newton iterations. 
        
        Inputs:
        - N: state vector dimension
        - M: order of the method

        Returns two N(M+1) × N(M+1) matrices, both formed by N×N blocks:
        - P: the (m,l) block is a diagonal matrix containing
        ∫ (dT_m/dτ) T_l dτ on the diagonal.
        - R: the (m,l) block is a diagonal matrix containing T_m(1)T_l(1) = 1
        on the diagonal.

        Basis functions are assumed to be Chebyshev polynomials everywhere.
        =#

        P = zeros(Float64, N*(M+1), N*(M+1))
        R = zeros(Float64, N*(M+1), N*(M+1))

        # Traverse columns first
        for l in 0 : M
            for m in 0 : M
                if m - l == 1
                    # This is the matrix P_ml as per the paper notation  
                    P[(m * N + 1): (m+1) * N, (l * N + 1): (l + 1) * N ] =
                    Diagonal((m * π/4.0) * ones(N))

                end
                
                # R_ml in the paper
                R[(m * N + 1): (m+1) * N, (l * N + 1): (l + 1) * N ] = Diagonal(ones(N))

            end
        end

        return P-R, P, R

    end

end

# function build_system_matrix_BVP(sys_size)

#     #=
#     This function builds the system matrix and optionally the Jacobian matrix
#     based on spectral basis, its derivative, Gauss-Legendre weights, Jacobian array, 
#     system size, and a boolean flag use_jacobian
#     =#

#     # Calculate the order of the spectral basis and the number of integration points
#     order, int_order = size(spectral_basis_gl_BVP)[1] - 1, size(spectral_basis_gl_BVP)[2]

#     # Initialize the system matrix and Jacobian matrix with zeros
#     sys_matrix = zeros(Float64, (order + 1) * sys_size, (order + 1) * sys_size)

#     # Calculate the system matrix
#     Threads.@threads for i in 1 : order + 1
#         for j in 1 : order + 1
#             for k in 1 : sys_size
#                 sys_matrix[(i - 1) * sys_size + k, (j - 1) * sys_size + k] = 
#                     sum(gl_weights_BVP .* spectral_basis_gl_BVP[j, :] .* spectral_basis_deriv2_gl_BVP[i, :]) + spectral_basis_deriv_gl_BVP[j, end] .* spectral_basis_gl_BVP[i, end] - spectral_basis_deriv_gl_BVP[j, 1] .* spectral_basis_gl_BVP[i, 1]
#             end
#         end
#     end

#     # Return the system matrix
#     return sys_matrix
# end

# function build_jacobian_matrix(Jacobian_spectrum, sys_size)
#     #=
#    This function builds the Jacobian matrix
#    based on spectral basis, its derivative, Gauss-Legendre weights, Jacobian array, 
#    system size, and a boolean flag use_jacobian
#    =#

#    # Calculate the order of the spectral basis and the number of integration points
#    order, int_order = size(Jacobian_spectrum)[1] - 1, size(Jacobian_spectrum)[2]

#    # Initialize the Jacobian matrix with zeros
#    jac_matrix = zeros(Float64, (order + 1) * sys_size, (order + 1) * sys_size)

#    # Calculate Jacobian matrix if use_jacobian is set to true
#    Threads.@threads for i in 1 : order + 1
#         for j in 1 : i

#             if i == j == 1
#                 jac_matrix[(j - 1) * sys_size + 1 : j * sys_size, (i - 1) * sys_size + 1 : i * sys_size] .= - pi * Jacobian_spectrum[1, :, :]
#             elseif i == j && i + j - 1 <= order + 1
#                 jac_matrix[(j - 1) * sys_size + 1 : j * sys_size, (i - 1) * sys_size + 1 : i * sys_size] .= - pi ./ 2 .* Jacobian_spectrum[1, :, :] .- pi / 4 .* Jacobian_spectrum[i + j - 1, :, :]
#             elseif i == j && i + j - 1 > order + 1
#                 jac_matrix[(j - 1) * sys_size + 1 : j * sys_size, (i - 1) * sys_size + 1 : i * sys_size] .= - pi ./ 2 .* Jacobian_spectrum[1, :, :]
#             elseif i != j && i + j - 1 <= order + 1
#                 jac_matrix[(j - 1) * sys_size + 1 : j * sys_size, (i - 1) * sys_size + 1 : i * sys_size] .= - pi ./ 4 .* (Jacobian_spectrum[abs(i - j + 1), :, :] .+ Jacobian_spectrum[i + j - 1, :, :])
#             else
#                 jac_matrix[(j - 1) * sys_size + 1 : j * sys_size, (i - 1) * sys_size + 1 : i * sys_size] .= - pi ./ 4 .* (Jacobian_spectrum[abs(i - j + 1), :, :])
#             end

#             jac_matrix[(i - 1) * sys_size + 1 : i * sys_size, (j - 1) * sys_size + 1 : j * sys_size] .=
#                     jac_matrix[(j - 1) * sys_size + 1 : j * sys_size, (i - 1) * sys_size + 1 : i * sys_size]
#         end
#     end
#    # Return the system matrix
#    return jac_matrix
# end

# function build_RHS_term_IVP(RHS_array, Jacobian_matrix, C, sys_size)
    
#     #=
#     This function builds the right-hand side (RHS) of a spectral system of equations. 
#     It uses spectral basis, Gauss-Legendre weights, RHS array, Jacobian array, solution array, 
#     system size, and a boolean flag to decide whether to use Jacobian or not.
#     It returns a vector "Source" which represents the RHS term of the system.
#     =#

#     # Calculate the order of the spectral basis and the number of integration points
#     order, int_order = size(spectral_basis_gl_IVP)[1] - 1, size(spectral_basis_gl_IVP)[2]

#     # Initialize the Source vector with zeros
#     Source = zeros(Float64, (order + 1) * sys_size)

#     # Calculate the Source vector
#     Threads.@threads for i in 1 : order + 1
#         for j in 1 : int_order
#             Source[(i - 1) * sys_size + 1: i * sys_size] .+= 
#             gl_weights_IVP[j] .* (- RHS_array[j, :] .* spectral_basis_gl_IVP[i, j])
#         end
#     end

    
#     Source .-= Jacobian_matrix * C

#     # Return the Source vector
#     return Source
# end

# function build_RHS_term_BVP(RHS_array, sys_size)
    
#     #=
#     This function builds the right-hand side (RHS) of a spectral system of equations. 
#     It uses spectral basis, Gauss-Legendre weights, RHS array, Jacobian array, solution array, 
#     system size, and a boolean flag to decide whether to use Jacobian or not.
#     It returns a vector "Source" which represents the RHS term of the system.
#     =#

#     # Calculate the order of the spectral basis and the number of integration points
#     order, int_order = size(spectral_basis_gl_BVP)[1] - 1, size(spectral_basis_gl_BVP)[2]

#     # Initialize the Source vector with zeros
#     Source = zeros(Float64, (order + 1) * sys_size)

#     # Calculate the Source vector
#     Threads.@threads for i in 1 : order + 1
#         for j in 1 : int_order
#             Source[(i - 1) * sys_size + 1: i * sys_size] .+= 
#             gl_weights_BVP[j] .* (RHS_array[j, :] .* spectral_basis_gl_BVP[i, j])
#         end
#     end

#     # Return the Source vector
#     return Source
# end

# function build_BCs_IVP(X0, sys_size)

#     #=
#     This function builds the boundary conditions (BCs) for a spectral system of equations.
#     It uses spectral basis, initial condition X0, and the system size.
#     It returns a vector "BC" which represents the boundary conditions of the system.
#     =#

#     # Calculate the order of the spectral basis
#     order = size(spectral_basis_gl_IVP)[1] - 1

#     # Initialize the BC vector with zeros
#     BC = zeros(Float64, (order + 1) * sys_size)

#     # Calculate the BC vector
#     for i in 1 : order + 1
#         BC[(i - 1) * sys_size + 1: i * sys_size] .= - spectral_basis_gl_IVP[i, 1] .* X0[:]
#     end

#     # Return the BC vector
#     return BC
# end

# function build_BCs_BVP(X1, X2, sys_size)

#     #=
#     This function builds the boundary conditions (BCs) for a spectral system of equations.
#     It uses spectral basis, initial condition X0, and the system size.
#     It returns a vector "BC" which represents the boundary conditions of the system.
#     =#

#     # Calculate the order of the spectral basis
#     order = size(spectral_basis_gl_BVP)[1] - 1

#     # Initialize the BC vector with zeros
#     BC = zeros(Float64, (order + 1) * sys_size)

#     # Calculate the BC vector
#     for i in 1 : order + 1
#         BC[(i - 1) * sys_size + 1: i * sys_size] .= spectral_basis_deriv_gl_BVP[i, end] .* X2[:] .- spectral_basis_deriv_gl_BVP[i, 1] .* X1[:]
#     end

#     # Return the BC vector
#     return BC
# end

# function build_RHS_array_IVP(solution, RHS, interval)

#     int_order = size(solution)[1]
#     sys_size = size(solution)[2]
#     t0, t1 = interval
#     RHS_array = zeros(Float64, int_order, sys_size)

#     Threads.@threads for i in 1 : int_order
#         RHS_array[i, :] .= 0.5 * (t1 - t0) * RHS(solution[i, :], [], (t1 - t0) / 2 .* gl_nodes_IVP[i] .+ (t1 + t0) / 2)
#     end

#     return RHS_array
# end

# function build_RHS_array_BVP(solution, RHS, interval)

#     int_order = size(solution)[1]
#     sys_size = size(solution)[2]
#     t0, t1 = interval
#     RHS_array = zeros(Float64, int_order, sys_size)

#     Threads.@threads for i in 1 : int_order
#         RHS_array[i, :] .= 0.25 * (t1 - t0) .^2 * RHS(solution[i, :], [], (t1 - t0) / 2 .* gl_nodes_BVP[i] .+ (t1 + t0) / 2)
#     end

#     return RHS_array
# end

# function build_Jacobian_array_IVP(solution, RHS, interval)

#     int_order, sys_size = size(solution)
#     Jacobian_array = zeros(Float64, int_order, sys_size, sys_size)
#     Threads.@threads for i in 1 : int_order
#         Jacobian_array[i, :, :] .= Jacobian_IVP(solution[i, :], RHS, interval, gc_nodes_IVP[i])
#     end
#     return Jacobian_array
# end

# function build_Jacobian_array_BVP(solution, RHS, interval)

#     int_order, sys_size = size(solution)
#     Jacobian_array = zeros(Float64, int_order, sys_size, sys_size)
#     Threads.@threads for i in 1 : int_order
#         Jacobian_array[i, :, :] .= Jacobian_BVP(solution[i, :], RHS, interval, gc_nodes_BVP[i])
#     end
#     return Jacobian_array
# end

# function print_progress(time_span, t0, t1, samples)

#     progress_now = (t1 - time_span[1]) / (time_span[2] - time_span[1]) * 100.0
#     progress_previous = (t0 - time_span[1]) / (time_span[2] - time_span[1]) * 100.0
#     progress_step = trunc(Int64, progress_now / (100.0 / samples)) * (100.0 / samples)

#     if progress_previous < progress_step
#         println("Solved ", round(progress_now, digits=0), " %")
#     end
# end

# function solve_step(jac_mat, source, sys_size, tolerance)

#     order = size(spectral_basis_gl_IVP)[1] - 1
#     error = Inf
#     C_new = zeros(Float64, (order + 1) * sys_size)
#     C_term = zeros(Float64, (order + 1) * sys_size)
#     C_term2 = zeros(Float64, (order + 1) * sys_size)

#     mul!(C_term, sys_matrix_IVP_inv, source)

#     while error > tolerance
#         C_new .+= C_term
#         mul!(C_term2, jac_mat, C_term)
#         mul!(C_term, sys_matrix_IVP_inv, C_term2)
#         error = norm(C_term)
#     end

#     return C_new
# end

# function solve_interval_IVP(RHS, C_initial, X0, interval, sys_size, max_iter, tolerance)
#     C = copy(C_initial)
#     B = build_BCs_IVP(X0, sys_size)

#     it_array = Vector{Int}(undef, max_iter)
#     error_array = Vector{Float64}(undef, max_iter)
    
#     error = Inf
#     for it in 1:max_iter
#         # Evaluate solution at GL and GC nodes
#         solution_array = build_solution(C, spectral_basis_gl_IVP, sys_size) # spectral_basis_gl_IVP: Chebyshev basis evaluated on GL nodes
#         solution_array_gc = build_solution(C, spectral_basis_gc_IVP, sys_size) # spectral_basis_gc_IVP: Chebyshev basis evaluated on GC nodes

#         # Evaluate RHS on GL nodes and Jacobian on GC nodes
#         RHS_array = build_RHS_array_IVP(solution_array, RHS, interval)
#         Jacobian_array_gc = build_Jacobian_array_IVP(solution_array_gc, RHS, interval)

#         # Assemble Jacobian matrix
#         JS = build_Jacobian_spectrum(Jacobian_array_gc, sys_size, "IVP")
#         Jacobian_matrix = build_jacobian_matrix(JS, sys_size)

#         # S = source + Jac*coefs (see eq. 16)
#         S = build_RHS_term_IVP(RHS_array, Jacobian_matrix, C, sys_size)
#         #@time C_new = solve_step(Jacobian_matrix, S .+ B, sys_size, 2e-16)
#         C_new = (sys_matrix_IVP .- Jacobian_matrix)\(S .+ B)
#         error = norm(C_new .- C) / norm(C)
#         it_array[it] = it
#         error_array[it] = error
#         #println("The error is ", error)
#         if error < tolerance
#             #println("Interval solved in ", it, " iterations!")
#             resize!(it_array, it)
#             resize!(error_array, it)
#             return C_new, it_array, error_array
#         end

#         C .= C_new
#     end

#     return C, it_array, error_array
# end

# function solve_interval_BVP(RHS, C_initial, X1, X2, interval, sys_size, max_iter, tolerance)
#     C = copy(C_initial)
#     B = build_BCs_BVP(X1, X2, sys_size)

#     it_array = Vector{Int}(undef, max_iter)
#     error_array = Vector{Float64}(undef, max_iter)
    
#     error = Inf
#     for it in 1:max_iter
#         solution_array = build_solution(C, spectral_basis_gl_BVP, sys_size)
#         solution_array_gc = build_solution(C, spectral_basis_gc_BVP, sys_size)
#         RHS_array = build_RHS_array_BVP(solution_array, RHS, interval)
#         Jacobian_array_gc = build_Jacobian_array_BVP(solution_array_gc, RHS, interval)
#         JS = build_Jacobian_spectrum(Jacobian_array_gc, sys_size, "BVP")
#         Jacobian_matrix = build_jacobian_matrix(JS,  sys_size)
#         S = build_RHS_term_BVP(RHS_array, sys_size)
#         #C_new = solve_step(Jacobian_matrix, S .+ B, sys_size, 2e-16)
#         C_new = (sys_matrix_BVP .+ Jacobian_matrix)\(B .+ S .+ Jacobian_matrix * C)
#         error = norm(C_new .- C) / norm(C)
#         it_array[it] = it
#         error_array[it] = error
#         #println("The error is ", error)
#         if error < tolerance
#             println("Interval solved in ", it, " iterations!")
#             resize!(it_array, it)
#             resize!(error_array, it)
#             return C_new, it_array, error_array
#         end

#         C .= C_new
#     end

#     return C, it_array, error_array
# end

# function ssolve_IVP(RHS, RHS_simple, X0, time_span, interval, order, relative_tolerance, iterative_tolerance, max_iter, h_adaptive)

#     SF = 0.9
#     sys_size = size(X0)[1]
#     solution_coefficients = [];
#     solution_time = [];
#     iteration_matrix = [];
#     error_matrix = [];
#     current_time = time_span[1]
#     step = 0
#     samples = 10
#     epsilon = 1e-8
    
#     reference_interval = interval
    
#     while current_time < time_span[2]

#         if step < 1
#             t0, t1 = time_span[1], time_span[1] + reference_interval
#         else
#             t0, t1 = solution_time[end][2], solution_time[end][2] + reference_interval
#         end
#         current_time = t0;
#         if current_time >= time_span[2] || abs(current_time - time_span[2]) < epsilon
#             break
#         end
#         coarse_problem = ODEProblem(RHS_simple, X0, (t0, t1));
#         sol = DifferentialEquations.solve(coarse_problem, Tsit5(), reltol=1e-5, abstol=1e-5, dtmax=(t1 - t0) / 5);
#         time_coarse = sol.t
#         sol_coarse = mapreduce(permutedims, vcat, sol.u)
#         C_initial = build_spectrum(time_coarse, sol_coarse);
#         if h_adaptive
#             rel_error = build_spectral_error(time_coarse, sol_coarse, C_initial, sys_size)
#             t1 = t0 + (t1 - t0) * (relative_tolerance / rel_error) ^ (1 / SF / min(order, -log10(iterative_tolerance)))
#             t1 = min(t1, time_span[2])
#             coarse_problem = ODEProblem(RHS_simple, X0, (t0, t1));
#             sol = DifferentialEquations.solve(coarse_problem, Tsit5(), reltol=1e-5, abstol=1e-5, dtmax=(t1 - t0) / 5);
#             time_coarse = sol.t
#             sol_coarse = mapreduce(permutedims, vcat, sol.u)
#             C_initial = build_spectrum(time_coarse, sol_coarse);
#         end
#         t1 = min(t1, time_span[2])
#         C, it, error = solve_interval_IVP(RHS, C_initial, X0, (t0, t1), sys_size, max_iter, iterative_tolerance);
#         push!(solution_time, [t0, t1]);
#         push!(solution_coefficients, C);
#         push!(iteration_matrix, it);
#         push!(error_matrix, error);
#         X0 = zeros(Float64, sys_size);
#         for j in 1 : order + 1
#             X0 .+= C[(j - 1) * sys_size + 1 : j * sys_size] .* spectral_basis_gl_IVP[j, end]
#         end
#         step += 1;
#         print_progress(time_span, t0, t1, samples);
#     end
#     solution_time = reduce(hcat, solution_time)'
#     solution_coefficients = reduce(hcat, solution_coefficients)'

#     return solution_time, solution_coefficients, iteration_matrix, error_matrix
    
# end

# function ssolve_BVP(RHS, X1, X2, time_span, order, iterative_tolerance, max_iter)

#     sys_size = size(X1)[1]
#     solution_coefficients = [];
#     solution_time = [];
#     iteration_matrix = [];
#     error_matrix = [];
    
#     t0, t1 = time_span[1], time_span[2]
#     C_initial = rand(Uniform(0.1,1.1),(order + 1) * sys_size);
#     C, it, error = solve_interval_BVP(RHS, C_initial, X1, X2, (t0, t1), sys_size, max_iter, iterative_tolerance);

#     push!(solution_time, [t0, t1]);
#     push!(solution_coefficients, C);
#     push!(iteration_matrix, it);
#     push!(error_matrix, error);
    
#     solution_time = reduce(hcat, solution_time)'
#     solution_coefficients = reduce(hcat, solution_coefficients)'

#     return solution_time, solution_coefficients, iteration_matrix, error_matrix
    
# end
