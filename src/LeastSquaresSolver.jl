using FastGaussQuadrature, DifferentialEquations, BasicInterpolators, ForwardDiff, IterativeSolvers, Preconditioners


function build_system_matrix_LS(spectral_basis, spectral_basis_deriv, gl_weights, sys_size)

    #=
    This function builds the system matrix and optionally the Jacobian matrix
    based on spectral basis, its derivative, Gauss-Legendre weights, Jacobian array, 
    system size, and a boolean flag use_jacobian
    =#

    # Calculate the order of the spectral basis and the number of integration points
    order, int_order = size(spectral_basis)[1] - 1, size(spectral_basis)[2]

    # Initialize the system matrix and Jacobian matrix with zeros
    sys_matrix = zeros(Float64, (order + 1) * sys_size, (order + 1) * sys_size)

    # Calculate the system matrix
    Threads.@threads for i in 1 : order + 1
        for j in 1 : order + 1
            for k in 1 : sys_size
                sys_matrix[(i - 1) * sys_size + k, (j - 1) * sys_size + k] = sum(gl_weights .* spectral_basis_deriv[j, :] .* spectral_basis_deriv[i, :])
            end
        end
    end

    # Impose boundary conditions
    for i in 1 : order + 1
        for k in 1 : sys_size
            sys_matrix[k, (i - 1) * sys_size + k] = spectral_basis[i, 1]
        end
    end

    # Return the system matrix
    return sys_matrix
end

function build_jacobian_matrix_LS(spectral_basis, spectral_basis_deriv, gl_weights, Jacobian_array, sys_size)
     #=
    This function builds the Jacobian matrix
    based on spectral basis, its derivative, Gauss-Legendre weights, Jacobian array, 
    system size, and a boolean flag use_jacobian
    =#

    # Calculate the order of the spectral basis and the number of integration points
    order, int_order = size(spectral_basis)[1] - 1, size(spectral_basis)[2]

    # Initialize the Jacobian matrix with zeros
    jac_matrix = zeros(Float64, (order + 1) * sys_size, (order + 1) * sys_size)

    # Calculate Jacobian matrix if use_jacobian is set to true
    Threads.@threads for i in 1 : order + 1
        for j in 1 : order + 1
            for k in 1 : int_order
                jac_matrix[(i - 1) * sys_size + 1 : i * sys_size, (j - 1) * sys_size + 1 : j * sys_size] .+= 
                    (gl_weights[k] .* Jacobian_array[k , :, :] .* spectral_basis_deriv[i, k] .* spectral_basis[j, k])
            end
        end
    end

    # Return the system matrix
    return jac_matrix
end

function build_RHS_term_LS(spectral_basis, spectral_basis_deriv, gl_weights, RHS_array, Jacobian_array, Solution_array, X0, sys_size)
    
    #=
    This function builds the right-hand side (RHS) of a spectral system of equations. 
    It uses spectral basis, Gauss-Legendre weights, RHS array, Jacobian array, solution array, 
    system size, and a boolean flag to decide whether to use Jacobian or not.
    It returns a vector "Source" which represents the RHS term of the system.
    =#

    # Calculate the order of the spectral basis and the number of integration points
    order, int_order = size(spectral_basis_deriv)[1] - 1, size(spectral_basis_deriv)[2]

    # Initialize the Source vector with zeros
    Source = zeros(Float64, (order + 1) * sys_size)

    # Calculate the Source vector
    for i in 2 : order + 1
        for j in 1 : int_order
            Source[(i - 1) * sys_size + 1: i * sys_size] .+= 
                    gl_weights[j] * (RHS_array[j, :] .* spectral_basis_deriv[i, j] .-
                    Jacobian_array[j, :, :] * Solution_array[j, :] .* spectral_basis_deriv[i, j])
        end
    end

    # Impose the initial value condition
    Source[1 : sys_size] = X0

    # Return the Source vector
    return Source
end

function build_RHS_array_LS(solution, RHS, gl_nodes, interval)

    int_order = size(solution)[1]
    sys_size = size(solution)[2]
    t0, t1 = interval
    RHS_array = zeros(Float64, int_order, sys_size)

    Threads.@threads for i in 1 : int_order
        RHS_array[i, :] .= 0.5 * (t1 - t0) * RHS(solution[i, :], [], (t1 - t0) / 2 .* gl_nodes[i] .+ (t1 + t0) / 2)
    end

    return RHS_array
end

function build_Jacobian_array_LS(solution, RHS, gl_nodes, interval)

    int_order, sys_size = size(solution)
    Jacobian_array = zeros(Float64, int_order, sys_size, sys_size)
    Threads.@threads for i in 1 : int_order
        Jacobian_array[i, :, :] .= Jacobian_automatic(solution[i, :], RHS, interval, gl_nodes[i])
    end

    return Jacobian_array
end

function print_progress_LS(time_span, t0, t1, samples)

    progress_now = (t1 - time_span[1]) / (time_span[2] - time_span[1]) * 100.0
    progress_previous = (t0 - time_span[1]) / (time_span[2] - time_span[1]) * 100.0
    progress_step = trunc(Int64, progress_now / (100.0 / samples)) * (100.0 / samples)

    if progress_previous < progress_step
        println("Solved ", round(progress_now, digits=0), " %")
    end
end

function solve_step_LS(sys_mat_inv, jac_mat, source, spectral_basis, sys_size, tolerance)

    order = size(spectral_basis)[1] - 1
    error = Inf
    C_new = zeros(Float64, (order + 1) * sys_size)
    C_term = zeros(Float64, (order + 1) * sys_size)
    C_term2 = zeros(Float64, (order + 1) * sys_size)

    mul!(C_term, sys_mat_inv, source)

    while error > tolerance
        C_new .+= C_term
        mul!(C_term, sys_mat_inv, mul!(C_term2, jac_mat, C_term))
        error = norm(C_term) / norm(C_new)
    end

    return C_new
end

function solve_interval_LS(RHS, C_initial, X0, interval, gl_nodes, gl_weights, spectral_basis, spectral_basis_deriv, spectral_matrix_inv, sys_size, max_iter, tolerance, type)
    C = copy(C_initial)

    it_array = Vector{Int}(undef, max_iter)
    error_array = Vector{Float64}(undef, max_iter)
    
    error = Inf
    for it in 1:max_iter
        solution_array = build_solution(C, spectral_basis, sys_size)
        RHS_array = build_RHS_array(solution_array, RHS, gl_nodes, interval)
        Jacobian_array = build_Jacobian_array_LS(solution_array, RHS, gl_nodes, interval)
        M = spectral_matrix_inv
        A = build_jacobian_matrix_LS(spectral_basis, spectral_basis_deriv, gl_weights, Jacobian_array, sys_size)
        S = build_RHS_term_LS(spectral_basis, spectral_basis_deriv, gl_weights, RHS_array, Jacobian_array, solution_array, X0, sys_size)
        C_new = copy(C)
        C_new = solve_step_LS(M, A, S, X0, spectral_basis, sys_size)
        error = norm((C_new .- C) / C)
        it_array[it] = it
        error_array[it] = error
        println("The error is ", error)
        if error < tolerance
            println("Solution converged in ", it, " iterations!")
            resize!(it_array, it)
            resize!(error_array, it)
            return C_new, it_array, error_array
        end

        C .= C_new
    end

    return C, it_array, error_array
end

function ssolve_LS(RHS, RHS_simple, X0, time_span, interval, order, relative_tolerance, iterative_tolerance, max_iter, type, h_adaptive)

    SF = 0.9
    sys_size = size(X0)[1]
    gl_nodes, gl_weights = build_quadrature(order);
    spectral_basis, spectral_basis_deriv = create_basis_set(gl_nodes, order, type);
    spectral_matrix = build_system_matrix_LS(spectral_basis, spectral_basis_deriv, gl_weights, sys_size)
    spectral_matrix_inv = inv(spectral_matrix)
    solution_coefficients = [];
    solution_time = [];
    iteration_matrix = [];
    error_matrix = [];
    current_time = time_span[1]
    step = 0
    samples = 10
    
    reference_interval = interval

    while current_time < time_span[2]

        if step < 1
            t0, t1 = time_span[1], time_span[1] + reference_interval
        else
            t0, t1 = solution_time[end][2], solution_time[end][2] + reference_interval
        end
        current_time = t0;
        if current_time >= time_span[2]
            break
        end
        coarse_problem = ODEProblem(RHS_simple, X0, (t0, t1));
        sol = DifferentialEquations.solve(coarse_problem, Tsit5(), reltol=1e-8, abstol=1e-8, dtmax=(t1 - t0) / 5);
        time_coarse = sol.t
        sol_coarse = mapreduce(permutedims, vcat, sol.u)
        C_initial = build_spectrum(time_coarse, sol_coarse, gl_nodes, gl_weights, spectral_basis, type);
        if h_adaptive
            rel_error = build_spectral_error(time_coarse, sol_coarse, C_initial, sys_size, type)
            t1 = t0 + (t1 - t0) * (relative_tolerance / rel_error) ^ (1 / SF / min(order, -log10(iterative_tolerance)))
            t1 = min(t1, time_span[2])
            coarse_problem = ODEProblem(RHS_simple, X0, (t0, t1));
            sol = DifferentialEquations.solve(coarse_problem, Tsit5(), reltol=1e-8, abstol=1e-8, dtmax=(t1 - t0) / 5);
            time_coarse = sol.t
            sol_coarse = mapreduce(permutedims, vcat, sol.u)
            C_initial = build_spectrum(time_coarse, sol_coarse, gl_nodes, gl_weights, spectral_basis, type);
        end
        t1 = min(t1, time_span[2])
        C, it, error = solve_interval_LS(RHS, C_initial, X0, (t0, t1), gl_nodes, gl_weights, spectral_basis, 
        spectral_basis_deriv, spectral_matrix_inv, sys_size, max_iter, iterative_tolerance, type);
        push!(solution_time, [t0, t1]);
        push!(solution_coefficients, C);
        push!(iteration_matrix, it);
        push!(error_matrix, error);
        X0 = zeros(Float64, sys_size);
        for j in 1 : order + 1
            X0 .+= C[(j - 1) * sys_size + 1 : j * sys_size] .* spectral_basis[j, end]
        end
        step += 1;
        print_progress(time_span, t0, t1, samples);
    end
    solution_time = reduce(hcat, solution_time)'
    solution_coefficients = reduce(hcat, solution_coefficients)'

    return solution_time, solution_coefficients, iteration_matrix, error_matrix
    
end