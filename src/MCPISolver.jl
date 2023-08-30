

function build_spectrum_MCPI(time, solution, spectral_basis, gc_nodes, order, sys_size)

    #=
    This function builds the spectral representation of a solution of a system
    depending on the type of spectral basis chosen ("Legendre" or "Chebyshev").
    =#
    
    # Normalize time variable tau in [-1, 1] interval.
    t0, t1 = time[1], time[end]
    tau = 2 ./ (t1 - t0) .* (time .- (t1 .+ t0) ./ 2)
    tau[1] = -1.0
    tau[end] = 1.0
    
    # Initialize the coefficients matrix.
    C = zeros(Float64, order + 1, sys_size)

    # Transpose the spectral basis matrix.
    Mat = spectral_basis'
    # For each variable in the system.
    for j in 1 : sys_size
        # Interpolate the solution using cubic interpolation.
        solution_interpolated = CubicInterpolator(tau, solution[:, j])
        # Compute the solution at gauss-legendre nodes.
        solution_points = solution_interpolated.(gc_nodes)
        # Compute coefficients for the current variable.
        C[:, j] = Mat \ solution_points
    end
    
    # Reshape the coefficients matrix into a single column vector.
    C = reshape(C', (order + 1) * sys_size)
    
    # Return the computed coefficients.
    return C
end

function post_process_MCPI(time, coefficients, sys_size, samples, type)

    #=
    This function post-prcoesses the coefficient and time arrays outputed by the spectral solve_interval
    into a time and solution set of arrays of an arbitrary number of samples

    Parameters:
    - time : 2D array of the edge points of each time interval
    - coeffificents : the 3D array containing the coefficients of the expansion for each time interval
    - sys_size : the number of degrees of freedom of the ODE system considered
    - samples : (Int) the number of samples for the output arrays
    =#

    N_intervals, n_coeffs = size(coefficients)
    order = trunc(Int, n_coeffs / sys_size) - 1
    tau = LinRange(-1, 1, samples)
    spectral_basis_local, _ = create_basis_set(tau, order, "Chebyshev")

    # Initialize time_array and solution_array with appropriate types and sizes
    time_array = Float64[]
    solution_array = zeros(Float64, 0, sys_size)

    for i in 1:N_intervals
        t = LinRange(time[i, 1], time[i, 2], samples)
        C = reshape(coefficients[i, :], (sys_size, order + 1))'
        sol = zeros(Float64, samples, sys_size)

        for j in 1:order + 1
            for k in 1:sys_size
                sol[:, k] .+= C[j, k] .* spectral_basis_local[j, :]
            end
        end

        # Concatenate solution_array and new solution, leaving out the last row
        solution_array = vcat(solution_array, sol[1:end - 1, :])

        # Append new time points, leaving out the last one
        append!(time_array, t[1:end - 1])
    end
    if type == "Equinoctial"
        solution_array = equinoctial_to_kepler_array(solution_array)
    elseif type == "Cowell"
        solution_array = cartesian_to_kepler_array(solution_array)
    end

    return time_array, solution_array
end

function build_solution_MCPI(C, spectral_basis, sys_size)

    order, int_order = size(spectral_basis)[1] - 1, size(spectral_basis)[2]
    Solution = zeros(Float64, int_order, sys_size)

    for i in order + 1 : -1 : 1
        for k in 1 : sys_size
            Solution[:, k] .+= C[(i - 1) .* sys_size + k] .* spectral_basis[i, :]
        end
    end

    return Solution
end

function build_RHS_array_MCPI_IVP(solution, RHS, interval, gc_nodes)

    int_order = size(solution)[1]
    sys_size = size(solution)[2]
    t0, t1 = interval
    RHS_array = zeros(Float64, int_order * sys_size)

    Threads.@threads for i in 1 : int_order
        RHS_array[(i - 1) * sys_size + 1 : i * sys_size] .= 0.5 * (t1 - t0) * RHS(solution[i, :], [], (t1 - t0) / 2 .* gc_nodes[i] .+ (t1 + t0) / 2)
    end

    return RHS_array
end

function build_RHS_array_MCPI_BVP(solution, RHS, interval, gc_nodes)

    int_order = size(solution)[1]
    sys_size = size(solution)[2]
    t0, t1 = interval
    RHS_array = zeros(Float64, int_order * sys_size)

    Threads.@threads for i in 1 : int_order
        RHS_array[(i - 1) * sys_size + 1 : i * sys_size] .= 0.25 * (t1 - t0).^2 * RHS(solution[i, :], [], (t1 - t0) / 2 .* gc_nodes[i] .+ (t1 + t0) / 2)
    end

    return RHS_array
end

function build_system_matrix_MCPI(order, sys_size, gc_nodes)

    c_basis, c_basis_deriv = create_basis_set(gc_nodes, order, "Chebyshev")
    T = zeros((order + 1) * sys_size, (order + 1) * sys_size)
    T_t = zeros((order + 1) * sys_size, (order + 1) * sys_size)
    R = zeros((order + 1) * sys_size, (order + 1) * sys_size)
    S = zeros((order + 1) * sys_size, (order + 1) * sys_size)
    V = zeros((order + 1) * sys_size, (order + 1) * sys_size)
    W = zeros((order + 1) * sys_size, (order + 1) * sys_size)

    for i in 1 : order + 1
        for j in 1 : order + 1
            for k in 1 : sys_size
                T[(i - 1) * sys_size + k, (j - 1) * sys_size + k] = c_basis[j, i]
                T_t[(i - 1) * sys_size + k, (j - 1) * sys_size + k] = c_basis[i, j]
            end
        end
    end

    for i in 1 : order + 1
        for k in 1 : sys_size
            if i == 1
                R[(i - 1) * sys_size + k, (i - 1) * sys_size + k] = 1.0
            else
                R[(i - 1) * sys_size + k, (i - 1) * sys_size + k] = 1 / (2 * (i - 1))
            end
            if i == 1 || i == order + 1
                V[(i - 1) * sys_size + k, (i - 1) * sys_size + k] = 1 / (order)
            else
                V[(i - 1) * sys_size + k, (i - 1) * sys_size + k] = 2 / (order)
            end
            if i == 1
                S[k, (i - 1) * sys_size + k] = 0.5
            elseif i == 2
                S[k, (i - 1) * sys_size + k] = -0.25
            else
                S[k, (i - 1) * sys_size + k] = (-1)^(i) * (1 / (i - 2) - 1 / (i)) / 2
            end
            if i > 1
                S[(i - 1) * sys_size + k, (i - 2) * sys_size + k] = 1.0
            end
            if i < order + 1
                S[(i - 1) * sys_size + k, (i) * sys_size + k] = -1.0
            end
        end
    end
    C_alpha = R * S * T_t * V
    Cx = T

    return C_alpha, Cx
end

function build_BCs_MCPI(X0, order, sys_size)

    BC = zeros(Float64, (order + 1) * sys_size)
    BC[1 : sys_size] .= X0

    return BC
end

function solve_interval_MCPI_IVP(RHS, C_initial, X0, interval, sys_size, max_iter, tolerance)

    order = trunc(Int64, size(C_initial)[1] / sys_size) - 1
    gc_nodes, gc_weights = build_quadrature(order, "Chebyshev_L")
    c_basis, c_basis_deriv = create_basis_set(gc_nodes, order, "Chebyshev")
    C = copy(C_initial)
    B = build_BCs_MCPI(X0, order, sys_size)
    M_alpha, M_x = build_system_matrix_MCPI(order, sys_size, gc_nodes)

    it_array = Vector{Int}(undef, max_iter)
    error_array = Vector{Float64}(undef, max_iter)
    
    error = Inf
    for it in 1:max_iter
        @time solution_array = build_solution_MCPI(C, c_basis, sys_size)
        @time RHS_array = build_RHS_array_MCPI_IVP(solution_array, RHS, interval, gc_nodes)
        @time C_new = B .+ M_alpha * RHS_array
        @time error = norm(C_new .- C) / norm(C)
        it_array[it] = it
        error_array[it] = error
        println("The error is ", error)
        if error < tolerance
            println("Interval solved in ", it, " iterations!")
            resize!(it_array, it)
            resize!(error_array, it)
            return C_new, it_array, error_array
        end

        C .= C_new
    end
    println("The MCPI method did not converge in ", max_iter, " iterations!")
    return C, it_array, error_array
end

function solve_interval_MCPI_BVP(RHS, C_initial, X1, X2, interval, sys_size, max_iter, tolerance)

    order = trunc(Int64, size(C_initial)[1] / sys_size) - 1
    gc_nodes, gc_weights = build_quadrature(order, "Chebyshev_L")
    c_basis, c_basis_deriv = create_basis_set(gc_nodes, order, "Chebyshev")
    C = copy(C_initial)
    M_alpha, M_x = build_system_matrix_MCPI(order, sys_size, gc_nodes)

    it_array = Vector{Int}(undef, max_iter)
    error_array = Vector{Float64}(undef, max_iter)
    
    error = Inf
    for it in 1:max_iter
        solution_array = build_solution_MCPI(C, c_basis, sys_size)
        RHS_array = build_RHS_array_MCPI_BVP(solution_array, RHS, interval, gc_nodes)
        C_new = M_alpha * M_x * M_alpha * RHS_array
        #display(C_new)
        C_new[1 : sys_size] .= (X1 .+ X2) .* 0.5
        C_new[sys_size + 1 : 2 * sys_size] .= (X2 .- X1) .* 0.5
        for i in 1 : sys_size
            for j in 3 : order + 1
                if j % 2 == 1
                    C_new[i] -= C_new[(j - 1) * sys_size + i]
                else
                    C_new[sys_size + i] -= C_new[(j - 1) * sys_size + i]
                end
            end
        end
        error = norm(C_new .- C) / norm(C)
        it_array[it] = it
        error_array[it] = error
        #println("The error is ", error)
        if error < tolerance
            println("Interval solved in ", it, " iterations!")
            resize!(it_array, it)
            resize!(error_array, it)
            return C_new, it_array, error_array
        end

        C .= C_new
    end
    println("The MCPI method did not converge in ", max_iter, " iterations!")
    return C, it_array, error_array
end

function ssolve_MCPI_IVP(RHS, RHS_simple, X0, time_span, interval, order, relative_tolerance, iterative_tolerance, max_iter, h_adaptive)

    SF = 0.9
    sys_size = size(X0)[1]
    solution_coefficients = [];
    solution_time = [];
    iteration_matrix = [];
    error_matrix = [];
    current_time = time_span[1]
    step = 0
    samples = 10
    epsilon = 1e-8
    
    reference_interval = interval

    gc_nodes, gc_weights = build_quadrature(order, "Chebyshev_L")
    c_basis, c_basis_deriv = create_basis_set(gc_nodes, order, "Chebyshev")
    
    while current_time < time_span[2]

        if step < 1
            t0, t1 = time_span[1], time_span[1] + reference_interval
        else
            t0, t1 = solution_time[end][2], solution_time[end][2] + reference_interval
        end
        current_time = t0;
        if current_time >= time_span[2] || abs(current_time - time_span[2]) < epsilon
            break
        end
        coarse_problem = ODEProblem(RHS_simple, X0, (t0, t1));
        sol = DifferentialEquations.solve(coarse_problem, Tsit5(), reltol=1e-5, abstol=1e-5, dtmax=(t1 - t0) / 5);
        time_coarse = sol.t
        sol_coarse = mapreduce(permutedims, vcat, sol.u)
        C_initial = build_spectrum_MCPI(time_coarse, sol_coarse, c_basis, gc_nodes, order, sys_size);
        t1 = min(t1, time_span[2])
        C, it, error = solve_interval_MCPI_IVP(RHS, C_initial, X0, (t0, t1), sys_size, max_iter, iterative_tolerance);
        push!(solution_time, [t0, t1]);
        push!(solution_coefficients, C);
        push!(iteration_matrix, it);
        push!(error_matrix, error);
        X0 = zeros(Float64, sys_size);
        for j in 1 : order + 1
            X0 .+= C[(j - 1) * sys_size + 1 : j * sys_size] .* c_basis[j, end]
        end
        step += 1;
        print_progress(time_span, t0, t1, samples);
    end
    solution_time = reduce(hcat, solution_time)'
    solution_coefficients = reduce(hcat, solution_coefficients)'

    return solution_time, solution_coefficients, iteration_matrix, error_matrix
    
end

function ssolve_MCPI_BVP(RHS, X1, X2, time_span, order, iterative_tolerance, max_iter)

    sys_size = size(X1)[1]
    solution_coefficients = [];
    solution_time = [];
    iteration_matrix = [];
    error_matrix = [];
    
    t0, t1 = time_span[1], time_span[2]
    C_initial = rand(Uniform(0.1,1.1),(order + 1) * sys_size);
    C, it, error = solve_interval_MCPI_BVP(RHS, C_initial, X1, X2, (t0, t1), sys_size, max_iter, iterative_tolerance);

    push!(solution_time, [t0, t1]);
    push!(solution_coefficients, C);
    push!(iteration_matrix, it);
    push!(error_matrix, error);
    
    solution_time = reduce(hcat, solution_time)'
    solution_coefficients = reduce(hcat, solution_coefficients)'

    return solution_time, solution_coefficients, iteration_matrix, error_matrix
    
end