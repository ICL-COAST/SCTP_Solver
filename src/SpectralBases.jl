using Plots

function build_spectrum(time, solution)

    #=
    This function builds the spectral representation of a solution of a system
    depending on the type of spectral basis chosen ("Legendre" or "Chebyshev").
    =#

    type = "Chebyshev"

    # Compute order of the spectral basis and size of the system.
    order, sys_size = size(spectral_basis_gl_IVP, 1) - 1, size(solution, 2)
    
    # Normalize time variable tau in [-1, 1] interval.
    t0, t1 = time[1], time[end]
    tau = 2 ./ (t1 - t0) .* (time .- (t1 .+ t0) ./ 2)
    tau[1] = -1.0
    tau[end] = 1.0
    
    # Initialize the coefficients matrix.
    C = zeros(Float64, order + 1, sys_size)
    
    # If spectral basis is Legendre.
    if type == "Legendre"
        # For each variable in the system.
        for j in 1 : sys_size
            # Interpolate the solution using cubic interpolation.
            solution_interpolated = CubicInterpolator(tau, solution[:, j])
            # Compute the solution at gauss-legendre nodes.
            solution_points = solution_interpolated.(gl_nodes_IVP)
            # For each order.
            for i in 1 : order + 1
                # Compute coefficients for the current variable and order.
                C[i, j] = sum(gl_weights_IVP .* solution_points .* spectral_basis_gl_IVP[i, :]) .* (2 * i - 1) / 2
            end
        end
    elseif type == "Chebyshev"  # If spectral basis is Chebyshev.
        # For each variable in the system.
        for j in 1 : sys_size
            # Interpolate the solution using cubic interpolation.
            solution_interpolated = CubicInterpolator(tau, solution[:, j])
            # Compute the solution at gauss-legendre nodes.
            solution_points = solution_interpolated.(gc_nodes_IVP)
            # For each order.
            for i in 1 : order + 1
                # Compute coefficients for the current variable and order.
                C[i, j] = sum(gc_weights_IVP .* solution_points .* spectral_basis_gc_IVP[i, :]) .* 2 ./ pi
                if i == 1
                    C[i, j] /= 2
                end
            end
        end
    end
    
    # Reshape the coefficients matrix into a single column vector.
    C = reshape(C', (order + 1) * sys_size)
    
    # Return the computed coefficients.
    return C
end

function build_Jacobian_spectrum(Jacobian_array, sys_size, problem_type)

    if problem_type == "IVP"
        order = size(spectral_basis_gc_IVP)[1] - 1
        int_order = size(spectral_basis_gc_IVP)[2] - 1
        Jacobian_spectrum = zeros(order + 1, sys_size, sys_size)
        bases_local = spectral_basis_gc_IVP
        nodes_local = gc_nodes_IVP
        weights_local = gc_weights_IVP
    else
        order = size(spectral_basis_gc_BVP)[1] - 1
        Jacobian_spectrum = zeros(order + 1, sys_size, sys_size)
        bases_local = spectral_basis_gc_BVP
        nodes_local = gc_nodes_BVP
        weights_local = gc_weights_BVP
    end

    for i in 1 : sys_size
        for j in 1 : sys_size
            for k in 1 : order + 1
                Jacobian_spectrum[k, i, j] = sum(weights_local .* Jacobian_array[:, i, j] .* bases_local[k, :] .* sqrt.(1 .- nodes_local.^2)) .* 2 ./ pi
                if k == 1
                    Jacobian_spectrum[k, i, j] /= 2
                end
            end
        end
    end
    return Jacobian_spectrum
end

function build_solution(C, spectral_basis, sys_size)

    order, int_order = size(spectral_basis)[1] - 1, size(spectral_basis)[2]
    Solution = zeros(Float64, int_order, sys_size) #int order

    for i in order + 1 : -1 : 1 #order
        for k in 1 : sys_size
            Solution[:, k] .+= C[(i - 1) .* sys_size + k] .* spectral_basis[i, :]
        end
    end

    return Solution
end

function build_spectral_error(time, solution, C, sys_size)

    order = trunc(Int64, size(C)[1] / sys_size) - 1
    int_order = 3 * order
    t0, t1 = time[1], time[end]
    tau_real = 2 / (t1 - t0) .* (time .- (t1 + t0) / 2)
    tau_real[1] = -1.0
    tau_real[end] = 1.0
    rel_error_array = zeros(int_order + 1, sys_size)
    total_error = 0.0

    gl_nodes_local, gl_weights_local = build_quadrature(int_order, "Chebyshev")
    spectral_basis_local, _, _ = create_basis_set(gl_nodes_local, order, "Chebyshev")

    tau_spectral = gl_nodes_local

    solution_spectral = build_solution(C, spectral_basis_local, sys_size)

    for i in 1 : sys_size
        solution_interpolated = CubicInterpolator(tau_real, solution[:, i])
        solution_real = solution_interpolated.(tau_spectral)
        rel_error_array[:, i] .= (solution_spectral[:, i] .- solution_real)
    end

    for i in 1 : int_order + 1
        total_error += norm(rel_error_array[i]) * gl_weights_local[i]
    end
    return total_error
end

function create_basis_set(tau, order, type)

    #=
    This function computes the first [order + 1] basis functions of the Legendre polynomial basis, as well as their derivatives
    at the time instances [tau].
    
    Parameters:
    - tau : 1D array of scaled time instances [-1, 1], nodes initialised from build_quadrature
    - order : the maximum order of the polynomial expansions
    - type : to be implemented (function will have multiple choices of basis expansions)
    =#
    if type == "Legendre"

        #finds polynomials themselves p_m (tau)
        basis_array = zeros(Float64, order + 1, size(tau)[1])
        basis_array[1, :] .= 1.0
        basis_array[2, :] .= tau

        for i in 3 : order + 1
            basis_array[i, :] .= 1. ./ (i - 1) .* ((2 * i - 3) .* tau .* basis_array[i - 1, :] .- (i - 2) * basis_array[i - 2, :])
        end

        #finds first derivatives dp_m/d_tau (tau)
        basis_array_deriv = zeros(Float64, order + 1, size(tau)[1])
        basis_array_deriv[2, :] .= 1.0

        for i in 3 : order + 1
            basis_array_deriv[i, :] = (i - 1) .* basis_array[i - 1, :] .+ tau .* basis_array_deriv[i-1, :]
        end

        #finds second derivatives d^2 p_m/ d_tau^2 (tau)
        basis_array_deriv2 = zeros(Float64, order + 1, size(tau)[1])

        for i in 3 : order + 1
            basis_array_deriv2[i, :] = (i + 1) .* basis_array_deriv[i - 1, :] .+ tau .* basis_array_deriv2[i-1, :]
        end

    elseif type == "Chebyshev"
        basis_array = zeros(Float64, order + 1, size(tau)[1])
        basis_array[1, :] .= 1.0
        basis_array[2, :] .= tau

        for i in 3 : order + 1
            basis_array[i, :] .= 2 .* tau .* basis_array[i - 1, :] .- basis_array[i - 2, :]
        end

        basis_array_deriv = zeros(Float64, order + 1, size(tau)[1])
        basis_array_deriv[2, :] .= 1.0
        basis_array_deriv[3, :] .= 4.0 .* tau

        for i in 4 : order + 1
            basis_array_deriv[i, :] = (i - 1) ./ (i - 2) .* 2 .* tau .* basis_array_deriv[i - 1, :] .- (i - 1) ./ (i - 3) .* basis_array_deriv[i - 2, :]
        end

        basis_array_deriv2 = zeros(Float64, order + 1, size(tau)[1])
        basis_array_deriv2[3, :] .= 4.0 

        for i in 4 : order + 1
            basis_array_deriv2[i, :] = (i - 1) ./ (i - 2) .* 2 .* basis_array_deriv[i - 1, :] .+ (i - 1) ./ (i - 2) .* 2 .* tau .* basis_array_deriv2[i - 1, :] .- (i - 1) ./ (i - 3) .* basis_array_deriv2[i - 2, :]
        end
    end

    return basis_array, basis_array_deriv, basis_array_deriv2
end 

function Jacobian_IVP(solution, RHS, interval, tau)

    #=
    This function calculates the Jacobian matrix for a given solution.
    It uses the solution, a right-hand side function RHS, time interval, and tau.
    It applies finite difference method to estimate the Jacobian.
    The function returns a Jacobian matrix. so for a system of size 6, the output is a 6x6 matrix
    =#

    # Set a small value epsilon for finite differences
    eps = 1e-6

    # Extract the start and end time from the interval
    t0, t1 = interval

    # Determine the size of the system
    sys_size = size(solution, 1)

    # Initialize matrices for finite differences and Jacobian
    Xm = zeros(Float64, sys_size, sys_size)
    Xp = zeros(Float64, sys_size, sys_size)
    Jacobian_array = zeros(Float64, sys_size, sys_size)

    # Calculate the Jacobian using finite differences
    for i in 1 : sys_size
        Xm[i, :] = copy(solution)
        Xp[i, :] = copy(solution)
        Xp[i, i] += eps
        Xm[i, i] -= eps

        #finite difference for differentiation for finding jacobian
        Jacobian_array[:, i] .= 
            ((t1 - t0) / 2 * (RHS(Xp[i, :], [], (t1 - t0) / 2 .* tau + (t1 + t0) / 2) -
             RHS(Xm[i, :], [], (t1 - t0) / 2 .* tau + (t1 + t0) / 2))) / (eps * 2)
    end

    # Return the Jacobian array
    return Jacobian_array
end

function Jacobian_BVP(solution, RHS, interval, tau)

    #=
    This function calculates the Jacobian matrix for a given solution.
    It uses the solution, a right-hand side function RHS, time interval, and tau.
    It applies finite difference method to estimate the Jacobian.
    The function returns a Jacobian matrix.
    =#

    # Set a small value epsilon for finite differences
    eps = 1e-6

    # Extract the start and end time from the interval
    t0, t1 = interval

    # Determine the size of the system
    sys_size = size(solution, 1)

    # Initialize matrices for finite differences and Jacobian
    Xm = zeros(Float64, sys_size, sys_size)
    Xp = zeros(Float64, sys_size, sys_size)
    Jacobian_array = zeros(Float64, sys_size, sys_size)

    # Calculate the Jacobian using finite differences
    for i in 1 : sys_size
        Xm[i, :] = copy(solution)
        Xp[i, :] = copy(solution)
        Xp[i, i] += eps
        Xm[i, i] -= eps
        Jacobian_array[:, i] .= 
            (((t1 - t0) / 2)^2 * (RHS(Xp[i, :], [], (t1 - t0) / 2 .* tau + (t1 + t0) / 2) -
             RHS(Xm[i, :], [], (t1 - t0) / 2 .* tau + (t1 + t0) / 2))) / (eps * 2)
    end

    # Return the Jacobian array
    return Jacobian_array
end

function Jacobian_automatic(solution, RHS, interval, tau)
    t0, t1 = interval
    F_scaled(X) = (t1 - t0) / 2 .* RHS(X, [], (t1 - t0) / 2 * tau + (t1 + t0) / 2)
    J = ForwardDiff.jacobian(F_scaled, solution)
    return J
end

function post_process_IVP(time, coefficients, sys_size, samples, type)

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
    spectral_basis_local, _, _ = create_basis_set(tau, order, "Chebyshev")

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

function post_process_BVP(time, coefficients, sys_size, samples)

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
    spectral_basis_local, _, _ = create_basis_set(tau, order, "Chebyshev")

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

    return time_array, solution_array
end

function compute_error(time_1, solution_1, time_2, solution_2, samples, problem_type)

    #=
    This function computes the error between two solutions written in the time domain, using cubic interpolation.
    The error will inherit the samples of the second solution. 
    =#
    
    if problem_type == "IVP"
        solution_1 = kepler_to_cartesian_array(solution_1)
        solution_2 = kepler_to_cartesian_array(solution_2)
    end

    sys_size = size(solution_1)[2]
    solution_points_1 = zeros(Float64, size(solution_2)[1], size(solution_2)[2])

    for i in 1 : sys_size
        solution_interpolated = CubicInterpolator(time_1, solution_1[:, i])
        solution_points_1[:, i] = solution_interpolated.(time_2)
    end

    x1, y1, z1 = solution_points_1[:, 1], solution_points_1[:, 2], solution_points_1[:, 3]
    x2, y2, z2 = solution_2[:, 1], solution_2[:, 2], solution_2[:, 3]

    error = ((x1 .- x2).^2 + (y1 .- y2).^2 + (z1 .- z2).^2).^0.5
    error_interpolated = CubicInterpolator(time_2, error)
    time_samples = vcat(LinRange(time_2[1], time_2[end], samples))
    error_sampled = error_interpolated.(time_samples)
    clamp!(error_sampled, 1e-16, 1000)

    return time_samples, error_sampled
end

function compute_hamiltonian(time, solution)
    energy = mu_Earth ./ 2 ./ solution[:, 1]
    energy_error = energy .- energy[1]
    clamp!(energy_error, 1e-20, 100)
    return time, energy_error
end