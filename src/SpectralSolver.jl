using FastGaussQuadrature, DifferentialEquations, BasicInterpolators, ForwardDiff, IterativeSolvers, Preconditioners, ODEInterfaceDiffEq, LoopVectorization

__precompile__()

function build_system_matrix_IVP(sys_size)

    #=
    This function builds the system matrix based on spectral basis, its derivative, Gauss-Legendre weights, Jacobian array, 
    =#

    # Calculate the order of the spectral basis and the number of integration points
    order, int_order = size(spectral_basis_gl_IVP)[1] - 1, size(spectral_basis_gl_IVP)[2]

    # Initialize the system matrix and Jacobian matrix with zeros
    sys_matrix = zeros(Float64, (order + 1) * sys_size, (order + 1) * sys_size)

    # Calculate the system matrix
    #This finds P - Br
    for i in 1 : order + 1
        for j in 1 : order + 1
            for k in 1 : sys_size # avoids vector operations
                sys_matrix[(i - 1) * sys_size + k, (j - 1) * sys_size + k] = 
                sum(gl_weights_IVP .* spectral_basis_gl_IVP[j, :] .* spectral_basis_deriv_gl_IVP[i, :]) - spectral_basis_gl_IVP[i, end] .* spectral_basis_gl_IVP[j, end]
            end
        end
    end

    # Return the system matrix
    return sys_matrix
end

function build_jacobian_matrix(Jacobian_spectrum, sys_size)
    #=
   This function builds the Jacobian matrix
   based on spectral basis, its derivative, Gauss-Legendre weights, Jacobian array, 
   system size, and a boolean flag use_jacobian
   =#

   # Calculate the order of the spectral basis and the number of integration points
   order, int_order = size(Jacobian_spectrum)[1] - 1, size(Jacobian_spectrum)[2]

   # Initialize the Jacobian matrix with zeros
   jac_matrix = zeros(Float64, (order + 1) * sys_size, (order + 1) * sys_size)
   #println(size(Jacobian_spectrum))
   # Calculate Jacobian matrix if use_jacobian is set to true
   Threads.@threads for i in 1 : order + 1
        for j in 1 : i

            if i == j == 1
                jac_matrix[(j - 1) * sys_size + 1 : j * sys_size, (i - 1) * sys_size + 1 : i * sys_size] .= - pi * Jacobian_spectrum[1, :, :]
            elseif i == j && i + j - 1 <= order + 1
                jac_matrix[(j - 1) * sys_size + 1 : j * sys_size, (i - 1) * sys_size + 1 : i * sys_size] .= - pi ./ 2 .* Jacobian_spectrum[1, :, :] .- pi / 4 .* Jacobian_spectrum[i + j - 1, :, :]
            elseif i == j && i + j - 1 > order + 1
                jac_matrix[(j - 1) * sys_size + 1 : j * sys_size, (i - 1) * sys_size + 1 : i * sys_size] .= - pi ./ 2 .* Jacobian_spectrum[1, :, :]
            elseif i != j && i + j - 1 <= order + 1
                jac_matrix[(j - 1) * sys_size + 1 : j * sys_size, (i - 1) * sys_size + 1 : i * sys_size] .= - pi ./ 4 .* (Jacobian_spectrum[abs(i - j + 1), :, :] .+ Jacobian_spectrum[i + j - 1, :, :])
            else
                jac_matrix[(j - 1) * sys_size + 1 : j * sys_size, (i - 1) * sys_size + 1 : i * sys_size] .= - pi ./ 4 .* (Jacobian_spectrum[abs(i - j + 1), :, :])
            end

            jac_matrix[(i - 1) * sys_size + 1 : i * sys_size, (j - 1) * sys_size + 1 : j * sys_size] .=
                    jac_matrix[(j - 1) * sys_size + 1 : j * sys_size, (i - 1) * sys_size + 1 : i * sys_size]
        end
    end
   # Return the jacobian matrix, NOTE: this is the negative of the jacobian matrix
   return jac_matrix
end

function build_RHS_term_IVP(RHS_array, Jacobian_matrix, C, sys_size)
    
    #=
    This function builds the right-hand side (RHS) of a spectral system of equations. 
    It uses spectral basis, Gauss-Legendre weights, RHS array, Jacobian array, solution array, 
    system size, and a boolean flag to decide whether to use Jacobian or not.
    It returns a vector "Source" which represents the RHS term of the system.
    =#

    # Calculate the order of the spectral basis and the number of integration points
    order, int_order = size(spectral_basis_gl_IVP)[1] - 1, size(spectral_basis_gl_IVP)[2]

    # Initialize the Source vector with zeros
    Source = zeros(Float64, (order + 1) * sys_size)

    # Calculate the Source vector, lower case s term in the equation
    Threads.@threads for i in 1 : order + 1
        for j in 1 : int_order
            Source[(i - 1) * sys_size + 1: i * sys_size] .+= 
            gl_weights_IVP[j] .* (- RHS_array[j, :] .* spectral_basis_gl_IVP[i, j])
        end
    end

    
    Source .-= Jacobian_matrix * C

    # Return the Source vector
    return Source
end

function build_BCs_IVP(X0, sys_size)

    #=
    This function builds the boundary conditions (BCs) for a spectral system of equations.
    It uses spectral basis, initial condition X0, and the system size.
    It returns a vector "BC" which represents the boundary conditions of the system.
    =#

    # Calculate the order of the spectral basis
    order = size(spectral_basis_gl_IVP)[1] - 1

    # Initialize the BC vector with zeros
    BC = zeros(Float64, (order + 1) * sys_size)

    # Calculate the BC vector
    for i in 1 : order + 1
        BC[(i - 1) * sys_size + 1: i * sys_size] .= - spectral_basis_gl_IVP[i, 1] .* X0[:]
    end

    # Return the BC vector
    return BC
end

function build_RHS_array_IVP(solution, RHS, interval)

    int_order = size(solution)[1]
    sys_size = size(solution)[2]
    t0, t1 = interval
    RHS_array = zeros(Float64, int_order, sys_size)

    Threads.@threads for i in 1 : int_order
        RHS_array[i, :] .= 0.5 * (t1 - t0) * RHS(solution[i, :], [], (t1 - t0) / 2 .* gl_nodes_IVP[i] .+ (t1 + t0) / 2)
    end

    return RHS_array
end


function build_Jacobian_array_IVP(solution, RHS, interval)
    #holds the array of jacobian matrices for each of the integration nodes

    #println("inside build jacobian array ivp")
    #println(size(solution))
    int_order, sys_size = size(solution)
    Jacobian_array = zeros(Float64, int_order, sys_size, sys_size)
    Threads.@threads for i in 1 : int_order
        Jacobian_array[i, :, :] .= Jacobian_IVP(solution[i, :], RHS, interval, gc_nodes_IVP[i])
    end
    #println("Jacobian array")
    #println(size(Jacobian_array))
    return Jacobian_array
end


function print_progress(time_span, t0, t1, samples)

    progress_now = (t1 - time_span[1]) / (time_span[2] - time_span[1]) * 100.0
    progress_previous = (t0 - time_span[1]) / (time_span[2] - time_span[1]) * 100.0
    progress_step = trunc(Int64, progress_now / (100.0 / samples)) * (100.0 / samples)

    if progress_previous < progress_step
        println("Solved ", round(progress_now, digits=0), " %")
    end
end

function solve_step(jac_mat, source, sys_size, tolerance)

    order = size(spectral_basis_gl_IVP)[1] - 1
    error = Inf
    C_new = zeros(Float64, (order + 1) * sys_size)
    C_term = zeros(Float64, (order + 1) * sys_size)
    C_term2 = zeros(Float64, (order + 1) * sys_size)

    mul!(C_term, sys_matrix_IVP_inv, source)

    while error > tolerance
        C_new .+= C_term
        mul!(C_term2, jac_mat, C_term)
        mul!(C_term, sys_matrix_IVP_inv, C_term2)
        error = norm(C_term)
    end

    return C_new
end

function solve_interval_IVP(RHS, C_initial, X0, interval, sys_size, max_iter, tolerance)
    #=

    GCN solution for one spectral interval
    Where the magic actually happens

    RHS is solver
    =#

    C = copy(C_initial) # initial guess from RK45/otherwise
    B = build_BCs_IVP(X0, sys_size) # can probably be shifted to be const, B = -bl

    it_array = Vector{Int}(undef, max_iter)
    error_array = Vector{Float64}(undef, max_iter)
    
    error = Inf
    try
        for it in 1:max_iter
            #println("ITERATION")
            #println(it)
            #println(C)
            
            #spectral basis gl and gc are constants, the legendre and chebyshev basis sets
            solution_array = build_solution(C, spectral_basis_gl_IVP, sys_size) # based on legendre polynomials
            solution_array_gc = build_solution(C, spectral_basis_gc_IVP, sys_size) # based on chebyshev polynomials
            #println("SOlution array")
            #println(solution_array_gc)
            #=
            try
                
                if any(!isfinite, solution_array_gc)
                    println("solution_array_gc ITS BROKEN!!!")
                    println(solution_array_gc)
                end
            catch e
                println(e)
                prinln("IHTFP")
            end
            =#

            #println(maximum(solution_array))
            #println(size(solution_array))
            #maxi(c)

            RHS_array = build_RHS_array_IVP(solution_array, RHS, interval) #array of g(x_k,tau)
            Jacobian_array_gc = build_Jacobian_array_IVP(solution_array_gc, RHS, interval)

            #if any(!isfinite, Jacobian_array_gc)
            #    println("Jacobian_array_gc ITS BROKEN!!!")
            #    println(Jacobian_array_gc)
            #end

            JS = build_Jacobian_spectrum(Jacobian_array_gc, sys_size, "IVP")
            #if any(!isfinite, JS)
            #    println("JS ITS BROKEN!!!")
            #    println(JS)
            #end
            Jacobian_matrix = build_jacobian_matrix(JS, sys_size) #creates jacobian for use in LHS directly and RHS in func
            #if any(!isfinite, Jacobian_matrix)
            #    println("Jacobian matrix ITS BROKEN!!!")
            #end
            #println("Diagnostics")
            #println(maximum(solution_array_gc))
            #println(maximum(Jacobian_array_gc))
            #println(maximum(JS))

            S = build_RHS_term_IVP(RHS_array, Jacobian_matrix, C, sys_size) #creates combined s and Jk ck term
            
            #@time C_new = solve_step(Jacobian_matrix, S .+ B, sys_size, 2e-16)

            C_new = (sys_matrix_IVP .- Jacobian_matrix)\(S .+ B) #Jacobian matrix  = -J^k from the paper
            error = norm(C_new .- C) / norm(C)
            it_array[it] = it
            error_array[it] = error
            #println("The error is ", error)
            if error < tolerance
                #println("Interval solved in ", it, " iterations!")
                resize!(it_array, it)
                resize!(error_array, it)
                return C_new, it_array, error_array
            end

            C .= C_new
            
        end
    catch e
        println(e)
        println("Could not calculate C_new")
        println("Crashing out!")
        println(size(sys_matrix_IVP))
        try
            println(Jacobian_matrix)
            println(size(Jacobian_matrix))
        catch e
            println("huh what the ####")
            println(e)
        end
        println(size(sys_matrix_IVP .- Jacobian_matrix))
        println(size(S))
        println(size(B))
        println(size(S .+ B))
    end

    return C, it_array, error_array
end
#=
function ssolve_IVP(RHS, RHS_simple, X0, time_span, interval, order, relative_tolerance, iterative_tolerance, max_iter, h_adaptive)
    #=
    Solution loop for GCN Spectral Solver over the whole time period
    This is essentially just a wrapper for calling the actual solver
    and then incrementing the time and X0

    RHS is the solver used to propagate solution
    RHS_simple is the solver used to propagate simpler RK45 solution
    =#
    
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
    
    #loop through the sub-intervals
    try
        while current_time < time_span[2]
            if step < 1
                t0, t1 = time_span[1], time_span[1] + reference_interval
                coarse_problem = ODEProblem(RHS_simple, X0, (t0, t1));
                sol = DifferentialEquations.solve(coarse_problem, Tsit5(), reltol=1e-5, abstol=1e-5, dtmax=(t1 - t0) / 5);
                time_coarse = sol.t
                sol_coarse = mapreduce(permutedims, vcat, sol.u)
                C_initial = build_spectrum(time_coarse, sol_coarse); 
            else
                t0, t1 = solution_time[end][2], solution_time[end][2] + reference_interval
                C_initial = C
            end

            current_time = t0;
            if current_time >= time_span[2] || abs(current_time - time_span[2]) < epsilon
                break
            end

            rk45loop = function_calls
            #Use a coarse RK45 with Tsit5 coefficients(better performance, newer coeffs) to initialise guess for spectral coeffss
            
            rk45loop = function_calls - rk45loop

            global call_structure[end,2] += rk45loop


            if h_adaptive
                println("Adapative, not permitted") # warn if it goes here
                break
            end
            
            t1 = min(t1, time_span[2])

            iterative_solution_loop = function_calls

            C, it, error = solve_interval_IVP(RHS, C_initial, X0, (t0, t1), sys_size, max_iter, iterative_tolerance);

            iterative_solution_loop = function_calls - iterative_solution_loop

            global call_structure[end,3] += iterative_solution_loop

            push!(solution_time, [t0, t1]);
            push!(solution_coefficients, C);
            push!(iteration_matrix, it);
            push!(error_matrix, error);
            X0 = zeros(Float64, sys_size);

            for j in 1 : order + 1
                X0 .+= C[(j - 1) * sys_size + 1 : j * sys_size] .* spectral_basis_gl_IVP[j, end]
            end

            step += 1;
        end

        solution_time = reduce(hcat, solution_time)'
        solution_coefficients = reduce(hcat, solution_coefficients)'

        return solution_time, solution_coefficients, iteration_matrix, error_matrix

    catch e
        println("early finish on execution, error")
        solution_time = reduce(hcat, solution_time)'
        solution_coefficients = reduce(hcat, solution_coefficients)'
        return solution_time, solution_coefficients, iteration_matrix, error_matrix
    end
end
=#

function ssolve_IVP(RHS, RHS_simple, X0, time_span, interval, order, relative_tolerance, iterative_tolerance, max_iter, h_adaptive)
    #=
    Solution loop for GCN Spectral Solver over the whole time period
    This is essentially just a wrapper for calling the actual solver
    and then incrementing the time and X0

    RHS is the solver used to propagate solution
    RHS_simple is the solver used to propagate simpler RK45 solution
    =#
    
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
    
    #loop through the sub-intervals
    try
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

            rk45loop = function_calls
            #Use a coarse RK45 with Tsit5 coefficients(better performance, newer coeffs) to initialise guess for spectral coeffss
            coarse_problem = ODEProblem(RHS_simple, X0, (t0, t1));
            sol = DifferentialEquations.solve(coarse_problem, Tsit5(), reltol=1e-5, abstol=1e-5, dtmax=(t1 - t0) / 5);

            rk45loop = function_calls - rk45loop

            global call_structure[end,2] += rk45loop

            time_coarse = sol.t
            sol_coarse = mapreduce(permutedims, vcat, sol.u)
            C_initial = build_spectrum(time_coarse, sol_coarse); 

            if h_adaptive
                #println("Adapative") # warn if it goes here
                rel_error = build_spectral_error(time_coarse, sol_coarse, C_initial, sys_size)
                t1 = t0 + (t1 - t0) * (relative_tolerance / rel_error) ^ (1 / SF / min(order, -log10(iterative_tolerance)))
                t1 = min(t1, time_span[2])

                #same as non-adapative RK45 initialiser
                coarse_problem = ODEProblem(RHS_simple, X0, (t0, t1));
                sol = DifferentialEquations.solve(coarse_problem, Tsit5(), reltol=1e-5, abstol=1e-5, dtmax=(t1 - t0) / 5);
                time_coarse = sol.t
                sol_coarse = mapreduce(permutedims, vcat, sol.u)
                C_initial = build_spectrum(time_coarse, sol_coarse);
            end
            
            t1 = min(t1, time_span[2])

            iterative_solution_loop = function_calls

            C, it, error = solve_interval_IVP(RHS, C_initial, X0, (t0, t1), sys_size, max_iter, iterative_tolerance);

            iterative_solution_loop = function_calls - iterative_solution_loop

            global call_structure[end,3] += iterative_solution_loop

            push!(solution_time, [t0, t1]);
            push!(solution_coefficients, C);
            push!(iteration_matrix, it);
            push!(error_matrix, error);
            X0 = zeros(Float64, sys_size);

            for j in 1 : order + 1
                X0 .+= C[(j - 1) * sys_size + 1 : j * sys_size] .* spectral_basis_gl_IVP[j, end]
            end

            step += 1;
        end

        solution_time = reduce(hcat, solution_time)'
        solution_coefficients = reduce(hcat, solution_coefficients)'

        return solution_time, solution_coefficients, iteration_matrix, error_matrix

    catch e
        println("early finish on execution, error")
        solution_time = reduce(hcat, solution_time)'
        solution_coefficients = reduce(hcat, solution_coefficients)'
        return solution_time, solution_coefficients, iteration_matrix, error_matrix
    end
end
