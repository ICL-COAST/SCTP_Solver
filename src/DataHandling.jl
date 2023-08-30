using System

function save_solution_spectral(root_folder, solution_name, solution_type, solution_time, solution_coefficients, iteration_matrix, error_matrix, sys_size, problem_type)

    N_intervals, n_coeffs = size(solution_coefficients)
    order = trunc(Int, n_coeffs / sys_size) - 1

    if problem_type == "IVP"
        if solution_type == "Cowell"
            solution_time_scaled = solution_time
            solution_coefficients_scaled = copy(solution_coefficients)
            for i in 1 : N_intervals
                C = reshape(solution_coefficients[i, :], (sys_size, order + 1))'
                solution_coefficients_scaled[i, :] = reshape(C', sys_size * (order + 1))
            end
        elseif solution_type == "Equinoctial"
            solution_time_scaled = solution_time
            solution_coefficients_scaled = copy(solution_coefficients)
            for i in 1 : N_intervals
                C = reshape(solution_coefficients[i, :], (sys_size, order + 1))'
                solution_coefficients_scaled[i, :] = reshape(C', sys_size * (order + 1))
            end
        end
    else
        solution_time_scaled = solution_time
        solution_coefficients_scaled = copy(solution_coefficients)
        for i in 1 : N_intervals
            C = reshape(solution_coefficients[i, :], (sys_size, order + 1))'
            solution_coefficients_scaled[i, :] = reshape(C', sys_size * (order + 1))
        end
    end

    if !isdir(root_folder)
        mkdir(root_folder)
    end

    # Create the directory
    if !isdir(root_folder * "/" * solution_name)
        mkdir(root_folder * "/" * solution_name)
    end
    
    # Create the first file
    open(joinpath( root_folder * "/" * solution_name, solution_name * "_coefficients.csv"), "w") do file
        for i in 1 : N_intervals
            write(file, string(solution_time_scaled[i, 1]), ", ")
            write(file, string(solution_time_scaled[i, 2]), ", ")
            for j in 1 : n_coeffs
                write(file, string(solution_coefficients_scaled[i, j]), ", ")
            end
            write(file, "\n")
        end
    end
    
    # Create the second file
    open(joinpath( root_folder * "/" * solution_name, solution_name * "_error.csv"), "w") do file
        for i in 1 : N_intervals
            write(file, string(size(iteration_matrix[i])[1]), ", ")
            for j in 1 : size(error_matrix[i, :])[1]
                write(file, string(error_matrix[i, j]), ", ")
            end
            write(file, "\n")
        end
    end
end

function open_solution_spectral(root_folder, solution_name)

    coefficient_data = []
    error_matrix = []

    # Open the file
    open(root_folder * "\\" * solution_name * "\\" * solution_name * "_coefficients.csv", "r") do file
        # Iterate over each line in the file
        for line in eachline(file)
            # Split the line on commas and convert it to an array
            row = split(replace(replace(line, '[' => ""), ']' => ""), ", ")
            # Push the row into the data array
            push!(coefficient_data, parse.(Float64,row[1:end-1]))
        end
    end

    # Open the file
    open(root_folder * "\\" * solution_name * "\\" * solution_name * "_error.csv", "r") do file
        # Iterate over each line in the file
        for line in eachline(file)
            # Split the line on commas and convert it to an array
            row = split(replace(replace(line, '[' => ""), ']' => ""), ", ")
            println
            # Push the row into the data array
            push!(error_matrix, parse.(Float64,row[2:end-1]))
        end
    end

    coefficient_data = hcat(coefficient_data...)'

    solution_time = coefficient_data[:, 1:2]
    solution_coefficients = coefficient_data[:, 3:end]

    return solution_time, solution_coefficients, error_matrix
end

function save_solution_numerical(root_folder, solution_name, solution_type, solution_time, solution_values, sys_size, problem_type)

  
    solution_time_scaled = solution_time
    solution_values_scaled = copy(solution_values)
    if problem_type == "IVP"
        solution_values_scaled[:, 1] .*= norm_distance_E
    end

    if !isdir(root_folder)
        mkdir(root_folder)
    end

    # Create the directory
    if !isdir(root_folder * "/" * solution_name)
        mkdir(root_folder * "/" * solution_name)
    end
    
    # Create the first file
    open(joinpath( root_folder * "/" * solution_name, solution_name * "_numerical_solution.csv"), "w") do file
        for i in 1 : size(solution_time)[1]
            write(file, string(solution_time_scaled[i]), ", ")
            for j in 1 : sys_size
                write(file, string(solution_values_scaled[i, j]), ", ")
            end
            write(file, "\n")
        end
    end
end

function save_error(root_folder, solution_name, numerical_solution_names, spectral_solution_names, solution_type, time_exact, solution_exact, time_nums, solution_nums, time_specs, coeff_specs, sys_size, samples, problem_type)

    if problem_type == "IVP"
        norm_time, norm_distance = norm_time_E, norm_distance_E
    else
        norm_time, norm_distance = norm_time_S, norm_distance_S
    end

    t_errors_numerical, errors_numerical = [], []
    t_errors_spectral, errors_spectral = [], []

    for i in 1 : size(numerical_solution_names)[1]
        t_error, error = compute_error(time_exact, solution_exact, time_nums[i], solution_nums[i], samples, problem_type)
        push!(t_errors_numerical, t_error)
        push!(errors_numerical, error)
    end

    for i in 1 : size(spectral_solution_names)[1]
        if problem_type == "IVP"
            spectral_time, spectral_solution = post_process_IVP(time_specs[i], coeff_specs[i], sys_size, samples, solution_type)
        else
            spectral_time, spectral_solution = post_process_BVP(time_specs[i], coeff_specs[i], sys_size, samples)
        end
        t_error, error = compute_error(time_exact, solution_exact, spectral_time, spectral_solution, samples, problem_type)
        push!(t_errors_spectral, t_error)
        push!(errors_spectral, error)
    end
    
    if !isdir(root_folder)
        mkdir(root_folder)
    end

     # Create the first file
     open(joinpath(root_folder, solution_name * "_errors.csv"), "w") do file
        for j in 1 : size(numerical_solution_names)[1]
            write(file, numerical_solution_names[j] * "_time", ", ")
            write(file, numerical_solution_names[j] * "_error", ", ")
        end
        for j in 1 : size(spectral_solution_names)[1]
            write(file, spectral_solution_names[j] * "_time", ", ")
            write(file, spectral_solution_names[j] * "_error", ", ")
        end
        write(file, "\n")
        for i in 1 : samples
            for j in 1 : size(numerical_solution_names)[1]
                write(file, string(t_errors_numerical[j][i]), ", ")
                write(file, string(errors_numerical[j][i]), ", ")
            end
            for j in 1 : size(spectral_solution_names)[1]
                write(file, string(t_errors_spectral[j][i]), ", ")
                write(file, string(errors_spectral[j][i]), ", ")
            end
            write(file, "\n")
        end
    end
    
end

function save_CPU_Times(root_folder, solution_name, numerical_solution_names, spectral_solution_names, CPUTimes_num, CPUTimes_spec)

    t_errors_numerical, errors_numerical = [], []
    t_errors_spectral, errors_spectral = [], []
    
    if !isdir(root_folder)
        mkdir(root_folder)
    end

     # Create the first file
     open(joinpath(root_folder, solution_name * "_CPUTimes.csv"), "w") do file
        for j in 1 : size(numerical_solution_names)[1]
            write(file, numerical_solution_names[j], ", ")
        end
        for j in 1 : size(spectral_solution_names)[1]
            write(file, spectral_solution_names[j], ", ")
        end
        write(file, "\n")
        for j in 1 : size(numerical_solution_names)[1]
            write(file, string(CPUTimes_num[j]), ", ")
        end
        for j in 1 : size(spectral_solution_names)[1]
            write(file, string(CPUTimes_spec[j]), ", ")
        end
    end
    
end