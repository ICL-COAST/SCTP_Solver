
function compare_numerical_spectral_IVP(solution_name, x0, formulation_type, numerical_solvers, numerical_solver_names, spectral_solvers, spectral_solver_names, relative_tolerance_numerical, absolute_tolerance_numerical, samples)
    
    if !isdir("Results")
        mkdir("Results")
    end

    if !isdir("Results/" * solution_name * "_performance_analysis")
        mkdir("Results/" * solution_name * "_performance_analysis")
    end

    t_nums, sol_nums, num_b_times = [], [], []
    t_specs, c_specs, e_mats, i_mats, spec_b_times = [], [], [], [], []

    for i in 1 : size(spectral_solvers)[1]
        println("Benchmarking spectral solver ", i, " ...")
        solver = spectral_solvers[i]
        spec_b_time = @belapsed $solver($x0, time_span_IVP ./ norm_time_E, s_interval_IVP / norm_time_E, relative_tol_IVP, spectral_tol_IVP, spectral_iter_IVP, $formulation_type);
        println("Computing the spectral solution ", i, " ...")
        t_spec, c_spec, e_mat, i_mat =  solver(x0, time_span_IVP ./ norm_time_E, s_interval_IVP / norm_time_E, relative_tol_IVP, spectral_tol_IVP, spectral_iter_IVP, formulation_type);
        push!(t_specs, t_spec);
        push!(c_specs, c_spec);
        push!(e_mats, e_mat);
        push!(i_mats, i_mat);
        push!(spec_b_times, spec_b_time);
    end

    for i in 1 : size(numerical_solvers)[1]
        println("Benchmarking numerical solver ", i, " ...")
        solver = numerical_solvers[i]
        num_b_time = @belapsed $solver($x0, time_span_IVP ./ norm_time_E, $relative_tolerance_numerical, $absolute_tolerance_numerical, $formulation_type);
        println("Computing the numerical solution ", i, " ...")
        t_num, sol_num =  solver(x0, time_span_IVP ./ norm_time_E,  relative_tolerance_numerical, absolute_tolerance_numerical, formulation_type);
        push!(t_nums, t_num);
        push!(sol_nums, sol_num);
        push!(num_b_times, num_b_time);
    end
    println("Computing the exact solution ...")
    solver_exact = numerical_solvers[1]
    t_exact, sol_exact =  solver_exact(x0, time_span_IVP ./ norm_time_E, 1e-14, 1e-14, formulation_type)
    println("Saving the spectral solutions ...")
    for i in 1 : size(spectral_solvers)[1]
        save_solution_spectral("Results/" * solution_name * "_performance_analysis", solution_name * "_" * spectral_solver_names[i], formulation_type, t_specs[i], c_specs[i], e_mats[i], i_mats[i], size(x0)[1], "IVP")
    end
    println("Saving the numerical solutions ...")
    for i in 1 : size(numerical_solvers)[1]
        save_solution_numerical("Results/" * solution_name * "_performance_analysis", solution_name * "_" * numerical_solver_names[i], formulation_type, t_nums[i], sol_nums[i], size(x0)[1], "IVP")
    end
    println("Plotting results ...")
    for i in 1 : size(spectral_solvers)[1]
        plot_spectral_solution("Results/" * solution_name * "_performance_analysis", t_specs[i], c_specs[i], solution_name * "_" * spectral_solver_names[i], formulation_type, samples, "IVP", size(x0)[1])
    end
    plot_spectral_numerical_error("Results/" * solution_name * "_performance_analysis", t_specs, c_specs, spec_b_times, t_nums, sol_nums, num_b_times, t_exact, sol_exact, solution_name, numerical_solver_names, spectral_solver_names, formulation_type, samples, size(x0)[1], "IVP")
    save_error("Results/" * solution_name * "_performance_analysis", solution_name, numerical_solver_names, spectral_solver_names, formulation_type, t_exact, sol_exact, t_nums, sol_nums, t_specs, c_specs, size(x0)[1], samples, "IVP")
    save_CPU_Times("Results/" * solution_name * "_performance_analysis", solution_name, numerical_solver_names, spectral_solver_names, num_b_times, spec_b_times)
    println("Done!")
    PyPlot.close()

end

function compare_numerical_spectral_BVP(solution_name, x1, x2, numerical_solvers, numerical_solver_names, spectral_solvers, spectral_solver_names, time_step, samples)
    
    if !isdir("Results")
        mkdir("Results")
    end

    if !isdir("Results/" * solution_name * "_performance_analysis")
        mkdir("Results/" * solution_name * "_performance_analysis")
    end

    t_nums, sol_nums, num_b_times = [], [], []
    t_specs, c_specs, e_mats, i_mats, spec_b_times = [], [], [], [], []

    for i in 1 : size(spectral_solvers)[1]
        println("Benchmarking spectral solver ", i, " ...")
        solver = spectral_solvers[i]
        spec_b_time = @belapsed $solver($x1, $x2, time_span_BVP, spectral_tol_BVP, spectral_iter_BVP);
        println("Computing the spectral solution ", i, " ...")
        t_spec, c_spec, e_mat, i_mat =  solver(x1, x2, time_span_BVP, spectral_tol_BVP, spectral_iter_BVP);
        push!(t_specs, t_spec);
        push!(c_specs, c_spec);
        push!(e_mats, e_mat);
        push!(i_mats, i_mat);
        push!(spec_b_times, spec_b_time);
    end

    for i in 1 : size(numerical_solvers)[1]
        println("Benchmarking numerical solver ", i, " ...")
        solver = numerical_solvers[i]
        num_b_time = @belapsed $solver($x1, $x2, time_span_BVP, $time_step);
        println("Computing the numerical solution ", i, " ...")
        t_num, sol_num =  solver(x1, x2, time_span_BVP, time_step);
        push!(t_nums, t_num);
        push!(sol_nums, sol_num);
        push!(num_b_times, num_b_time);
    end
    println("Computing the exact solution ...")
    solver_exact = numerical_solvers[1]
    t_exact, sol_exact =  solver_exact(x1, x2, time_span_BVP, exact_time_step);
    println("Saving the spectral solutions ...")
    for i in 1 : size(spectral_solvers)[1]
        save_solution_spectral("Results/" * solution_name * "_performance_analysis", solution_name * "_" * spectral_solver_names[i], "Cowell", t_specs[i], c_specs[i], e_mats[i], i_mats[i], size(x1)[1], "BVP")
    end
    println("Saving the numerical solutions ...")
    for i in 1 : size(numerical_solvers)[1]
        save_solution_numerical("Results/" * solution_name * "_performance_analysis", solution_name * "_" * numerical_solver_names[i], "Cowell", t_nums[i], sol_nums[i], size(x1)[1], "BVP")
    end
    println("Plotting results ...")
    for i in 1 : size(spectral_solvers)[1]
        plot_3D_orbit("Results/" * solution_name * "_performance_analysis", t_specs[i], c_specs[i], solution_name * "_" * spectral_solver_names[i], "Cowell", samples, "BVP", size(x1)[1])
    end
    plot_spectral_numerical_error("Results/" * solution_name * "_performance_analysis", t_specs, c_specs, spec_b_times, t_nums, sol_nums, num_b_times, t_exact, sol_exact, solution_name, numerical_solver_names, spectral_solver_names, "Cowell", samples, size(x1)[1], "BVP")
    save_error("Results/" * solution_name * "_performance_analysis", solution_name, numerical_solver_names, spectral_solver_names, "Cowell", t_exact, sol_exact, t_nums, sol_nums, t_specs, c_specs, size(x1)[1], samples, "BVP")
    save_CPU_Times("Results/" * solution_name * "_performance_analysis", solution_name, numerical_solver_names, spectral_solver_names, num_b_times, spec_b_times)
    println("Done!")

end