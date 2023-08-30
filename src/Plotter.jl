
using LaTeXStrings, PyPlot

function plot_spectral_solution(root_folder, time_spectral, coefficients_spectral, solution_name, solution_type, samples, problem_type, sys_size)

    if problem_type == "IVP"
        norm_time, norm_distance = norm_time_E, norm_distance_E
    else
        norm_time, norm_distance = norm_time_S, norm_distance_S
    end

    if problem_type == "IVP"
        spectral_time, spectral_solution = post_process_IVP(time_spectral, coefficients_spectral, sys_size, samples, solution_type)
    else
        spectral_time, spectral_solution = post_process_BVP(time_spectral, coefficients_spectral, sys_size, samples)
    end

    PyPlot.figure(figsize=(7, 5))
    PyPlot.plot(spectral_time .* norm_time, spectral_solution[:, 1] .* norm_distance, label="");
    PyPlot.xlabel("Time [s]")
    PyPlot.ylabel("Semi-major axis [km]")
    PyPlot.title("Semi-major axis vs. time")
    PyPlot.savefig(root_folder * "/" * solution_name * "_semimajor_axis.png", dpi=600);
    PyPlot.close_figs()

    PyPlot.figure(figsize=(7, 5))
    PyPlot.plot(spectral_time .* norm_time, spectral_solution[:, 2], label="");
    PyPlot.xlabel("Time [s]")
    PyPlot.ylabel("Eccentricity [-]")
    PyPlot.title("Eccentricity vs. time")
    PyPlot.savefig(root_folder * "/" * solution_name * "_eccentricity.png", dpi=600);
    PyPlot.close_figs()

    PyPlot.figure(figsize=(7, 5))
    PyPlot.plot(spectral_time .* norm_time, spectral_solution[:, 3] .* 180 ./ pi, label="");
    PyPlot.xlabel("Time [s]")
    PyPlot.ylabel("Inclination [deg]")
    PyPlot.title("Inclination vs. time")
    PyPlot.savefig(root_folder * "/" * solution_name * "_inclination.png", dpi=600);
    PyPlot.close_figs()

    PyPlot.figure(figsize=(7, 5))
    PyPlot.plot(spectral_time .* norm_time, spectral_solution[:, 4] .* 180 ./ pi, label="");
    PyPlot.xlabel("Time [s]")
    PyPlot.ylabel("RAAN [deg]")
    PyPlot.title("Right Ascension of Ascending Node vs. time")
    PyPlot.savefig(root_folder * "/" * solution_name * "_RAAN.png", dpi=600);
    PyPlot.close_figs()

    PyPlot.figure(figsize=(7, 5))
    PyPlot.plot(spectral_time .* norm_time, spectral_solution[:, 5] .* 180 ./ pi, label="");
    PyPlot.xlabel("Time [s]")
    PyPlot.ylabel("Argument of Perigee [deg]")
    PyPlot.title("Argument of Perigee vs. time")
    PyPlot.savefig(root_folder * "/" * solution_name * "_arg_perigee.png", dpi=600);
    PyPlot.close_figs()

    PyPlot.figure(figsize=(7, 5))
    PyPlot.plot(spectral_time .* norm_time, spectral_solution[:, 6] .* 180 ./ pi, label="");
    PyPlot.xlabel("Time [s]")
    PyPlot.ylabel("True Anomaly [deg]")
    PyPlot.title("True Anomaly vs. time")
    PyPlot.savefig(root_folder * "/" * solution_name * "_true_anomaly.png", dpi=600);
    PyPlot.close_figs()

end

function plot_3D_orbit(root_folder, time_spectral, coefficients_spectral, solution_name, solution_type, samples, problem_type, sys_size)
    
    spectral_time, spectral_solution = post_process_BVP(time_spectral, coefficients_spectral, sys_size, samples)

    PyPlot.figure(figsize=(7, 5))
    PyPlot.plot3D(spectral_solution[:, 1], spectral_solution[:, 2], spectral_solution[:, 3], label="");
    PyPlot.xlabel("Distance [AU]")
    PyPlot.ylabel("Distance [AU]")
    PyPlot.zlabel("Distance [AU]")
    PyPlot.title("Semi-major axis vs. time")
    PyPlot.savefig(root_folder * "/" * solution_name * "_3D_orbit.png", dpi=600);
    PyPlot.close_figs()

end

function plot_spectral_numerical_error(root_folder, times_spectral, coefficients_spectral, btimes_spectral, times_numerical, solutions_numerical, btimes_numerical, time_exact, solution_exact, solution_name, solver_names, spectral_solver_names, solution_type, samples, sys_size, problem_type)

    if problem_type == "IVP"
        norm_time, norm_distance = norm_time_E, norm_distance_E
    else
        norm_time, norm_distance = 1.0, 1.0
    end

    t_errors_numerical, errors_numerical = [], []
    t_errors_spectral, errors_spectral = [], []
    for i in 1 : size(times_spectral)[1]
        if problem_type == "IVP"
            spectral_time, spectral_solution = post_process_IVP(times_spectral[i], coefficients_spectral[i], sys_size, samples, solution_type)
        else
            spectral_time, spectral_solution = post_process_BVP(times_spectral[i], coefficients_spectral[i], sys_size, samples)
        end
        t_error_spectral, error_spectral = compute_error(time_exact, solution_exact, spectral_time, spectral_solution, samples, problem_type)
        push!(t_errors_spectral, t_error_spectral)
        push!(errors_spectral, error_spectral)
    end

    for i in 1 : size(times_numerical)[1]
        t_error_numerical, error_numerical = compute_error(time_exact, solution_exact, times_numerical[i], solutions_numerical[i], samples, problem_type)
        push!(t_errors_numerical, t_error_numerical)
        push!(errors_numerical, error_numerical)
    end
    #println(size(t_errors_spectral), " ", size(errors_spectral), " ", size(spectral_solver_names))
    PyPlot.figure(figsize=(7, 5))
    PyPlot.yscale("log")
    if problem_type == "IVP"
        PyPlot.xlabel("Time [s]")
        PyPlot.ylabel("Error norm [km]")
    else
        PyPlot.xlabel("Time [years]")
        PyPlot.ylabel("Error norm [AU]")
    end
    PyPlot.title("Spectral vs. Numerical Errors")
    for i in 1 : size(times_spectral)[1]
        PyPlot.plot(t_errors_spectral[i] .* norm_time, errors_spectral[i] .* norm_distance, label=spectral_solver_names[i] * " Spectral Error [km]", marker="^");
    end
    for i in 1 : size(times_numerical)[1]
        PyPlot.plot(t_errors_numerical[i] .* norm_time, errors_numerical[i] .* norm_distance, label=solver_names[i] * " Numerical Error [km]", marker="o");
    end
    PyPlot.legend()
    PyPlot.grid(true, which="both", linewidth=0.2)
    PyPlot.savefig(root_folder * "/" * solution_name * "_spectral_numerical_error.png", dpi=600);
    PyPlot.close_figs()

    f, ax = PyPlot.subplots(figsize=(7, 5))
    PyPlot.title("CPU Times for Spectral and Numerical Solvers")
    bp = ax.bar(vcat(solver_names, spectral_solver_names), vcat(btimes_numerical, btimes_spectral))
    ax.set_ylabel("CPU Times [s]")
    PyPlot.savefig(root_folder * "/" * solution_name * "_spectral_numerical_CPUtimes.png", dpi=600);
    PyPlot.close_figs()

end