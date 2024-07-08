
function RK45_solve(x0, t_span, rtol, atol, type)

    if type == "Cowell"
        r, v = kepler_to_cartesian(x0)
        x0_cartesian = [r[1], r[2], r[3], v[1], v[2], v[3]]
        prob = ODEProblem(Cowell, x0_cartesian, t_span);
        sol = DifferentialEquations.solve(prob, Tsit5(), reltol=rtol, abstol=atol, progress=true, progress_steps = 1);
        solution = mapreduce(permutedims, vcat, sol.u) 
        solution = cartesian_to_kepler_array(solution)
        time_s = sol.t 
    elseif type == "Equinoctial"
        x0_equinoctial = kepler_to_equinoctial(x0)
        prob = ODEProblem(Equinoctial, x0_equinoctial, t_span);
        sol = DifferentialEquations.solve(prob, Tsit5(), reltol=rtol, abstol=atol, progress=true, progress_steps = 1);
        solution = mapreduce(permutedims, vcat, sol.u) 
        solution = equinoctial_to_kepler_array(solution)
        time_s = sol.t 
    end

    return time_s, solution
end

function Spectral_solve(x0, t_span, interval, rel_tolerance, it_tolerance, max_iter, type)

    if type == "Cowell"
        r, v = kepler_to_cartesian(x0)
        x0_cartesian = [r[1], r[2], r[3], v[1], v[2], v[3]]
        time_array, coefficient_array, iteration_matrix, error_matrix = ssolve_IVP(Cowell, Cowell_simple, x0_cartesian, t_span, interval, s_order_IVP, rel_tolerance, it_tolerance, max_iter, h_adaptive)

    elseif type == "Equinoctial"
        x0_equinoctial = kepler_to_equinoctial(x0)
        time_array, coefficient_array, iteration_matrix, error_matrix = ssolve_IVP(Equinoctial, Equinoctial_simple, x0_equinoctial, t_span, interval, s_order_IVP, rel_tolerance, it_tolerance, max_iter, h_adaptive)
    end

    return time_array, coefficient_array, iteration_matrix, error_matrix
end
