
function RK4_solve_BVP(x1, x2, t_span, time_step)

    function bc1!(residual, u, p, t)
        residual[1:3] .= u[1][1:3] .- x1 
        residual[4:6] .= u[end][1:3] .- x2 
    end

    function Cowell_Lambert!(du, u, p, t)
        du[4:6] .= Cowell_Lambert(u[1:3], p, t);
        du[1:3] .= u[4:6];
    end

    bp = BVProblem(Cowell_Lambert!, bc1!, [x1[1], x1[2], x1[3], 1.0, 1.0, 1.0], t_span)
    sol = DifferentialEquations.solve(bp, GeneralMIRK4(), dt=time_step)
    solution = mapreduce(permutedims, vcat, sol.u)
    time_solution = sol.t 
    return time_solution, solution[:, 1:3]
end

function Spectral_solve_BVP(x1, x2, t_span, it_tolerance, max_iter)

    time_array, coefficient_array, iteration_matrix, error_matrix = ssolve_BVP(Cowell_Lambert, x1, x2, t_span, s_order_BVP, it_tolerance, max_iter)

    return time_array, coefficient_array, iteration_matrix, error_matrix
end

function MCPI_solve_BVP(x1, x2, t_span, it_tolerance, max_iter)

    time_array, coefficient_array, iteration_matrix, error_matrix = ssolve_MCPI_BVP(Cowell_Lambert, x1, x2, t_span, m_order_BVP, it_tolerance, max_iter)

    return time_array, coefficient_array, iteration_matrix, error_matrix
end