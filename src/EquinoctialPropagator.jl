using LinearAlgebra, CPUTime, Plots
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())




function Equinoctial(x, par, t)

    #=
    `Equinoctial(x, par, t)`

    This function computes the equinoctial orbital elements and their time derivatives based on the current state of the spacecraft.

    Inputs:
    - `x`: A 1D array of 6 elements representing initial orbital elements (p, f, g, h, k, L)
    - `par`: Auxiliary parameters used in the computation (not directly used in this function)
    - `t`: Time instance at which the computations are done

    Outputs:
    - The function returns a 1D array of 6 elements representing the time derivatives of the equinoctial orbital elements

    The function involves a variety of physical computations, including transformation to Earth Centered Inertial and Earth Centered Earth Fixed coordinates, 
    aerodynamic drag and geopotential computations, as well as derivation of the equations of motion in the equinoctial element set.
    =#

    # Unpack the elements of the state vector
    p, f, g, h, k, L = x[1], x[2], x[3], x[4], x[5], x[6]

    # Compute auxiliary variables used in subsequent calculations
    s2 = 1.0 + h^2 + k^2
    w = 1.0 + f * cos(L) + g * sin(L)
    alpha2 = h^2 - k^2
    r = p / w

    # Compute the position in the Earth-centered inertial (ECI) frame
    r_ECI = [r / s2 * (cos(L) + alpha2 * cos(L) + 2.0 * h * k * sin(L)), 
             r / s2 * (sin(L) - alpha2 * sin(L) + 2.0 * h * k * cos(L)),
             2.0 * r / s2 * (h * sin(L) - k * cos(L))]

    # Compute the velocity in the ECI frame
    v_ECI = [- 1.0 / s2 * sqrt(mu_Earth / p) * (sin(L) + alpha2 * sin(L) - 2 * h * k * cos(L) + g - 2.0 * f * h * k + alpha2 * g),
             - 1.0 / s2 * sqrt(mu_Earth / p) * (- cos(L) + alpha2 * cos(L) + 2 * h * k * sin(L) - f + 2.0 * g * h * k + alpha2 * f),
             2.0 / s2 * sqrt(mu_Earth / p) * (h * cos(L) + k * sin(L) + f * h + g * k)]

    # Transform the position and velocity vectors to the Earth-centered Earth-fixed (ECEF) frame
    r_ECEF = Tz(GMST(t, "IVP")) * r_ECI
    v_ECEF = Tz(GMST(t, "IVP")) * v_ECI

    # Compute the aerodynamic acceleration, if applicable
    if aero_drag
        a_drag_ECEF = aero_acceleration(r_ECEF, v_ECEF, spacecraft_mass, spacecraft_Cd, spacecraft_area, t)
        a_drag_ECI = Tz(GMST(t, "IVP"))' * a_drag_ECEF
    else
        a_drag_ECI = [0 0 0]'
    end

    # Compute the local orbital coordinate system (i_r, i_t, i_n)
    i_r = r_ECI ./ norm(r_ECI)
    i_n = cross(r_ECI, v_ECI) ./ norm(cross(r_ECI, v_ECI))
    i_t = cross(i_n, i_r) / norm(cross(i_n, i_r))
    Mat = hcat([i_r i_t i_n])'
    
    # Compute the geopotential acceleration
    if geopotential
        g_P_ECEF = geopotential_acceleration(geopotential_order, C_gp, S_gp, Tuple(r_ECEF))
    else
        g_P_ECEF = geopotential_acceleration(1, C_gp, S_gp, Tuple(r_ECEF))
    end
    g_P_ECI = Tz(GMST(t, "IVP"))' * g_P_ECEF + mu_Earth / norm(r_ECI)^3 .* r_ECI
    g_P_EQ = Mat * g_P_ECI
    a_drag_EQ = Mat * a_drag_ECI

    # Compute the total accelerations in the local orbital frame
    a_r = g_P_EQ[1] + a_drag_EQ[1]
    a_t = g_P_EQ[2] + a_drag_EQ[2]
    a_n = g_P_EQ[3] + a_drag_EQ[3]
    
    # Compute the time derivatives of the equinoctial elements
    dpdt = 2.0 * p / w * sqrt(p / mu_Earth) * a_t
    dfdt = sqrt(p / mu_Earth) * (a_r * sin(L) + ((w + 1) * cos(L) + f) * a_t / w - (h * sin(L) - k * cos(L)) * g * a_n / w)
    dgdt = sqrt(p / mu_Earth) * (-a_r * cos(L) + ((w + 1) * sin(L) + g) * a_t / w + (h * sin(L) - k * cos(L)) * f * a_n / w)
    dhdt = sqrt(p / mu_Earth) * s2 * a_n / 2.0 / w * cos(L)
    dkdt = sqrt(p / mu_Earth) * s2 * a_n / 2.0 / w * sin(L)
    dLdt = sqrt(mu_Earth * p) * (w / p)^2 + 1.0 / w * sqrt(p / mu_Earth) * (h * sin(L) - k * cos(L)) * a_n

    return [dpdt, dfdt, dgdt, dhdt, dkdt, dLdt]
end


function Equinoctial_simple(x, par, t)

    p, f, g, h, k, L = x[1], x[2], x[3], x[4], x[5], x[6]
    s2 = 1.0 + h^2 + k^2
    w = 1.0 + f * cos(L) + g * sin(L)
    alpha2 = h^2 - k^2
    r = p / w

    r_ECI = [r / s2 * (cos(L) + alpha2 * cos(L) + 2.0 * h * k * sin(L)), 
             r / s2 * (sin(L) - alpha2 * sin(L) + 2.0 * h * k * cos(L)),
             2.0 * r / s2 * (h * sin(L) - k * cos(L))]
    v_ECI = [- 1.0 / s2 * sqrt(mu_Earth / p) * (sin(L) + alpha2 * sin(L) - 2 * h * k * cos(L) + g - 2.0 * f * h * k + alpha2 * g),
             - 1.0 / s2 * sqrt(mu_Earth / p) * (- cos(L) + alpha2 * cos(L) + 2 * h * k * sin(L) - f + 2.0 * g * h * k + alpha2 * f),
             2.0 / s2 * sqrt(mu_Earth / p) * (h * cos(L) + k * sin(L) + f * h + g * k)]

    i_r = r_ECI ./ norm(r_ECI)
    i_n = cross(r_ECI, v_ECI) ./ norm(cross(r_ECI, v_ECI))
    i_t = cross(i_n, i_r)
    Mat = hcat([i_r i_t i_n])'
    
    r_ECEF = Tz(GMST(t, "IVP")) * r_ECI

    g_P_ECEF = geopotential_acceleration(4, C_gp, S_gp, Tuple(r_ECEF))
    g_P_ECI = Tz(GMST(t, "IVP"))' * g_P_ECEF
    g_P_EQ = Mat * g_P_ECI
    a_r = g_P_EQ[1] + mu_Earth / norm(r_ECI)^2
    a_t = g_P_EQ[2]
    a_n = g_P_EQ[3]
    
    dpdt = 2 * p / w * sqrt(p / mu_Earth) * a_t
    dfdt = sqrt(p / mu_Earth) * (a_r * sin(L) + ((w + 1) * cos(L) + f) * a_t / w - (h * sin(L) - k * cos(L)) * g * a_n / w)
    dgdt = sqrt(p / mu_Earth) * (- a_r * cos(L) + ((w + 1) * sin(L) + g) * a_t / w + (h * sin(L) - k * cos(L)) * f * a_n / w)
    dhdt = sqrt(p / mu_Earth) * s2 * a_n / 2 / w * cos(L)
    dkdt = sqrt(p / mu_Earth) * s2 * a_n / 2 / w * sin(L)
    dLdt = sqrt(mu_Earth * p) * (w / p)^2 + 1 / w * sqrt(p / mu_Earth) * (h * sin(L) - k * cos(L)) * a_n

    return [dpdt, dfdt, dgdt, dhdt, dkdt, dLdt]
end