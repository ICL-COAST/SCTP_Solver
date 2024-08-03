using LinearAlgebra, CPUTime, Plots
using Logging: global_logger
using TerminalLoggers: TerminalLogger
global_logger(TerminalLogger())


function Cowell(x, p, t)
    global function_calls

    rx, ry, rz, vx, vy, vz = x[1], x[2], x[3], x[4], x[5], x[6]

    r_ECI = [rx, ry, rz]
    r_ECEF = Tz(GMST(t, "IVP")) * r_ECI

    v_ECI = [vx, vy, vz]
    v_ECEF = Tz(GMST(t, "IVP")) * v_ECI

    if geopotential
        g_ECEF = geopotential_acceleration(geopotential_order, C_gp, S_gp, Tuple(r_ECEF))
        g_ECI = Tz(GMST(t, "IVP"))' * g_ECEF
    else
        g_ECEF = geopotential_acceleration(1, C_gp, S_gp, Tuple(r_ECEF))
        g_ECI = Tz(GMST(t, "IVP"))' * g_ECEF
    end

    if aero_drag
        a_drag_ECEF = aero_acceleration(r_ECEF, v_ECEF, spacecraft_mass, spacecraft_Cd, spacecraft_area, t)
        a_drag_ECI = Tz(GMST(t, "IVP"))' * a_drag_ECEF
    else
        a_drag_ECI = [0 0 0]
    end
   
    dvxdt = g_ECI[1] + a_drag_ECI[1]
    dvydt = g_ECI[2] + a_drag_ECI[2]
    dvzdt = g_ECI[3] + a_drag_ECI[3]
    drxdt = vx
    drydt = vy
    drzdt = vz
    
    function_calls += 1

    return [drxdt, drydt, drzdt, dvxdt, dvydt, dvzdt]
end

function Cowell_simple(x, p, t)
    global function_calls
    
    rx, ry, rz, vx, vy, vz = x[1], x[2], x[3], x[4], x[5], x[6]

    r_ECI = [rx, ry, rz]
    r_ECEF = Tz(GMST(t, "IVP")) * r_ECI

    g_ECEF = geopotential_acceleration(4, C_gp, S_gp, Tuple(r_ECEF))
    g_ECI = Tz(GMST(t, "IVP"))' * g_ECEF
    
    dvxdt = g_ECI[1]
    dvydt = g_ECI[2]
    dvzdt = g_ECI[3]
    drxdt = vx
    drydt = vy
    drzdt = vz

    function_calls += 1
    return [drxdt, drydt, drzdt, dvxdt, dvydt, dvzdt]
end