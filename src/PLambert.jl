
function Cowell_Lambert(x, p, t)

    rx, ry, rz = x[1], x[2], x[3]

    r_I = [rx, ry, rz]

    g_I = - mu_Sun / norm(r_I) ^3 .* r_I

    a_SRP = srp_canonball_acceleration(r_I, spacecraft_mass_BVP, spacecraft_CR_BVP, spacecraft_area_BVP)

    drxdt2 = g_I[1] + a_SRP[1]
    drydt2 = g_I[2] + a_SRP[2]
    drzdt2 = g_I[3] + a_SRP[3]


    return [drxdt2, drydt2, drzdt2]
end