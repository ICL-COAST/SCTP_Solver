

function srp_canonball_acceleration(r_I, mass, Cr, area)

    x, y, z = r_I
    r = norm(r_I)

    a_srp = S_Sun / c * Cr * area / mass .* r_I ./ r .^3 .* norm_time_S .^2 ./ norm_distance_S

    return a_srp

end