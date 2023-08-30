using Dates

function Tx(angle)
    return [1.0 0.0 0.0; 0.0 cos.(angle) sin.(angle); 0.0 -sin.(angle) cos.(angle)]
end

function Ty(angle)
    return [cos.(angle) 0.0 -sin.(angle); 0.0 1.0 0.0; sin.(angle) 0.0 cos.(angle)]
end

function Tz(angle)
    return [cos.(angle) sin.(angle) 0; -sin.(angle) cos.(angle) 0.0; 0.0 0.0 1.0]
end

function kepler_to_cartesian(elements)

    a, e, i, RAAN, omega, theta = elements
    h = (mu_Earth * a * (1 - e^2))^0.5
    r = a * (1 - e.^2) / (1 + e .* cos(theta))
    r_perifocal = [r * cos(theta); r * sin(theta); 0.0]
    v_perifocal = [-sin(theta); e + cos(theta); 0.0] .* mu_Earth ./ h
    r_ECI = (Tz(omega) * Tx(i) * Tz(RAAN))' * r_perifocal
    v_ECI = (Tz(omega) * Tx(i) * Tz(RAAN))' * v_perifocal
    return r_ECI', v_ECI'
end

function kepler_to_cartesian_array(kepler_array)

    N = size(kepler_array)[1]
    elements_cartesian = zeros(N, size(kepler_array)[2])

    for i in 1 : N
        elements_cartesian[i, 1:3], elements_cartesian[i, 4:6] = kepler_to_cartesian(kepler_array[i, :])
    end

    return elements_cartesian
end 

function cartesian_to_kepler(r, v)
    tolerance = 1e-15
    x, y, z = r
    r_n = norm(r)
    v_n = norm(v)
    a = mu_Earth / 2 / (mu_Earth / r_n - v_n^2 / 2)
    h = cross(r, v)
    h_n = norm(h)
    hx, hy, hz = h
    i = acos(hz / h_n)
    e_v = cross(v, h) ./ mu_Earth - r ./ r_n
    e = norm(e_v)
    N = cross([0, 0, 1], h)
    N_n = norm(N)

    if abs(e) < tolerance && abs(i) < tolerance
        RAAN = 0.0
        omega = 0.0
        theta = acos(x / r_n)
        if z < 0.0
            theta = 2 * pi - theta
        end
    elseif abs(e) < tolerance
        omega = 0.0
        RAAN = acos(N[1] / N_n)
        if N[2] < 0.0
            RAAN = 2 * pi - RAAN
        end
        theta = acos(min(1, max(-1, dot(r, N) / N_n / r_n)))
        if y < 0.0
            theta = 2 * pi - theta
        end
    elseif abs(i) < tolerance
        RAAN = 0.0
        omega = acos(min(1, max(-1, e_v[1] / e)))
        theta = acos(min(1, max(-1, dot(r, e_v) / r_n / e)))
        if dot(r, v) < 0.0
            theta = 2 * pi - theta
        end
    else
        omega = acos(min(1, max(-1, dot(N, e_v) / N_n / e)))
        if e_v[3] < 0.0
            omega = 2 * pi - omega
        end
        RAAN = acos(min(1, max(-1, N[1] / N_n)))
        if N[2] < 0.0
            RAAN = 2 * pi - RAAN
        end
        theta = acos(min(1, max(-1, dot(r, e_v) / r_n / e)))
        if dot(r, v) < 0.0
            theta = 2 * pi - theta
        end
    end
    return [a, e, i, RAAN, omega, theta]
end

function cartesian_to_kepler_array(cartesian_array)

    N = size(cartesian_array)[1]
    elements_kepler = zeros(Float64, N, size(cartesian_array)[2])

    for i in 1 : N
        elements_kepler[i, :] = cartesian_to_kepler(cartesian_array[i, 1:3], cartesian_array[i, 4:6])
    end

    return elements_kepler
end 

function kepler_to_equinoctial(elements)

    a, e, i, RAAN, omega, theta = elements[1], elements[2], elements[3], elements[4], elements[5], elements[6]
    p = a * (1 - e^2)
    f = e * cos(omega + RAAN)
    g = e * sin(omega + RAAN)
    h = tan(i / 2) * cos(RAAN)
    k = tan(i / 2) * sin(RAAN)
    L = RAAN + omega + theta

    return [p, f, g, h, k, L]
end

function equinoctial_to_kepler(elements)
    p, f, g, h, k, L = elements[1], elements[2], elements[3], elements[4], elements[5], elements[6]

    a = p / (1 - f^2 - g^2)
    e = sqrt(f^2 + g^2)
    i = atan(2 * sqrt(h^2 + k^2), 1 - h^2 - k^2)
    RAAN = atan(k, h)
    omega = atan(g * h - f * k, f * h + g * k)
    u = atan(h * sin(L) - k * cos(L), h * cos(L) + k * sin(L))
    theta = u - omega

    return [a, e, i, RAAN, omega, theta]

end

function equinoctial_to_cartesian(elements)

    elements = equinoctial_to_kepler(elements)
    r, v = kepler_to_cartesian(elements)
    return r, v
end

function equinoctial_to_kepler_array(elements_array)

    N = size(elements_array)[1]
    elements_kepler = zeros(eltype(elements_array), N, size(elements_array)[2])

    for i in 1 : N
        elements_kepler[i, :] = equinoctial_to_kepler(elements_array[i, :])
    end

    return elements_kepler
end 

function equinoctial_to_cartesian_array(elements_array)

    N = size(elements_array)[1]
    elements_cartesian = zeros(eltype(elements_array), N, size(elements_array)[2])

    for i in 1 : N
        elements = equinoctial_to_kepler(elements_array[i, :])
        elements_cartesian[i, 1:3], elements_cartesian[i, 4:6] = kepler_to_cartesian(elements)
    end

    return elements_cartesian
end 

function cartesian_to_equinoctial_array(elements_array)

    N = size(elements_array)[1]
    elements_equinoctial = zeros(eltype(elements_array), N, size(elements_array)[2])

    for i in 1 : N
        elements = cartesian_to_kepler(elements_array[i, 1:3], elements_array[i, 4:6])
        elements = kepler_to_equinoctial(elements)
        elements_equinoctial[i, :] = elements
    end

    return elements_equinoctial
end 

function GMST(t, problem)
    start_date = DateTime(start_year, start_month, start_day, start_hour, start_minute, start_second)
    if problem == "IVP"
        delta_t = Second(trunc(Int64, t * norm_time_E))
    else
        delta_t = Second(trunc(Int64, t * norm_time_S))
    end
    current_date = start_date + delta_t
    y = Dates.year(current_date)
    m = Dates.month(current_date)
    d = Dates.day(current_date)
    h = Dates.hour(current_date)
    min = Dates.minute(current_date)
    s = Dates.second(current_date)
    residual = t - trunc(Int64, t)
    s += residual
    J0 = 367 * y - trunc(Int64, 7 * (y + trunc(Int64, (m + 9) / 12) / 4)) + trunc(Int64, 275 * m / 9) + d + 1721013.5
    T0 = (J0 - 2451545) / 36525
    Theta_G0 = 100.4606184 + 36000.77004 * T0 + 0.000387933 * T0^2 - 2.583 * 10^(-8) * T0^3
    Theta_GMST = Theta_G0 + 360.98564724 * (h + min / 60 + s / 3600) / 24
    Theta_GMST -= trunc(Int64, Theta_GMST / 360) * 360
    return Theta_GMST * pi / 180
end
