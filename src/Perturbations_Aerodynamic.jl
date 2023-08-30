using SatelliteToolbox


function atmosphere_model_NRLMSISE00(variables)

    year = convert(Int64, variables[1])
    month = convert(Int64, variables[2])
    day = convert(Int64, variables[3])
    hour = convert(Int64, variables[4])
    minute = convert(Int64, variables[5])
    second = variables[6]
    j_day = date_to_jd(year, month, day, hour, minute, second)
    altitude = variables[7]                                             # altitude in m
    latitude = variables[8]                                 # geodetic latitude in degrees
    longitude = variables[9]                                # geodetic longitude in degrees
    f107A = variables[10]                                                # 81 day average f10.7 flux around the specified year
    f107 = variables[11]                                                # daily f10.7 flux for previous day
    ap = variables[12]                                                  # magnetic index
    outputs = nrlmsise00(j_day, altitude, latitude, longitude, f107A, f107, ap, output_si = true,
    dversion = true)
    den_N, den_N2, den_O, den_aO, den_O2, den_H, den_He, den_Ar, 
    density, T_exo, T_alt = outputs.den_N, outputs.den_N2, outputs.den_O, outputs.den_aO, outputs.den_O2, outputs.den_H, outputs.den_He, outputs.den_Ar, outputs.den_Total, outputs.T_exo, outputs.T_alt
    R_gas = R / (den_N * mu_N.^2 + den_N2 * mu_N2.^2 + den_O * mu_O.^2 + den_aO * mu_AO.^2 + den_O2 * mu_O2.^2 + den_H * mu_H.^2 + den_He * mu_He.^2 + den_Ar * mu_Ar.^2) * Na * density
    pressure = density * R_gas * T_alt
    return [den_N, den_N2, den_O, den_aO, den_O2, den_H, den_He, den_Ar, density, pressure, T_alt, T_exo]
end

function aero_acceleration(position_ECEF, velocity_ECEF, mass, Cd, area, time)

    Omega_E = [0.0, 0.0, pi * 2]
    F107 = 72.1
    F107A = 62.1
    ap = 15.1

    start_date = DateTime(start_year, start_month, start_day, start_hour, start_minute, start_second)
    delta_t = Second(trunc(Int64, time * norm_time_E))
    current_date = start_date + delta_t
    y = Dates.year(current_date)
    m = Dates.month(current_date)
    d = Dates.day(current_date)
    h = Dates.hour(current_date)
    min = Dates.minute(current_date)
    s = Dates.second(current_date)

    r = norm(position_ECEF)
    latitude = asin(position_ECEF[3] / r)
    longitude = atan(position_ECEF[2], position_ECEF[1])
    altitude = (r - R_Earth) * norm_distance_E * 1000

    atmosphere = atmosphere_model_NRLMSISE00([y, m, d, h, min, s, altitude, latitude, longitude, F107A, F107, ap])
    rho = atmosphere[9]
    v_rel = velocity_ECEF - cross(Omega_E, position_ECEF)
    a_drag = - 0.5 * rho * Cd * area * norm(v_rel) * v_rel / mass .* norm_time_E .^ 2 ./ norm_distance_E ./ 1000
    return a_drag
end