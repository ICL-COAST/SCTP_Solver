
const norm_distance_E = 6378
const norm_time_E = 86164.0905 #one solar day in seconds

const norm_distance_S = 1.495978707 * 10 ^ 11
const norm_time_S = 365.25 * 24 * 3600

const R_Earth = 6378  / norm_distance_E                             # [km] - Earth Radius
const mu_Earth = 398600 / norm_distance_E ^ 3 * norm_time_E ^ 2       # [km^3/s^2] - Earth Gravitational Parameter
const sid_day = 86164.0905 / norm_time_E                            # [seconds] - sidereal day

const mu_Sun = 1.32712440018 * 10^20  / norm_distance_S ^ 3 * norm_time_S ^ 2
const S_Sun = 1367
const c = 2.998 * 10 ^ 8

const R = 8314.46261815324
const kB = 1.38e-23
const mu_O = 15.9994
const mu_AO = 17.999
const mu_O2 = 2 * mu_O
const mu_H = 1.00784
const mu_He = 4.002602
const mu_N = 14.0067
const mu_N2 = 2 * mu_N
const mu_Ar = 39.948
const Na = 6.023e26