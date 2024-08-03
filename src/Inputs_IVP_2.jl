#=
Please define the initial-value problem inputs of AstroSIM in this file. 
The inputs can be categorized as: simulation parameters and initial conditions. 
=#

# Simulation Initial Conditions (Keplerian Elements) :

a       =       13356        # [km]  - semi-major axis   
e       =       0.50        # [-]   - eccentricity  
i       =       30.00        # [deg] - inclination
RAAN    =       0.00        # [deg] - right ascencion of ascending node
omega   =       0.00        # [deg] - argument of perigee
theta   =       10.0        # [deg] - true anomaly

orbital_period = round(2 * pi * sqrt((a^3)/3.986004418e5), digits = 8)

start_year    =       2020    # [years] - the initial time year
start_month   =       8       # [month] - the initial time month
start_day     =       20      # [days]  - the initial time day
start_hour    =       13      # [hours] - the initial time hour
start_minute  =       24      # [min]   - the initial time minute
start_second  =       10      # [sec]   - the initial time second

# Physics Moded Parameters :

geopotential    =   true           # include geopotential perturbations
aero_drag       =   true           # include atmospheric drag perturbations
solar_pressure  =   false           # include solar radiation pressure perturbations
third_body      =   false           # include third-body perturbations from sun and moon

geopotential_order      =       10                # order of harmonics in geopotential expansion
atmospheric_model       =       "NRLMSISE00"        # atmospheric model to be used ("exponential", "NRLMSISE00")


# Spacecraft Parameters :

spacecraft_mass     =   0.4               # [kg] - spacecraft mass
spacecraft_area     =   4             # [m^2] - include atmospheric drag perturbations
spacecraft_Cd       =   2.2             # [-] - include solar radiation pressure perturbations
