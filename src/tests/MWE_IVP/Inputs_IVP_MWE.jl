#=
Please define the initial-value problem inputs of AstroSIM in this file. 
The inputs can be categorized as: simulation parameters and initial conditions. 
=#

# Simulation Initial Conditions (Keplerian Elements) :

# Test case 1 from GCN paper
a       =       6678.0      # [km]  - semi-major axis   
e       =       0.0         # [-]   - eccentricity  
i       =       30.00       # [deg] - inclination
RAAN    =       0.00        # [deg] - right ascencion of ascending node
omega   =       0.00        # [deg] - argument of perigee
theta   =       10.0        # [deg] - true anomaly

start_year    =       2020    # [years] - the initial time year
start_month   =       8       # [month] - the initial time month
start_day     =       20      # [days]  - the initial time day
start_hour    =       13      # [hours] - the initial time hour
start_minute  =       24      # [min]   - the initial time minute
start_second  =       10      # [sec]   - the initial time second

# Physics Moded Parameters :

geopotential    =   true           # include geopotential perturbations
aero_drag       =   false           # include atmospheric drag perturbations
solar_pressure  =   false           # include solar radiation pressure perturbations
third_body      =   false           # include third-body perturbations from sun and moon

geopotential_order      =       2                 # order of harmonics in geopotential expansion
atmospheric_model       =       "NRLMSISE00"        # atmospheric model to be used ("exponential", "NRLMSISE00")

coefs_path = "/Users/davide/Documents/Research/SCTP_Solver/physical_data/EGM2008_to2190_TideFree"


# Spacecraft Parameters :

spacecraft_mass     =   0.4               # [kg] - spacecraft mass
spacecraft_area     =   4             # [m^2] - include atmospheric drag perturbations
spacecraft_Cd       =   2.2             # [-] - include solar radiation pressure perturbations



# Simulation Parameters :

global time_span_IVP       =       (0.0 , 2.0) .* 86400.0        # [seconds] - total integration time time_span
global s_interval_IVP      =       0.5 * 86400.0                  # [seconds] - the spectral time interval
global M                   =       300                             # [-] - GCN order of the polynomial approximation
global m_order_IVP         =       50                             # [-] - MCPI order of the polynomial approximation
global rel_tol             =       1e-13                           # [-] - relative tolerance of the RK45 integrator
global abs_tol             =       1e-13                           # [ER] - absolute tolerance of the RK45 integrator  
spectral_tol_IVP    =       1e-10                           # [-] - relative tolerance of the spectral integrator
relative_tol_IVP    =       1e-12                          # [-] - relative tolerance of the adaptive spectral integrator
spectral_iter_IVP   =       10000                           # [-] - maximum number of iterations of the spectral propagator
h_adaptive      =       false