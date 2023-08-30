#=
Please define the boundary-value problem inputs of AstroSIM in this file. 
The inputs can be categorized as: simulation parameters and boundary conditions. 
=#

# Simulation Boundary Conditions (Cartesian Position Coordinates) :

x_1       =         1.0        # [AU] - the x-axis start position
y_1       =         0.0        # [AU] - the y-axis start position
z_1       =         0.0        # [AU] - the z-axis start position

x_2       =         0.0         # [AU] - the x-axis end position
y_2       =         1.0        # [AU] - the y-axis end position
z_2       =         0.0        # [AU] - the z-axis end position


year_start    =       2020    # [years] - the initial time year
month_start   =       8       # [month] - the initial time month
day_start     =       20      # [days]  - the initial time day
hour_start    =       13      # [hours] - the initial time hour
minute_start  =       24      # [min]   - the initial time minute
second_start  =       10      # [sec]   - the initial time second

time_span_BVP =       (0.0, 2.5)      # [years] - the propagation time

# Physics Moded Parameters :

solar_radiation_pressure   =   true           # include SRP perturbations


# Spacecraft Parameters :

spacecraft_mass_BVP     =   0.4           # [kg] - spacecraft mass
spacecraft_area_BVP     =   4             # [m^2] - include atmospheric drag perturbations
spacecraft_CR_BVP       =   1.04          # [-] - the reflectivity coefficient

# Simulation Parameters :

s_order_BVP         =       20                              # [-] - spectral order of the polynomial approximation
m_order_BVP         =       20                             # [-] - MCPI order of the polynomial approximation 
spectral_tol_BVP    =       1e-13                           # [-] - relative tolerance of the spectral integrator
relative_tol_BVP    =       1e-14                           # [-] - relative tolerance of the adaptive spectral integrator
exact_time_step     =       1e-2                            # [years] - exact time step for numerical solver
num_time_step       =       1e-1                            # [years] - time step for numerical solver
spectral_iter_BVP   =       10000                           # [-] - maximum number of iterations of the spectral propagator