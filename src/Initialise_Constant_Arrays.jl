#=
#create legendere and chebyshev based quadrature nodes and weights
const gl_nodes_IVP, gl_weights_IVP = build_quadrature(s_order_quad, "Legendre");
const gc_nodes_IVP, gc_weights_IVP = build_quadrature(s_order_quad, "Chebyshev");

#create and evaluate quadrature polynomial functions
const spectral_basis_gl_IVP, spectral_basis_deriv_gl_IVP, spectral_basis_deriv2_gl_IVP = create_basis_set(gl_nodes_IVP, s_order_IVP, "Chebyshev");
const spectral_basis_gc_IVP, spectral_basis_deriv_gc_IVP, spectral_basis_deriv2_gc_IVP = create_basis_set(gc_nodes_IVP, s_order_IVP, "Chebyshev");


const sys_matrix_IVP = build_system_matrix_IVP(sys_size); # P - Br returned
const sys_matrix_IVP_inv = inv(sys_matrix_IVP); # pre-compute P - Br inverse

t_nums, sol_nums, num_b_times = [], [], []
t_specs, c_specs, e_mats, i_mats, spec_b_times = [], [], [], [], []

#counter for function calls, check destats.nf is accurate
global function_calls = 0

=#

#create legendere and chebyshev based quadrature nodes and weights
global gl_nodes_IVP, gl_weights_IVP = build_quadrature(s_order_quad, "Legendre");
global gc_nodes_IVP, gc_weights_IVP = build_quadrature(s_order_quad, "Chebyshev");

#create and evaluate quadrature polynomial functions
global spectral_basis_gl_IVP, spectral_basis_deriv_gl_IVP, spectral_basis_deriv2_gl_IVP = create_basis_set(gl_nodes_IVP, s_order_IVP, "Chebyshev");
global spectral_basis_gc_IVP, spectral_basis_deriv_gc_IVP, spectral_basis_deriv2_gc_IVP = create_basis_set(gc_nodes_IVP, s_order_IVP, "Chebyshev");


global sys_matrix_IVP = build_system_matrix_IVP(sys_size); # P - Br returned
global sys_matrix_IVP_inv = inv(sys_matrix_IVP); # pre-compute P - Br inverse

t_nums, sol_nums, num_b_times = [], [], []
t_specs, c_specs, e_mats, i_mats, spec_b_times = [], [], [], [], []

#counter for function calls, check destats.nf is accurate
global function_calls = 0