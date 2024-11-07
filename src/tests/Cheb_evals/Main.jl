#= Check Chebyshev expansion =#
# function build_quadrature(order::Int, type::String)

#     if type == "Legendre"
#         nodes, weights = FastGaussQuadrature.gausslobatto(order + 1)
#     elseif type == "Chebyshev"
#         index = 1:M
#         nodes = cos.((index .* 2 .+ 1) ./ 2 ./ (order + 1) .* pi)
#         weights = ones(order + 1) .* pi / (order + 1)
#     elseif type == "Chebyshev_L"
#         index = LinRange(order, 0, order + 1)
#         nodes = cos.(index / order .* pi)
#         weights = ones(order + 1) .* pi / (order + 1)
#     end
#     return nodes, weights
# end
function GC_quad(Q)
    #= Return nodes and weights for Chebyshev quadrature of order Q =#
        τk = [cos((2k - 1)*π / (2Q)) for k = 1:Q ]
        wk = π/Q

    return τk, wk
end

function T_eval(τ, M)
    #= Evaluate Chebyshev polynomials from order 0 to M at τ ∈ [-1, 1] =#

    # Columns are orders, rows are epochs
    Ts = zeros(Float64, M + 1, size(τ)[1])

    # Initialise recursions
    Ts[1,:] .= 1.0      # Order 0
    Ts[2,:] .= τ        # Order 1

    for k = 3:M+1
        Ts[k, :] .= 2.0.*τ .* Ts[k-1,:] - Ts[k-2,:]

    end

    return Ts
end

function coef_analytical(l)
    #= l-th coefficient of the Chebyshev expansion of √(1-x^2). See Mason and Handscomb (2003), Example 5.1 =#

    if l % 2 == 0
        coef = -4/π * (1/(l^2 - 1))
    else
        coef = 0
    end

end

function chebExp_eval(Cs,Ts)
    #= Evaluate the Chebyshev expansion with coefficient array C at time τ =#

    M = size(Cs)[1] - 1   # Order of the expansion
    fvals = zeros(size(Ts)[2])
    for k = M:-1:0
        if k !== 0
            fvals += Cs[k+1] .* Ts[k+1,:]
        else
            fvals += 0.5 * Cs[k+1] .* Ts[k+1,:]
        end

    end

    return fvals
end

function create_basis_set(tau, order, type)

    #=
    This function computes the first [order + 1] basis functions of the Legendre polynomial basis, as well as their derivatives
    at the time instances [tau].
    
    Parameters:
    - tau : 1D array of scaled time instances [-1, 1]
    - order : the maximum order of the polynomial expansions
    - type : to be implemented (function will have multiple choices of basis expansions)
    =#
    if type == "Legendre"
        basis_array = zeros(Float64, order + 1, size(tau)[1])
        basis_array[1, :] .= 1.0
        basis_array[2, :] .= tau

        for i in 3 : order + 1
            basis_array[i, :] .= 1. ./ (i - 1) .* ((2 * i - 3) .* tau .* basis_array[i - 1, :] .- (i - 2) * basis_array[i - 2, :])
        end

        basis_array_deriv = zeros(Float64, order + 1, size(tau)[1])
        basis_array_deriv[2, :] .= 1.0

        for i in 3 : order + 1
            basis_array_deriv[i, :] = (i - 1) .* basis_array[i - 1, :] .+ tau .* basis_array_deriv[i-1, :]
        end

        basis_array_deriv2 = zeros(Float64, order + 1, size(tau)[1])

        for i in 3 : order + 1
            basis_array_deriv2[i, :] = (i + 1) .* basis_array_deriv[i - 1, :] .+ tau .* basis_array_deriv2[i-1, :]
        end

    elseif type == "Chebyshev"
        basis_array = zeros(Float64, order + 1, size(tau)[1])
        basis_array[1, :] .= 1.0
        basis_array[2, :] .= tau

        for i in 3 : order + 1
            basis_array[i, :] .= 2 .* tau .* basis_array[i - 1, :] .- basis_array[i - 2, :]
        end

        basis_array_deriv = zeros(Float64, order + 1, size(tau)[1])
        basis_array_deriv[2, :] .= 1.0
        basis_array_deriv[3, :] .= 4.0 .* tau

        for i in 4 : order + 1
            basis_array_deriv[i, :] = (i - 1) ./ (i - 2) .* 2 .* tau .* basis_array_deriv[i - 1, :] .- (i - 1) ./ (i - 3) .* basis_array_deriv[i - 2, :]
        end

        basis_array_deriv2 = zeros(Float64, order + 1, size(tau)[1])
        basis_array_deriv2[3, :] .= 4.0 

        for i in 4 : order + 1
            basis_array_deriv2[i, :] = (i - 1) ./ (i - 2) .* 2 .* basis_array_deriv[i - 1, :] .+ (i - 1) ./ (i - 2) .* 2 .* tau .* basis_array_deriv2[i - 1, :] .- (i - 1) ./ (i - 3) .* basis_array_deriv2[i - 2, :]
        end
    end

    return basis_array, basis_array_deriv, basis_array_deriv2
end 

M = 100;
func = (τ) -> √(1.0 - τ^2)
coefs = zeros(Float64,M+1)

# Generate coefficients of Chebyshev expansion of M-th order
τk, wk = GC_quad(M)
Tik = T_eval(τk, M)

# Evaluate function at τk
FEs = func.(τk)

for i = 0:M
    coefs[i+1] = 2/M * sum(FEs .* Tik[i+1,:])

end

# Compare to analytical result for coefficients of √(1-τ^2)
coefs_true = coef_analytical.(0:M)

# # Spectral error
# Δc = abs.((coefs .- coefs_true)./coefs_true)

# using Plots
# plotlyjs()

# # Plot spectrum
# p_spectrum = scatter(0:M,log10.(abs.(coefs)), label = "Quad, M = $M", ylim = (-5,0.5))
# scatter!(0:M,log10.(abs.(coefs_true)), label = "True")
# ylabel!("log10(c_i)")
# display(p_spectrum)

# p_error = plot(0:2:M,log10.(Δc[1:2:end]), legend = false, xlim = (0, 50), title = "M = $M, first 50 coefs")
# ylabel!("log10(Δc)")
# display(p_error)

# Accuracy of spectral representation

# True solution
τ_eval = LinRange(-1,1,200)
f_true = func.(τ_eval)

# Chebyshev representation
Tcheb = T_eval(τ_eval,M)
f_Cheb = chebExp_eval(coefs,Tcheb)

# Chebyshev representation with analytical coefficients
f_Cheb_true = chebExp_eval(coefs_true,Tcheb)

# Plot results
using Plots
p_sols = plot(τ_eval,f_true, label = "True", ylim = (0.0, 1.1), aspect_ratio = :equal)
plot!(τ_eval,f_Cheb, label = "Cheb, M = $M")
xlabel!("τ")
ylabel!("√(1-τ²)")

# Error between function and spectral representation: spectral coefficients from numerical quadrature VS analytical coefficients
δf = abs.(f_Cheb - f_true)
δf_true = abs.(f_Cheb_true - f_true)

δf_RMS = √(sum( [δf[i]^2 for i in eachindex(δf) ] )/length(δf))
δf_RMS_true = √(sum( [δf_true[i]^2 for i in eachindex(δf_true) ] )/length(δf_true))

using Printf

tstring = @sprintf("M = %d, RMS = %.3g, RMS_true = %.3g",M,δf_RMS,δf_RMS_true)
p_err = plot(τ_eval,log10.(δf), label = "Quad")
plot!(τ_eval,log10.(δf_true), label = "True", title = tstring)
xlabel!("τ")
ylabel!("log10(δf)")
