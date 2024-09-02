using FastGaussQuadrature

function build_quadrature(order, type)
    #= This function computes nodes and weights for various quadrature schemes.=#

    if type == "Legendre"
        nodes, weights = gausslobatto(order + 1)
    elseif type == "Chebyshev"
        index = LinRange(order, 0, order + 1)
        nodes = cos.((index .* 2 .+ 1) ./ 2 ./ (order + 1) .* pi)
        weights = ones(order + 1) .* pi / (order + 1)
    elseif type == "Chebyshev_L"
        index = LinRange(order, 0, order + 1)
        nodes = cos.(index / order .* pi)
        weights = ones(order + 1) .* pi / (order + 1)
    end
    return nodes, weights
end

M = 50
nodes_GC, weights_GC = build_quadrature(M-1,"Chebyshev")

# Compare to Ch. 1 of Trivlin
ξ = Vector{Float64}(undef,M)
λ = ones(M)

# Fill in nodes ξ and weights λ
for j in 1:M
    ξ[j] = cos( ((2j - 1)/M) * (π/2) )
    
end
reverse!(ξ)
λ = (π/M) * λ

# === Gauss-Legendre or Gauss-Lobatto? ===#
nodes_GL, weights_GL = build_quadrature(3,"Legendre")

sleep(0.01)
