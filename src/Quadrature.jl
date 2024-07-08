function build_quadrature(order, type)

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
