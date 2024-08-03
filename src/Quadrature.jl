function build_quadrature(order, type)
#this function constructs the basis for different quadratures
#it outputs the nodes (roots of the i_th quadrature polynomial), the nodes are times tau
#and it outputs the weights used for the quadrature itself

    if type == "Legendre"
        nodes, weights = gausslobatto(order + 1)

    elseif type == "Chebyshev" # of the 1st kind
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
