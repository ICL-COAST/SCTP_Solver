using DelimitedFiles
using Plots
using LinearAlgebra


__precompile__()

function PI(order)

    pi_norm = zeros(Float64, order + 1, order + 1)

    for i in 1 : order + 1
        for j in 1 : i
            l = i - 1
            m = j - 1
            if m == 0
                pi_norm[i, j] = (2 * l + 1) ^ 0.5
            elseif m == 1
                pi_norm[i, j] = pi_norm[i, j - 1] * (1 / (l - m + 1) / (l + m) * 2)^0.5
            else
                pi_norm[i, j] = pi_norm[i, j - 1] * (1 / (l - m + 1) / (l + m))^0.5
            end
        end
    end

    return pi_norm
end

function import_coefficients(file, order)

    #=
    Returns the C and S geopotential normalised harmonic coefficients, from EGM2008
    =#

    println("Opening coefficient file ...")
    f = open(file, "r")
    C = zeros(Float64, (order + 1, order + 1))
    S = zeros(Float64, (order + 1, order + 1))
    println("Importing coefficients ...")
    lines = readlines(f)
    for i in 1 : order * (order - 1)
        string_array = split(lines[i], ' ')
        list_array = []
        for str in string_array
            if str != ""
                str = replace(str, "D" => "E")
                push!(list_array, parse(Float64, str))
            end
        end
        index1 = floor(Int64, list_array[1]) + 1
        index2 = floor(Int64, list_array[2]) + 1
        if index1 <= order + 1 && index2 <= order + 1
            C[index1, index2] = list_array[3]
            S[index1, index2] = list_array[4]
        end
    end
    C[1, 1] = 1.0
    println("Coefficients imported!")
    return C, S
end

function legendre_coefficients(order, r_ECEF :: NTuple{3})

    V = zeros(eltype(r_ECEF), order + 1, order + 1)
    W = zeros(eltype(r_ECEF), order + 1, order + 1)
    x, y, z = r_ECEF
    r = (x ^ 2 + y ^ 2 + z ^ 2) ^ 0.5
    R_E = R_Earth  

    V[1, 1] = R_E / r
    V[2, 1] = R_E ^ 2 / r ^ 2 * z / r

    for i in 2 : order + 1
        for j in 1 : i
            l = i - 1
            m = j - 1
            if l == m
                ratio = sqrt((2 * l + 1) / (2 * l - 1) / (l + m) / (l + m - 1))
                if m == 1
                    ratio *= sqrt(2)
                end
                V[i, j] = (2 * l - 1) * ((x * R_E / r ^ 2) * V[i - 1, j - 1] * ratio - (y * R_E / r ^ 2) * W[i - 1, j - 1] * ratio)
                W[i, j] = (2 * l - 1) * ((x * R_E / r ^ 2) * W[i - 1, j - 1] * ratio + (y * R_E / r ^ 2) * V[i - 1, j - 1] * ratio)
            elseif l == m + 1
                ratio = sqrt((2 * l + 1) / (2 * l - 1) / (l + m) * (l - m))
                V[i, j] = (2 * l - 1) / (l - m) * (z * R_E / r ^ 2) * V[i - 1, j] * ratio
                W[i, j] = (2 * l - 1) / (l - m) * (z * R_E / r ^ 2) * W[i - 1, j] * ratio
            else
                ratio1 = sqrt((2 * l + 1) / (2 * l - 1) / (l + m) * (l - m))
                ratio2 = sqrt((2 * l + 1) / (2 * l - 3) / (l + m) / (l + m - 1) * (l - m) * (l - m - 1))
                V[i, j] = (2 * l - 1) / (l - m) * (z * R_E / r ^ 2) * V[i - 1, j] * ratio1 - (l + m - 1) / (l - m) * (R_E ^ 2 / r ^ 2) * V[i - 2, j] * ratio2
                W[i, j] = (2 * l - 1) / (l - m) * (z * R_E / r ^ 2) * W[i - 1, j] * ratio1 - (l + m - 1) / (l - m) * (R_E ^ 2 / r ^ 2) * W[i - 2, j] * ratio2
            end
        end
    end

    return V, W
end

function geopotential_acceleration(order, C, S, r_ECEF :: NTuple{3})
    mu = mu_Earth
    R_E = R_Earth
    x, y, z = r_ECEF
    gx, gy, gz = 0.0, 0.0, 0.0
    V, W = legendre_coefficients(order, r_ECEF)

    for i in order : -1 : 1
        for j in i : -1 : 1
            l = i - 1
            m = j - 1
            ratio1 = sqrt((2 * l + 1) / (2 * l + 3) * (l + m + 1) * (l + m + 2))
            ratio2 = sqrt((2 * l + 1) / (2 * l + 3) * (l + m + 1) / (l - m + 1))
            ratio3 = sqrt((2 * l + 1) / (2 * l + 3) / (l - m + 1) / (l - m + 2))
            if m == 0
                ratio1 /= sqrt(2)
            elseif m == 1
                ratio3 *= sqrt(2)
            end
            if m == 0
                gx -= C[i, 1] * V[i + 1, 2] * ratio1
                gy -= C[i, 1] * W[i + 1, 2] * ratio1
            end
            if l > 0 && m > 0
                gx += 0.5 * (- C[i, j] * V[i + 1, j + 1] * ratio1 - 
                        S[i, j] * W[i + 1, j + 1] * ratio1)  
                gy += 0.5 * (- C[i, j] * W[i + 1, j + 1] * ratio1 + 
                        S[i, j] * V[i + 1, j + 1] * ratio1)   
            end
            gz += (l - m + 1) * (- C[i, j] * V[i + 1, j] * ratio2 - S[i, j] * W[i + 1, j] * ratio2)
            if m > 0
                gx += (l - m + 2) * (l - m + 1) * (C[i, j] * V[i + 1, j - 1] * ratio3 + 
                        S[i, j] * W[i + 1, j - 1] * ratio3)
                gy += (l - m + 2) * (l - m + 1) * (- C[i, j] * W[i + 1, j - 1] * ratio3 + 
                        S[i, j] * V[i + 1, j - 1] * ratio3)
            end
        end
    end
    gx, gy, gz = gx * mu / R_E^2, gy * mu / R_E^2, gz * mu / R_E^2

    return [gx, gy, gz]
end
