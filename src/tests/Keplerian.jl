using LinearAlgebra
using Test

function keplProp(x0::Vector{Float64},Δt::Float64,μ::Float64=1.0)
    #=
    Propagate initial conditions x0 = [r0, v0] through a time interval Δt and assuming gravitational 
    parameter μ according to Keplerian motion.
    =#

    # Transform to orbital elements and compute mean anomaly
    r0 = x0[1:3]
    v0 = x0[4:6]
    coe0 = cart2coe(r0, v0, μ, "mean", "rad")

    # Advance mean anomaly
    n = √(μ/coe[1]^3)
    Mf = mod(coe0[6] + n*Δt, 2π)

    # Transform back to position and velocity


end

function cart2coe(r::Vector{Float64}, v::Vector{Float64}, μ::Float64=1.0,
    anomalyType::String="true", angles::String="deg")
    #=
    Transform position and velocity r0, v0 into orbital elements with gravitational 
    parameter μ.
    =#

    # Eccentricity
    rMag = norm(r)
    hVec = r × v
    nVec = [0.; 0.; 1.] × hVec

    eVec = ((v'*v - μ/rMag) .* r - (r'*v) .* v )./μ
    e = norm(eVec)
    nMag = norm(nVec)

    # SMA and semi-latus rectum
    ε = 0.5v'*v - μ/rMag

    if !isapprox(abs(ε),0.0)
        a = -0.5μ/ε
        p = a*(1-e^2)
        
    else
        # Parabolic orbit
        a = Inf
        p = hVec'*hVec/μ

    end

    # Inclination, RAAN, AoP, true anomaly
    hMag = norm(hVec)
    inc = acos(hVec[3]/hMag)
    Ω = acos(nVec[1]/nMag)
    if nVec[2] < 0.
        Ω = 2π - Ω

    end
    
    ω = acos((nVec'*eVec)/(nMag*e))
    if eVec[3] < 0.
        ω = 2π - ω
    end

    ν = acos((eVec'*r)/(e*rMag))
    if r'*v < 0.
        ν = 2π - ν
    end

    # Special cases
    tol = 2*eps(1.0)
    Equatorial = isapprox(abs(abs(hVec[3]/hMag)),1.0; atol=tol)
    Circular = isapprox(e,0.;atol=tol)
    if Equatorial
        # Elliptical, equatorial. AoP = true longitude of periapsis
        Ω = 0.
        ω = acos(eVec[1]/e)
        if eVec[2] < 0
            ω = 2π - ω
        end
    end

    if Circular
        # Circular, inclined. ν = argument of latitude.
        ω = 0.
        ν = acos((nVec'*r)/(nMag*rMag))
        if r[3] < 0
            ν = 2π - ν
        end
        M = ν

    end

    if Circular && Equatorial
        # ν = true longitude.
        Ω = 0.
        ω = 0.
        ν = acos(r[1]/rMag)
        if r[2] < 0
            ν = 2π - ν

        end

    end

    # Mean anomaly
    sν = sin(ν); cν = cos(ν)

    if e < 1
        # Elliptical
        sE = (sν * √(1.0 -e^2))/(1.0 + e*cν)
        cE = (e + cν)/(1.0 + e*cν)
        EA = atan(sE,cE)
        M = EA - e*sE

    else
        # Hyperbolic
        EA = 2 * atan( √( (e - 1.0)/(e + 1.0) ) * tan(0.5*ν) )  # Gudermannian anomaly
        M = e * tan(EA) - log( tan( 0.5*EA + 0.25π) )

    end

    # Wrap M to [0, 360] deg
    M = mod(M + 2π, 2π)

    anomalyType = lowercase(anomalyType)
    if anomalyType == "true"
        anomaly = ν

    elseif anomalyType == "eccentric"
        anomaly = EA

    elseif anomalyType == "mean"
        anomaly = M

    end

    angles = lowercase(angles)
    if angles == "rad"
        return [a; e; inc ; Ω ; ω ; anomaly]

    elseif angles == "deg"
        return [a; e; inc * 180.0/π; Ω * 180.0/π; ω * 180.0/π; anomaly * 180.0/π]

    end
    
end

# r = [1; 0; 0.0]; v = [0.0; 1.0; 0.0]; μ = 1.0;
# oEls = cart2coe(r,v,μ)

# #= TESTS - CART2COE =#
# # Circular Equatorial
# r = [1; 0; 0.0]; v = [0.0; 1.0; 0.0]
# @test isapprox(cart2coe(r,v),[1.0; 0.0; 0.0; 0.0; 0.0; 0.0])

# # Circular Inclined
# r = [1; 0; 0.0]; v = [0.0; 0.0; 1.0]
# @test isapprox(cart2coe(r,v),[1.0; 0.0; 90.0; 0.0; 0.0; 0.0])

# # Elliptical Equatorial
# r = [0.520680950221347; 0.189512367430012; 0.0]
# v = [-0.150932076454254; 1.60832061994676; 0.0]
# @test isapprox(cart2coe(r,v),[1.0; 0.5; 0.0; 0.0; 335.0; 45.0])

# # Elliptical Inclined
# r = [0.364516134427945; 0.175849716406476; 0.378455353131388]
# v = [-0.147960790329928; 1.60858057379873; -0.00720072017325568]
# @test isapprox(cart2coe(r,v),[1.0; 0.5; 45.0; 275.0; 60.0; 45.0])

#= TESTS - KEPLPROP =#
r0 = [0.0; 0.0; 1.0]; v0 = [0.0; -1.0; 0.0]; Δt = 1.0
x0 = [r0; v0]
keplProp(x0,Δt)