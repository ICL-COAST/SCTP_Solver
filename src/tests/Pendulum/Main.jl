#= Performance analysis for pendulum test case 
    We assume g = l = 1 for the nonlinear pendulum. =#
# See https://math.libretexts.org/Bookshelves/Differential_Equations/A_First_Course_in_Differential_Equations_for_Scientists_and_Engineers_(Herman)/07%3A_Nonlinear_Systems/7.09%3A_The_Period_of_the_Nonlinear_Pendulum for nonlinear pendulum period.

using DifferentialEquations, LSODA
using BenchmarkTools

# Define ODE system
function pendulum!(du, u, p, t)
    θ  = u[1]
    dθ = u[2]

    du[1] = dθ
    du[2] = -sin(θ)

    nothing
end

# Compute Hamiltonian
Ham = (θ, dθ) -> 0.5 * dθ^2 - cos(θ)

# Initial conditions (from Wang et al paper)
u0 = [3.13, 0.0]
tspan = (0.0, 2500.0)

#= FINITE DIFFERENCES =#
# Generate reference solution

# Tsit5
prob = ODEProblem(pendulum!, u0, tspan)
@btime sol = solve(prob, Tsit5(), reltol = 1e-8, abstol = 1e-8)
FEs = sol.destats

#= GCN =#


#= MCPI =#


# Plots (preliminary)
using Plots
times = tspan[1]:0.25:tspan[2]
θs = sol(times, idxs = 1)
dθs = sol(times, idxs = 2)

# Evaluate Hamiltonian on the solution
H0 = Ham(u0[1],u0[2])
Hs = Ham.(θs.u,dθs.u)

# Time and phase plot
p1 = plot(times, θs.u, legend = false)
xlabel!("t")
ylabel!("θ")
display(p1)

p2 = plot(θs.u, dθs.u, legend = false)
xlabel!("θ")
ylabel!("dθ/dt")
display(p2)

# Hamiltonian error plot
p3 = plot(times, log10.(abs.(H0 .- Hs) ./ Hs), legend = false)
xlabel!("t")
ylabel!("δH")
display(p3)