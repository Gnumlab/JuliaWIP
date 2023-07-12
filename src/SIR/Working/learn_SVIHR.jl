using DifferentialEquations
using SciMLSensitivity
using Statistics
using Flux

# Define the SIR model
function sir_model!(du, u, p, t)
    S, I, R = u
    β, γ = p

    N = S + I + R
    new_I = β*S*I / N  #÷ integer division for integer results
    new_R = γ*I

    du[1] = -new_I
    du[2] = new_I - new_R
    du[3] = new_R
end

# Set initial conditions and parameters
N = 1000       # total population
I0 = 10         # initial number of infected individuals
S0 = N - I0    # initial number of susceptible individuals
R0 = 0         # initial number of recovered individuals
u0 = [S0, I0, R0]
tspan = (0, 100)  # time span to simulate over
params = [0.3, 0.1] # β and γ parameters

# Define the ODE problem and solve it
prob = ODEProblem(sir_model!, u0, tspan, params)
sol = solve(prob, Tsit5())

# Plot the results
using Plots
plot(sol, xlabel="Time", ylabel="Population", label=["S" "I" "R"])








using DifferentialEquations

# Define the SVIHR model
function svihr_model!(du, u, p, t)
    S, V, I, H, R = u
    β, κ, η, ω1, ω2, μ, λ = p

    N = S + V + I + H + R
    dS = λ - β*S/N*(1 + κ)*I - (V + μ)*S
    dV = V*S - β*S/N*κ*I - μ*V
    dI = β*S/N*(1 + κ)*I - (η + ω1 + μ)*I
    dH = η*I - (ω2 + μ)*H
    dR = ω1*I + ω2*H - μ*R

    du[1] = dS
    du[2] = dV
    du[3] = dI
    du[4] = dH
    du[5] = dR
end

# Set initial conditions and parameters
N = 101     # total population
S0 = 90     # initial number of susceptible individuals
V0 = 1      # initial number of vaccinated individuals
I0 = 10       # initial number of infected individuals
H0 = 0          # initial number of hospitalized individuals
R0 = 0          # initial number of recovered individuals
u0 = [S0, V0, I0, H0, R0]
tspan = (0, 365)  # time span to simulate over
params = [0.3, 0.2, 0.1, 0.05, 0.05, 0.0, 0.0] # β, κ, η, ω1, ω2, μ, λ

# Define the ODE problem and solve it
prob = ODEProblem(svihr_model!, u0, tspan, params)
sol = solve(prob, Tsit5())

# Plot the results
using Plots
plot(sol, xlabel="Time", ylabel="Population", label=["S" "V" "I" "H" "R"])




using DifferentialEquations

# Define the SVIHR model with hospitalization and vaccination
function svihr_model!(du, u, p, t)
    S, V, I, H, R = u
    β, κ, ξ, T_I, M, Φ, μ, Λ, T_H = p

    N = S + V + I + H + R
    dS = Λ - β*S/N*I - (Φ + μ)*S        # Susceptible individuals
    dV = Φ*S - β*κ*S/N - μ*V            # Vaccinated individuals
    dI = β*(1 + κ)*S/N - (ξ/T_I + (1 - ξ)/T_I + μ)*I    # Infected individuals
    dH = ξ/T_I*I - (M/T_H + μ)*H         # Hospitalized individuals
    dR = (1 - ξ)/T_I*I + (1 - M)/T_H*H - μ*R   # Recovered individuals

    du[1] = dS
    du[2] = dV
    du[3] = dI
    du[4] = dH
    du[5] = dR
end

# Set initial conditions and parameters
N = 1000     # total population
S0 = 0.99N      # initial fraction of susceptible individuals
V0 = 0.01N      # initial fraction of vaccinated individuals
I0 = 0.001N     # initial fraction of infected individuals
H0 = 0.0        # initial fraction of hospitalized individuals
R0 = 0.0        # initial fraction of recovered individuals
u0 = [S0, V0, I0, H0, R0]
tspan = (0, 2*365)  # time span to simulate over
params = [0.3,     # Transmission rate (β)
          0.2,     # Vaccination effectiveness (κ)
          0.079718848,     # Hospitalization rate (ξ)
          1.42*7,      # Infectious period (T_I)
          0.026720524,     # Mortality rate (M)
          0.013517486,   # Vaccination rate (Φ)
          0.0,     # Natural death rate (μ)
          0.0,     # Birth rate (Λ)
          1.5*7]      # Hospitalization duration (T_H)

# Define the ODE problem and solve it
prob = ODEProblem(svihr_model!, u0, tspan, params)
sol = solve(prob, Tsit5(), dtmax=0.1, abstol=1e-8, reltol=1e-8)

# Plot the results
using Plots
plot(sol, xlabel="Time", ylabel="Population Fraction", label=["S" "V" "I" "H" "R"])













using Flux
using DifferentialEquations
using Optim


# Define the SVIHR model with hospitalization and vaccination
function svihr_model!(du, u, p, t)
    S, V, I, H, R = u
    β, κ, ξ, T_I, M, Φ, μ, Λ, T_H = p

    N = S + V + I + H + R
    dS = Λ - β*S/N*I - (Φ + μ)*S        # Susceptible individuals
    dV = Φ*S - β*κ*S/N - μ*V            # Vaccinated individuals
    dI = β*(1 + κ)*S/N - (ξ/T_I + (1 - ξ)/T_I + μ)*I    # Infected individuals
    dH = ξ/T_I*I - (M/T_H + μ)*H         # Hospitalized individuals
    dR = (1 - ξ)/T_I*I + (1 - M)/T_H*H - μ*R   # Recovered individuals

    du[1] = dS
    du[2] = dV
    du[3] = dI
    du[4] = dH
    du[5] = dR
end

# Generate training data
N = 1
S0 = 0.99N
V0 = 0.01N
I0 = 0.001N
H0 = 0.0
R0 = 0.0
u0 = [S0, V0, I0, H0, R0]
years = 1
tspan = (0.0, years * 365.0)
params_true = [0.3,     # Transmission rate (β)
          0.2,     # Vaccination effectiveness (κ)
          0.079718848,     # Hospitalization rate (ξ)
          1.42*7,      # Infectious period (T_I)
          0.026720524,     # Mortality rate (M)
          0.013517486,   # Vaccination rate (Φ)
          0.0,     # Natural death rate (μ)
          0.0,     # Birth rate (Λ)
          1.5*7]      # Hospitalization duration (T_H)

prob = ODEProblem(svihr_model!, u0, tspan, params_true)
t = range(tspan[1], tspan[2], length=100)
sol_actual = solve(prob, Tsit5(), saveat=t)
#x_data = [sol_true.t, sol_true.u[1,:], sol_true.u[2,:]]
#y_data = [sol_true.u[3,:]]

t_data = sol_actual.t
I_data = sol_actual[3, :] #.+ 0.01 .* randn(length(t_data))#sin.(I_data)
S_data = sol_actual[1, :] #.- I_data
R_data = sol_actual[5, :]
V_data = sol_actual[2, :]
H_data = sol_actual[4, :]
SVIHR_data = hcat(S_data, V_data, I_data, H_data, R_data)

# Initialize beta and kappa parameters randomly
beta = rand(Float64)
kappa = rand(Float64)


# Define the neural network
m = Chain(Dense(2, 250, relu), Dense(250, 250, relu), Dense(250, 2))  # 2 output nodes for the two parameters


# Define the loss function
function obj_func()
    ξ, T_I, M, Φ, μ, Λ, T_H = [0.079718848, 1.427, 0.026720524, 0.013517486, 0.0, 0.0, 1.57]
    β, κ = m([beta, kappa])
    #u0 = [0.99, 0.01, 0.0]
    p_pred = [β, κ, ξ, T_I, M, Φ, μ, Λ, T_H]
    prob_pred = ODEProblem(svihr_model!, u0, tspan, p_pred)
    sol_pred = solve(prob_pred, Tsit5(), saveat=0.1, maxiters=Int(1e8))
    SVIHR_pred = [sol_pred[1, :], sol_pred[2, :], sol_pred[3, :], sol_pred[4, :], sol_pred[5, :]]
    loss_fn(SVIHR_pred, SVIHR_data)
end


# Define the loss function for the physical-informed neural network
function loss_fn_(SVIHR_pred, SVIHR_actual)
    λ = 0.01
    S_pred, V_pred, I_pred, H_pred, R_pred = SVIHR_pred
    S_actual, V_actual, I_actual, H_actual, R_actual = SVIHR_actual
    return sum((S_pred .- S_actual).^2 + (V_pred .- V_actual).^2 + (I_pred .- I_actual).^2 + (H_pred .- H_actual).^2 + (R_pred .- R_actual).^2) + λ * sum(abs2,(sum(S_pred + V_pred + I_pred + H_pred + R_pred) .- N))
end


function loss_fn(SVIHR_pred, SVIHR_actual; λ=0.01, α=0.01)
    S_pred, V_pred, I_pred, H_pred, R_pred = SVIHR_pred
    S_actual, V_actual, I_actual, H_actual, R_actual = SVIHR_actual
    
    # Calculate the prediction loss
    pred_loss = sum((S_pred .- S_actual).^2 + (V_pred .- V_actual).^2 + (I_pred .- I_actual).^2 + (H_pred .- H_actual).^2 + (R_pred .- R_actual).^2)
    
    # Calculate the regularization term
    weights = Flux.params(m)
    reg_loss = sum(α .* sum(abs2, w) for w in weights)
    
    # Return the total loss
    return pred_loss + λ * reg_loss
end


# Train the second neural network
opt2 = Flux.ADAM(0.01)
data2 = Iterators.repeated((), 50)
iter2 = 0
cb2 = function () #callback function to observe training
  global iter2 += 1
  if iter2 % 5 == 0
    display(obj_func())
  end
end
display(obj_func())
Flux.train!(obj_func, Flux.params(m), data2, opt2; cb=cb2)


ξ, T_I, M, Φ, μ, Λ, T_H = [0.079718848, 1.427, 0.026720524, 0.013517486, 0.0, 0.0, 1.57]
β, κ = m([beta, kappa])
#u0 = [0.99, 0.01, 0.0]
p = [β, κ, ξ, T_I, M, Φ, μ, Λ, T_H]
println(p)
prob = ODEProblem(svihr_model!, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=0.1)

# Plot the results
using Plots
plot(t_data, S_data, label="S")
plot!(t_data, V_data, label="V")
plot!(t_data, I_data, label="I")
plot!(t_data, H_data, label="H")
p1 = plot!(t_data, R_data, label="R")
plot(sol.t, sol[1, :], label="S_predicted")
plot!(sol.t, sol[2, :], label="V_predicted")
plot!(sol.t, sol[3, :], label="I_predicted")
plot!(sol.t, sol[4, :], label="H_predicted")
p2 = plot!(sol.t, sol[5, :], label="R_predicted")

plot(p1, p2, plot_title = "SVIHR",
    layout=(2,1), xtickfontsize = 12, ytickfontsize = 12, xguidefontsize=15, yguidefontsize=15,
    legendfontsize=12, titlefontsize=20)



