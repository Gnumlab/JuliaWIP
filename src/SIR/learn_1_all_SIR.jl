using DifferentialEquations
using SciMLSensitivity
using Statistics
using Flux

# Define SIR ODE function
function sir_ode(du, u, p, t)
    β, γ = p[1], p[2]
    S, I, R = u

    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

tspan = (0.0,100.0)
tspan_learn = (0.0,20.0)  #learn only on a fraction of the actual data generated

# Generate fake data
t_data = collect(0.0:0.1:100.0)
β_actual = 0.2
γ_actual = 0.1
u0 = [0.99, 0.01, 0.0]
p_actual = [β_actual, γ_actual]
prob_actual = ODEProblem(sir_ode, u0, tspan, p_actual)
sol_actual = solve(prob_actual, Tsit5(), saveat=0.1)
I_data = sol_actual[2, :] .+ 0.01 .* randn(length(t_data))
S_data = sol_actual[1, :] .- I_data
R_data = sol_actual[3, :]

# Initialize gamma parameter randomly
gamma = rand(Float64)

# Define a neural network that learns the parameters beta and gamma
model = Chain(Dense(1, 100, relu), Dense(100, 3))

# Define the objective function
function loss_fn(y_pred, y_actual)
    λ = 0.01
    return sum((y_pred .- y_actual).^2) + λ * sum(param -> sum(param.^2), Flux.params(model))
end

function obj_func()
    β = 0.2
    γ = model([gamma])[1:3]
    u0 = [0.99, 0.01, 0.0]
    p_pred = [β, γ[1], γ[2]]
    prob_pred = ODEProblem(sir_ode, u0, tspan_learn, p_pred)
    sol_pred = solve(prob_pred, Tsit5(), saveat=0.1, maxiters=Int(1e8))
    [sol_pred[1, end], sol_pred[2, end], sol_pred[3, end]]
end

# Train the model to learn beta and gamma
opt = Flux.ADAM(0.001)

data = Iterators.repeated((), 1000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(obj_func())
  end
end

display(obj_func())

Flux.train!(obj_func, Flux.params(model), data, opt; cb=cb)

# Use learned beta and gamma parameters to solve SIR ODE
p = [0.2, model([gamma])[1], model([gamma])[2]]
prob = ODEProblem(sir_ode, [0.99, 0.01, 0.0], (0.0, 100.0), p)
sol = solve(prob, Tsit5(), saveat=0.1)

# Plot the results
using Plots
plot(t_data, S_data, label="S")
plot!(t_data, I_data, label="I")
plot!(t_data, R_data, label="R")
plot!(sol.t, sol[1, :], label="S_predicted")
plot!(sol.t, sol[2, :], label="I_predicted")
plot!(sol.t, sol[3, :], label="R_predicted")
