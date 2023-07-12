# Define SIR ODE function
function sir_ode(du, u, p, t)
    β, γ = p
    S, I, R = u

    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

tspan = (0.0, 100.0)
tspan_learn = (0.0, 20.0)  #learn only on a fraction of the actual data generated

# Generate fake data
t_data = collect(0.0:0.1:100.0)
β_actual = 0.2
γ_actual = 0.1
u0 = [0.99, 0.01, 0.0]
p_actual = [β_actual, γ_actual]
prob_actual = ODEProblem(sir_ode, u0, tspan, p_actual)
sol_actual = solve(prob_actual, Tsit5(), saveat=0.1)
I_data = sol_actual[2, :] .+ 0.01 .* randn(length(t_data))#sin.(I_data)
S_data = sol_actual[1, :] .- I_data
R_data = sol_actual[3, :]

# Initialize beta and gamma parameters randomly
#params = Flux.Params([randn() for _ in 1:2])

# Define a neural network that learns the parameters beta and gamma
model = Chain(Dense(2, 100, relu), Dense(100, 1))

# Define the objective function
function loss_fn(I_pred, I_actual)
    λ = 0.01
    return sum((I_pred .- I_actual).^2) + λ * sum(param -> sum(param.^2), Flux.params(model))
end

function obj_func()
    β, γ = model(params)
    u0 = [0.99, 0.01, 0.0]
    p_pred = [β, γ]
    prob_pred = ODEProblem(sir_ode, u0, tspan, p_pred)
    sol_pred = solve(prob_pred, Tsit5(), saveat=0.1, maxiters=Int(1e8))
    I_pred = sol_pred[2, :]
    loss_fn(I_pred, I_data) + loss_fn(I_pred, I_data)
end

# Train the model to learn beta and gamma
opt = Flux.ADAM(0.001)
#params = Flux.Params([Float32.randn() for _ in Flux.params(model)])
data = Iterators.repeated((), 1000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 500 == 0
    display(obj_func())
  end
end

Flux.train!(obj_func, Flux.params(model), data, opt; cb=cb)

# Use learned beta and gamma parameters to solve SIR ODE
β, γ = model(params)
p = [β, γ]
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
