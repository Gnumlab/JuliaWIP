using DifferentialEquations
using SciMLSensitivity
using Flux

# Define SIR ODE function
function sir_ode(du, u, p, t)
    β, γ = p
    S, I, R = u

    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end




tspan = (0.0,100.0)
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
gamma = rand(Float64, 2)



# Define a neural network that learns the parameter gamma
model = Chain(Dense(1, 50, relu), Dense(50, 1))

# Define the objective function


# Define the Huber loss function with full-size y array
function huber_loss(y_pred, y_true, delta)
    errors = abs.(y_true - y_pred)
    mask = errors .<= delta
    inliers = 0.5 .* errors[mask] .^ 2

    outliers = delta .* (errors[.!mask] - 0.5 .* delta)


    return mean([inliers; outliers])
end

function loss_fn(I_pred, I_actual)
    #sum(huber_loss.(I_pred, I_actual, 0.1))
    sum(abs2,I_pred .- I_actual)
end

function loss_fn(I_pred, I_actual, w, lambda)
    # Add L2 regularization penalty term
    loss = sum(abs2, I_pred .- I_actual) + lambda * sum(abs2.(w))
    return loss
end

function obj_func()
    β = 0.2
    γ = model([0.2])[1]
    u0 = [0.99, 0.01, 0.0]
    p_pred = [β, γ]
    prob_pred = ODEProblem(sir_ode, u0, tspan, p_pred)
    sol_pred = solve(prob_pred, Tsit5(), saveat=0.1, maxiters=Int(1e8))
    I_pred = sol_pred[2, :]
#    loss_fn(I_pred, I_data)

    # Use L2 regularization with lambda = 0.01
    loss_fn(I_pred, I_data, Flux.params(model), 0.01)
end

# Train the model to learn gamma



opt = Flux.Descent(0.01)
data = Iterators.repeated((), 100)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 50 == 0
    display(obj_func())
  end
end
display(obj_func())
Flux.train!(obj_func, Flux.params(model), data, opt; cb=cb)



# Use learned gamma parameter to solve SIR ODE
p = [0.2, model([0.2])[1]]
prob = ODEProblem(sir_ode, [0.99, 0.01, 0.0], (0.0, 100.0), p)
sol = solve(prob, Tsit5(), saveat=0.1)

# Plot the results
using Plots
plot(t_data, S_data, label="S")
plot!(t_data, I_data, label="I")
plot!(t_data, R_data, label="R")
plot!(sol.t, sol[1, :], label="S_predicted")
plot!(sol.t, sol[2, :], label="I_predicted")
plot!(sol.t, sol[3, :], label="R_predicted")z
