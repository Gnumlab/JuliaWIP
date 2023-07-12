using DifferentialEquations
using SciMLSensitivity
using Statistics
using Flux

# Define SIR ODE function
function sir_ode(du, u, p, t)
    β, γ = p
    S, I, R = u

    du[1] = -β * S * I
    du[2] = β * S * I - γ * I
    du[3] = γ * I
end

function actual_sir_ode(du, u, p, t)
    β, γ = p
    S, I, R = u

    du[1] = -β * S * I +0.01*randn()+0.001
    du[2] = β * S * I - γ * I + 0.05*randn()+0.001
    du[3] = γ * I
end



tspan = (0.0,100.0)
#tspan_learn = (0.0,20.0)  #learn only on a fraction of the actual data generated
# Generate fake data
t_data = collect(0.0:0.1:100.0)
β_actual = 0.2
γ_actual = 0.1
u0 = [0.99, 0.01, 0.0]
p_actual = [β_actual, γ_actual]
prob_actual = ODEProblem(actual_sir_ode, u0, tspan, p_actual)
sol_actual = solve(prob_actual, Tsit5(), saveat=0.1)
I_data = sol_actual[2, :] #.+ 0.01 .* randn(length(t_data))#sin.(I_data)
S_data = sol_actual[1, :] #.- I_data
R_data = sol_actual[3, :]

# Initialize gamma parameter randomly
gamma = rand(Float64)

println(gamma)

# Define a neural network that learns the parameter gamma
model = Chain(Dense(1, 30, relu), Dense(30, 1))

# Define the objective function
function loss_fn_mse(I_pred, I_actual)
    sum(abs2,I_pred .- I_actual)
end

function loss_fn(I_pred, I_actual)
    λ = 0.01
    return sum((I_pred .- I_actual).^2) + λ * sum(param -> sum(param.^2), Flux.params(model))
end

function obj_func()
    β = 0.2
    γ = model([gamma])[1]
    u0 = [0.99, 0.01, 0.0]
    p_pred = [β, γ]
    prob_pred = ODEProblem(sir_ode, u0, tspan, p_pred)
    sol_pred = solve(prob_pred, Tsit5(), saveat=0.1, maxiters=Int(1e8))
    I_pred = sol_pred[2, :]
    loss_fn(I_pred, I_data)
end

# Train the model to learn gamma



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



# Use learned gamma parameter to solve SIR ODE
p = [0.2, model([gamma])[1]]
println(p)
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










############## SECOND MODEL ################


# Define the objective function for the physical-informed neural network
function obj_func2()
    β = 0.2
    γ = model([1.0])[1]
    u0 = [0.99, 0.01, 0.0]
    p_pred = [β, γ]
    SIR_pred = model2(u0)
    prob_pred = ODEProblem(sir_ode, u0, tspan, p_pred)
    sol_pred = solve(prob_pred, Tsit5(), saveat=0.1, maxiters=Int(1e8))
    SIR_actual = [sol_pred[1, :], sol_pred[2, :], sol_pred[3, :]]
    SIR_loss(SIR_pred, SIR_actual)
end

# Define the loss function for the physical-informed neural network
function SIR_loss(SIR_pred, SIR_actual)
    S_pred, I_pred, R_pred = SIR_pred
    S_actual, I_actual, R_actual = SIR_actual
    loss = sum((S_pred .- S_actual).^2 + (I_pred .- I_actual).^2 + (R_pred .- R_actual).^2)
    return loss
end


function obj_func2_()
    β = 0.2
    γ = model([gamma])[1]
    u0 = [0.99, 0.01, 0.0]
    p_pred = [β, γ]
    SIR_pred = model2(u0)
    #SIR_pred_jac = Flux.jacobian(model2, u0)
    SIR_pred_jac = Flux.jacobian(x -> model2(x)[2, :], u0)
    prob_pred = ODEProblem(sir_ode, u0, tspan, p_pred)
    sol_pred = solve(prob_pred, Tsit5(), saveat=0.1, maxiters=Int(1e8))
    SIR_actual = [sol_pred[1, :], sol_pred[2, :], sol_pred[3, :]]
    loss_fn2(SIR_pred, SIR_actual, SIR_pred_jac)
end


# Define the loss function for the physical-informed neural network
function loss_fn2(SIR_pred, SIR_actual, SIR_pred_jac)
    λ = 0.01
    S_pred, I_pred, R_pred = SIR_pred
    S_actual, I_actual, R_actual = SIR_actual
    return sum((S_pred .- S_actual).^2 + (I_pred .- I_actual).^2 + (R_pred .- R_actual).^2) + λ * sum(param -> sum(param.^2), Flux.params(model, model2)) + sum(x -> sum(abs2, x), SIR_pred_jac)
end
# Define the second neural network to learn the whole SIR model with the predicted value of gamma
model2 = Chain(Dense(3, 30, relu), Dense(30, 3))

# Define the loss function for the second neural network
function loss_fn2_mse(SIR_pred, SIR_actual)
    sum(abs2,SIR_pred .- SIR_actual)
end

# Train the second neural network
opt2 = Flux.ADAM(0.001)

data2 = Iterators.repeated((), 5000)
iter2 = 0
cb2 = function () #callback function to observe training
  global iter2 += 1
  if iter2 % 500 == 0
    display(obj_func2())
  end
end
display(obj_func2())
Flux.train!(obj_func2, Flux.params(model2), data2, opt2; cb=cb2)

# Use learned gamma parameter and learned SIR model to predict S, I, and R
p2 = [0.2, model([gamma])[1], model2([0.99, 0.01, 0.0])[1], model2([0.99, 0.01, 0.0])[2], model2([0.99, 0.01, 0.0])[3]]
prob2 = ODEProblem(sir_ode, [0.99, 0.01, 0.0], tspan, p2)
sol2 = solve(prob2, Tsit5(), saveat=0.1)

# Plot the results
plot(t_data, S_data, label="S")
plot!(t_data, I_data, label="I")
plot!(t_data, R_data, label="R")
plot!(sol2.t, sol2[1, :], label="S_predicted")
plot!(sol2.t, sol2[2, :], label="I_predicted")
plot!(sol2.t, sol2[3, :], label="R_predicted")






