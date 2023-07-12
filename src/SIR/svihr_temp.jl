




# Define the loss function
#function loss(p)
#    β, κ = m(x_data)[1,:]
#    p_pred = [β, κ, 0.2, 14.0, 0.1, 0.005, 0.0, 0.0, 21.0]
#    prob = ODEProblem(svihr_model!, u0, tspan, p_pred)
#    sol_pred = solve(prob, Tsit5(), saveat=t)
#    loss_u = sum(abs2, y_data - sol_pred.u[3,:])
#    loss_p = sum(abs2, p .- p_pred)
#    loss_u + loss_p
#end

#function loss(m, x_data, y_data, p)
#    β, κ = m(x_data)[1,:]
#    p_pred = [β, κ, 0.2, 14.0, 0.1, 0.005, 0.0, 0.0, 21.0]
#    prob = ODEProblem(svihr_model!, u0, tspan, p_pred)
#    sol_pred = solve(prob, Tsit5(), saveat=t)
#    loss_u = sum(abs2, y_data - sol_pred.u[3,:])
#    loss_p = sum(abs2, p .- p_pred)
#    loss_u + loss_p
#end
#println(size(y_data))
#println(size(m(x_data)))

function loss(m, x_data, y_data, p)
    y_pred = m(x_data)
    loss_u = sum(abs2, y_data - y_pred[3,:])
    loss_p = sum(abs2, p)
    loss_u + loss_p
end

# Define the callback function to print the loss
function callback(p, l)
    println("Loss: ", l)
    return false
end

using Optim

opt = Descent(0.01)
ps = Flux.params(m)
Flux.train!(params -> loss(m, x_data, y_data, params), ps, x_data, opt)

#Flux.train!(loss, Flux.params(NNLogistic), data, opt; cb=cb)
#for epoch = 1:1000
#    Flux.train!(loss, ps, [(opt,)], cb=callback)
#end
# Predict the parameters and plot the results
params_pred = m(x_data)[1,:]
prob = ODEProblem(svihr_model!, u0, tspan, params_pred)
sol_pred = solve(prob, Tsit5(), saveat=t)
using Plots
plot(sol_true, label=["S" "V" "I" "H" "R"])
plot!(sol_pred, label=["S" "V" "I" "H" "R"])




