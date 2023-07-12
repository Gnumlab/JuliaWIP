using Plots, OrdinaryDiffEq, ReservoirComputing, Random
using Plots.PlotMeasures


Random.seed!(42)
#define lorenz system
# Define the function for the Lotka-Volterra equations
function lotka_volterra(du, u, p, t)
    P = u[1]
    Q = u[2]
    α = p[1]
    β = p[2]
    γ = p[3]
    δ = p[4]
    du[1] = α*P - β*P*Q # prey equation
    du[2] = -γ*Q + δ*β*P*Q # predator equation
end

# Define the initial conditions, parameters, and time span
u₀ = [10.0, 1.0] # initial prey and predator populations
p = [1.5, 1.0, 3.0, 1.0] # α, β, γ, δ
tspan = (0.0, 200.0) # time span

# Create the ODE problem and solve it using the Tsit5() solver
prob = ODEProblem(lotka_volterra, u₀, tspan, p)
data = solve(prob, ABM54(), dt=0.02)

# Plot the results
#plot(data)


shift = 100
train_len = 5000
predict_len = 2000
input_data = data[:, shift:shift+train_len-1]
target_data = data[:, shift+1:shift+train_len]
test_data = data[:,shift+train_len+1:shift+train_len+predict_len]



#define ESN parameters
res_size = 50       #this value determines the dimensions of the reservoir matrix

res_radius = 1.19     #The res_radius determines the scaling of the spectral radius of the reservoir matrix

res_sparsity = 5/50 #The sparsity of the reservoir matrix in this case is obtained 
                     #by choosing a degree of connections and dividing that by the reservoir size

input_scaling = 0.4 #The value of input_scaling determines the upper and lower bounds 
                     #of the uniform distribution of the weights in the WeightedLayer()
                     #The value of 0.1 represents the default.

#build ESN struct
esn = ESN(input_data;
    variation = Default(),
    reservoir = RandSparseReservoir(res_size, radius=res_radius, sparsity=res_sparsity),
    input_layer = WeightedLayer(scaling=input_scaling),
    reservoir_driver = RNN(),
    nla_type = NLADefault(),
    states_type = StandardStates())

#define training method
training_method = StandardRidge(0.0)

#obtain output layer
output_layer = train(esn, target_data, training_method) #here we are using target data,
                                                        #that in this case is the one step ahead evolution of the Lorenz system.
output = esn(Generative(predict_len), output_layer)


ts = 0.0:0.02:200.0
predict_ts = ts[shift+train_len+1:shift+train_len+predict_len]

p1 = plot( predict_ts, [test_data[1,:] output[1,:]], label = ["actual" "predicted"], 
    ylabel = "Prey", linewidth=2.5, xticks=false, yticks = true);
p2 = plot( predict_ts, [test_data[2,:] output[2,:]], label = ["actual" "predicted"], 
    ylabel = "Predator", linewidth=2.5, xlabel = "Time",  yticks = true);


plot(p1, p2, size=(1080, 720), plot_title = "Lotka Volterra ", 
    layout=(3,1), xtickfontsize = 12, ytickfontsize = 12, xguidefontsize=15, yguidefontsize=15,
    legendfontsize=12, titlefontsize=20, left_margin=4mm)
