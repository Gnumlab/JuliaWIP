#Generate the data

using DifferentialEquations
using Plots, OrdinaryDiffEq, ReservoirComputing, Random
using Plots.PlotMeasures
#Random.seed!(4242)

function lotka_volterra!(du,u,p,t)
    🐰,🐺 = u
    α,β,γ,δ = p
    du[1] = d🐰 = α*🐰 - β*🐰*🐺
    du[2] = d🐺 = γ*🐰*🐺 - δ*🐺
end
u₀ = [1.0,1.0]
tspan = (0.0, 10000.0)
p = [0.5,1.0,3.0,1.0]
prob = ODEProblem(lotka_volterra!,u₀,tspan,p)
data = DifferentialEquations.solve(prob, Tsit5(), dt=0.15625)
#determine kind of prediction model: we use Generative prediction
#target data is next step of input data

#Shift the transient period, not representative of the steady state
#determine shift length, training length and prediction length
data_size = size(data)[2]
println(data_size)
shift = Int(floor(0.05*data_size))  #skip 1% data
train_len = Int(floor(0.7*data_size))
predict_len = 25#data_size-shift-train_len


#split the data accordingly
input_data = data[:, shift:shift+train_len-1]
target_data = data[:, shift+1:shift+train_len]
test_data = data[:,shift+train_len+1:shift+train_len+predict_len]


#After prepared the data (I think it could be reading the data and preprocess them to generate the right format and remove transient period) we generate the ESN
using ReservoirComputing

#define ESN parameters
res_size = 500
res_radius = 1.0245
res_sparsity = 30/res_size
input_scaling = 0.350

#build ESN struct
esn = ESN(input_data;
    variation = Default(),
    reservoir = RandSparseReservoir(res_size, radius=res_radius, sparsity=res_sparsity),
    input_layer = WeightedLayer(scaling=input_scaling),
    reservoir_driver = RNN(leaky_coefficient=0.5),
    nla_type = NLADefault(),
    states_type = StandardStates())
    
    
#After generating the ESN we can train it and then predict new values
#define training method
training_method = StandardRidge(0.9)

#obtain output layer: it contains the trained output matrix together with other information useful for data prediction
output_layer = train(esn, target_data, training_method)

#execute the prediction
output = esn(Generative(predict_len), output_layer)



#inspect6 the results through plotting
using Plots, Plots.PlotMeasures

#ts = 0.0:200.0
#lorenz_maxlyap = 0.9056
#predict_ts = ts[shift+train_len+1:shift+train_len+predict_len]
#lyap_time = (predict_ts .- predict_ts[1])*(1/lorenz_maxlyap)

p1 = plot([test_data[1,:] output[1,:]], label = ["actual" "predicted"],
    ylabel = "x(t)", linewidth=2.5,);
p2 = plot([test_data[2,:] output[2,:]], label = ["actual" "predicted"],
    ylabel = "y(t)", linewidth=2.5, );


plot(p1, p2, plot_title = "Lorenz System Coordinates",
    layout=(2,1), xtickfontsize = 12, ytickfontsize = 12, xguidefontsize=15, yguidefontsize=15,
    legendfontsize=12, titlefontsize=20)
