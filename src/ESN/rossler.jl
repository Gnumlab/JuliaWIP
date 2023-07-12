#Generate the data
using Plots, OrdinaryDiffEq, ReservoirComputing, Random
using Plots.PlotMeasures

#Random.seed!(4242)
#define lorenz system
function lorenz!(du,u,p,t)
    du[1] =  -u[2]-u[3]
    du[2] = u[1]+(0.2*u[2])
    du[3] = 0.2 + u[3]*(u[1] - 5.7)
end

#solve and take data
prob = ODEProblem(lorenz!, [1.0,0.0,0.0], (0.0,20000.0))
data = solve(prob, ABM54(), dt=0.02)

#determine kind of prediction model: we use Generative prediction
#target data is next step of input data

#Shift the transient period, not representative of the steady state
#determine shift length, training length and prediction length


data_size = size(data)[2]
println(data_size)
shift = Int(floor(0.05*data_size))  #skip 1% data
train_len = Int(floor(0.7*data_size))
predict_len = data_size-shift-train_len

shift = 300
train_len = 5000
predict_len = 4500




#split the data accordingly
input_data = data[:, shift:shift+train_len-1]
target_data = data[:, shift+1:shift+train_len]
test_data = data[:,shift+train_len+1:shift+train_len+predict_len]


#After prepared the data (I think it could be reading the data and preprocess them to generate the right format and remove transient period) we generate the ESN
using ReservoirComputing

#define ESN parameters
res_size = 100
res_radius = 1.02
res_sparsity = 6/300
input_scaling = 0.1

#build ESN struct
esn = ESN(input_data;
    variation = Default(),
    reservoir = RandSparseReservoir(res_size, radius=res_radius, sparsity=res_sparsity),
    input_layer = WeightedLayer(scaling=input_scaling),
    reservoir_driver = RNN(),
    nla_type = NLADefault(),
    states_type = StandardStates())
    
    
#After generating the ESN we can train it and then predict new values
#define training method
training_method = StandardRidge(0.0)

#obtain output layer: it contains the trained output matrix together with other information useful for data prediction
output_layer = train(esn, target_data, training_method)

#execute the prediction
output = esn(Generative(predict_len), output_layer)



#inspect6 the results through plotting
using Plots, Plots.PlotMeasures

ts = 0.0:0.02:200.0
lorenz_maxlyap = 0.9056
predict_ts = ts[shift+train_len+1:shift+train_len+predict_len]
lyap_time = (predict_ts .- predict_ts[1])*(1/lorenz_maxlyap)

p1 = plot(lyap_time, [test_data[1,:] output[1,:]], label = ["actual" "predicted"],
    ylabel = "x(t)", linewidth=2.5);
p2 = plot(lyap_time, [test_data[2,:] output[2,:]], label = ["actual" "predicted"],
    ylabel = "y(t)", linewidth=2.5);
p3 = plot(lyap_time, [test_data[3,:] output[3,:]], label = ["actual" "predicted"],
    ylabel = "z(t)", linewidth=2.5);


plot(p1, p2, p3, plot_title = "Lorenz System Coordinates",
    layout=(3,1), xtickfontsize = 12, ytickfontsize = 12, xguidefontsize=15, yguidefontsize=15,
    legendfontsize=12, titlefontsize=20)
