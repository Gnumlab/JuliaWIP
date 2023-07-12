#Generate the data

using DifferentialEquations
using Plots, OrdinaryDiffEq, Flux, Random, StatsBase
using Plots.PlotMeasures
using MLJLinearModels
#Random.seed!(4242)


# Define the function for the logistic growth equation
function logistic_growth(du, u, p, t)
    r = p[1] # intrinsic growth rate
    K = p[2] # carrying capacity
    du[1] = r * u[1] * (K - u[1]) / K # logistic growth equation
end

# Define the initial conditions, parameters, and time span
u₀ = [1.0] # initial population size
p = [0.051, 1000] # intrinsic growth rate and carrying capacity
tspan = (0.0, 250.0) # time span

# Create the ODE problem and solve it using the Tsit5() solver
prob = ODEProblem(logistic_growth, u₀, tspan, p)
data = solve(prob, reltol=1e-6, saveat=2.0, dt=0.015)
#determine kind of prediction model: we use Generative prediction
#target data is next step of input data

#Shift the transient period, not representative of the steady state
#determine shift length, training length and prediction length
data_size = size(data)[2]
println(data_size)
shift = Int(floor(0.01*data_size)) + 1  #skip 1% data
train_len = Int(floor(0.304*data_size))
predict_len = data_size-shift-train_len

#split the data accordingly
input_data = data[:, shift:shift+train_len-1]
target_data = data[:, shift+1:shift+train_len]
test_data = data[:,shift+train_len+1:shift+train_len+predict_len]


#After prepared the data (I think it could be reading the data and preprocess them to generate the right format and remove transient period) we generate the ESN

number_test = 0
trials = 0
less_ij = []

random_positions = [rand(1:train_len) for i in 1:5] # random values in [-1,1]
random_input = [input_data[:, i] for i in random_positions]
random_target = [target_data[:, i] for i in random_positions]



NNLogistic = Chain(x -> [x],
           Dense(1 => 32,tanh),
           Dense(32 => 16),
           Dense(16 => 1),
           first)


loss() = sum(abs2,NNLogistic(input_data[i]) - target_data[i] for i in random_positions)


opt = Flux.Descent(0.01)
data = Iterators.repeated((), 25)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 5 == 0
    display(loss())
  end
end
display(loss())
Flux.train!(loss, Flux.params(NNLogistic), data, opt; cb=cb)


learned_logistic = NNLogistic.(target_data)


using Plots, Plots.PlotMeasures


p1 = plot(learned_logistic[1,:], label = ["predicted"],
    ylabel = "x(t)", linewidth=2.5);
p2 = plot(test_data[1,:], label = ["actual"],
    ylabel = "x(t)", linewidth=2.5);
#p1 = plot([test_data[1,:] learned_logistic[1,:]], label = ["actual" "predicted"],
#    ylabel = "x(t)", linewidth=2.5);
#p2 = plot([test_data[2,:] output[2,:]], label = ["actual" "predicted"],
#    ylabel = "y(t)", linewidth=2.5);


plot(p1, p2, plot_title = "Lorenz System Coordinates",
    layout=(2,1), xtickfontsize = 12, ytickfontsize = 12, xguidefontsize=15, yguidefontsize=15,
    legendfontsize=12, titlefontsize=20)
    




