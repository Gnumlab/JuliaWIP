#Generate the data

using DifferentialEquations
using Plots, OrdinaryDiffEq, ReservoirComputing, Random, StatsBase
using Plots.PlotMeasures
using MLJLinearModels
#Random.seed!(4242)

function exponential_growth!(du, u, p, t)

    r = p[1] # growth rate
    du[1] = r * u[1] # dP/dt = kP
end


function logistic_growth(du, u, p, t)
    r = p[1] # intrinsic growth rate
    K = p[2] # carrying capacity
    du[1] = r * u[1] * (K - u[1]) / K # logistic growth equation
end

# Define the initial conditions, parameters, and time span
u₀ = [1.0] # initial population size
p = [1.1, 100] # intrinsic growth rate and carrying capacity
tspan = (0.0, 100.0) # time span


p₀ = [10.0] # initial population size
r = [1.1] # growth rate
tspan = (0.0, 10.0) # time span to integrate over

# Create the ODE problem and solve it using the Tsit5() solver
#prob = ODEProblem(logistic_growth, u₀, tspan, p)
#prob = ODEProblem(exponential_growth!,p₀,tspan,r)

#data = solve(prob, Tsit5(), dt=0.015625)


function exponential_growth(P₀, r, n)
    
    # p0 is the initial population size
    # r is the growth rate
    # n is the time
    P = [1, P₀]
    for i in 1:n
        Pₙ = P[1, end] * r  #P[i] = P[i-1]*r
        push!(P, Pₙ)
    end
    return P
end

#data = exponential_growth(1.0, 1.01, 40)


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
data = solve(prob, reltol=1e-6, saveat=0.1, dt=0.015)
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
using ReservoirComputing


number_test = 0
trials = 0
less_ij = []

for i = 0:number_test
    res_size = 10#0 + 5*100
for j = 0:0
count = 0
min = 1.0
max = 0.0
print(i)
print(" ")
println(j)
for k = 0:trials
    #define ESN parameters

    res_radius = 0.1140 + 2*0.05
    res_sparsity = 0.051 +0*0.05#10/res_size
    input_scaling = 0.1
    
    #build ESN struct
    esn = ESN(input_data;
        variation = Default(),
        reservoir = RandSparseReservoir(res_size, radius=res_radius, sparsity=res_sparsity),
        input_layer = SparseLayer(scaling=input_scaling),
        reservoir_driver = RNN(),
        nla_type = NLADefault(),
        states_type = StandardStates())
    
    
    #After generating the ESN we can train it and then predict new values
    #define training method
    training_method = StandardRidge(0.0)


    #obtain output layer: it contains the trained output matrix together with other information useful for data prediction
    output_layer = ReservoirComputing.train(esn, target_data, training_method)

    #execute the prediction
    global output = esn(Generative(predict_len), output_layer)
    if ( rmsd(test_data, output, normalize=true) < 1.0)
        count +=1
        value = rmsd(test_data, output, normalize=true)
        if (value < min)
            min = value
            global best_out = output
        end
        if (value > max)
            max = value
        end
    end


end  #k
    if ( count >=trials * 0.7)
        push!(less_ij, [i,j, min, max])
    end

end  #j   
end  #i

for i in less_ij
    println(i)
end


#inspect6 the results through plotting
using Plots, Plots.PlotMeasures


p1 = plot([test_data[1,:] best_out[1,:]], label = ["actual" "predicted"],
    ylabel = "x(t)", linewidth=2.5);
#p2 = plot([test_data[2,:] output[2,:]], label = ["actual" "predicted"],
#    ylabel = "y(t)", linewidth=2.5);
p2 = plot(input_data[1,:], label = ["train"],
    ylabel = "y(t)", linewidth=2.5);


plot(p1, p2, plot_title = "Lorenz System Coordinates",
    layout=(2,1), xtickfontsize = 12, ytickfontsize = 12, xguidefontsize=15, yguidefontsize=15,
    legendfontsize=12, titlefontsize=20)
    
savefig("log_2.png")
