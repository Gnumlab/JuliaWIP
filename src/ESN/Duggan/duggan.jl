#Generate the data
using OrdinaryDiffEq
using DifferentialEquations, Plots
using Interpolations

using Plots, OrdinaryDiffEq, ReservoirComputing, Random
using Plots.PlotMeasures
#Random.seed!(4242)


T = 100.0

function interpolation(x)
    x_resource = 0:100:1000
    y_efficiency = [0, 0.25, 0.45, 0.63, 0.75, 0.85, 0.92, 0.96, 0.98, 0.99, 1.0]
    func_efficiency = LinearInterpolation(x_resource, y_efficiency, extrapolation_bc=Line())
    return func_efficiency(x)
end


# Define the system of differential equations
function subsystem1(du, u, p, t)
    population, resources = u
    death_birth_rate, desired_growth_fraction, revenue_per_unit_extracted, fraction_profits_reinvested, cost_per_investment = p

    # To use a different extrapolation behavior, specify the `extrapolation_bc` argument accordingly.
    extraction_efficiency_per_unit_capital = interpolation(resources)

    # Calculate the resource usage (extraction) and total revenue
    resource_usage = population .* extraction_efficiency_per_unit_capital
    total_revenue = revenue_per_unit_extracted .* resource_usage

    # Calculate the capital costs, profit, and capital funds
    capital_costs = population .* 0.10
    profit = total_revenue - capital_costs
    capital_funds = profit .* fraction_profits_reinvested

    # Calculate the maximum investment and set the investment to the maximum
    maximum_investment = capital_funds ./ cost_per_investment
    investment = maximum_investment

    # Calculate the rate of change of the population and resources
    du[1] = dpopulation = investment - population .* death_birth_rate
    du[2] = dresources = -resource_usage

    return du
end



# Set up the initial conditions
u0 = [5.0, 1000.0]
p = [0.05, 0.07, 3.0, 0.12, 2.0]
tspan = (0.0, T)

# Set up the ODE problem and solve it
prob = ODEProblem(subsystem1, u0, tspan, p)
data = solve(prob, Tsit5(), dt=0.015625, dtmax=0.015625)


#determine kind of prediction model: we use Generative prediction
#target data is next step of input data

#Shift the transient period, not representative of the steady state
#determine shift length, training length and prediction length
shift = 1
train_len = 5000
predict_len = 1250
print(size(data))
#split the data accordingly
input_data = data[:, shift:shift+train_len-1]
target_data = data[:, shift+1:shift+train_len]
test_data = data[:,shift+train_len+1:shift+train_len+predict_len]


#After prepared the data (I think it could be reading the data and preprocess them to generate the right format and remove transient period) we generate the ESN
using ReservoirComputing

#define ESN parameters
res_size = 100
res_radius = 1.9
res_sparsity = 6/100
input_scaling = 0.2

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
training_method = StandardRidge(5.9)

#obtain output layer: it contains the trained output matrix together with other information useful for data prediction
output_layer = train(esn, target_data, training_method)

#execute the prediction
output = esn(Generative(predict_len), output_layer)



#inspect the results through plotting
using Plots, Plots.PlotMeasures

ts = 0.0:0.015625:T
lorenz_maxlyap = 0.9056
predict_ts = ts[shift+train_len+1:shift+train_len+predict_len]
lyap_time = (predict_ts .- predict_ts[1])*(1/lorenz_maxlyap)

p3 = plot(data, 
    lw=2, title = "Exponential Growth Model",
    xlabel = "time", ylabel = "population size"
)

p1 = plot(lyap_time, [test_data[1,:] output[1,:]], label = ["actual" "predicted"],
    ylabel = "x(t)", linewidth=2.5);
p2 = plot(lyap_time, [test_data[2,:] output[2,:]], label = ["actual" "predicted"],
    ylabel = "y(t)", linewidth=2.5, xticks=false);


plot(p1, p2, p3, plot_title = "Lorenz System Coordinates",
    layout=(3,1), xtickfontsize = 12, ytickfontsize = 12, xguidefontsize=15, yguidefontsize=15,
    legendfontsize=12, titlefontsize=20)
