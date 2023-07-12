using DifferentialEquations

t = 0:0.001:1.0
k = 100.0
force(dx,x,k,t) = -k*x + 0.1sin(x)
function logistic_growth(du, u, p, t)
    r = p[1] # intrinsic growth rate
    K = p[2] # carrying capacity
    du[1] = r * u[1] * (K - u[1]) / K # logistic growth equation
end

# Define the initial conditions, parameters, and time span
u0 = [1.0] # initial population size
p = [0.051, 1000] # intrinsic growth rate and carrying capacity
tspan = (0.0, 250.0) # time span

prob = ODEProblem(logistic_growth, u0, tspan, p)
sol = solve(prob)

plot(sol)



plot_t = 0:0.1:250
data_plot = sol(plot_t)
positions_plot = [state[1] for state in data_plot]
#force_plot = [force(state[1],state[2],k,t) for state in data_plot]

# Generate the dataset
t1 = 0:10.3:100
t2 = 100:10.3:250
t  = 0:10.3:250


#position_data1 = [state[1] for state in sol(t1)]
#position_data2 = [state[1] for state in sol(t2)]
position_data = [state[1] for state in sol(t)]
#force_data = [force(state[1],state[2],k,t) for state in sol(t)]

 #plot(plot_t,positions_plot,xlabel="t",label="True Force")
#scatter!(t1,position_data,label="Force Measurements")
#scatter!(t2,position_data1,label="Force Measurements")
#scatter!(t2,position_data2,label="Force Measurements")


NNLogistic = Chain(x -> [x],
           Dense(1 => 10,tanh),
           Dense(10 => 1),
           first)
           
           
loss() = sum(abs2,NNLogistic(i) - position_data[i] for i in 1:length(position_data))
loss()


opt = Flux.Descent(0.01)
data = Iterators.repeated((), 500000)
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 50000 == 0
    display(loss())
  end
end
display(loss())
Flux.train!(loss, Flux.params(NNLogistic), data, opt; cb=cb)


learned_logistic_plot = NNLogistic.(plot_t)

#plot(plot_t,positions_plot,xlabel="t",label="True Force")
plot!(plot_t,learned_logistic_plot,label="Predicted Force")
scatter!(t,position_data,label="Force Measurements")
#scatter!(t1,position_data1,label="Force Measurements")
#scatter!(t2,position_data2,label="Force Measurements")

