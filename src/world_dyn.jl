using CSV, DataFrames, Flux

# Read the 'births.csv' file and extract the second column
data_births = CSV.read("./Dataset/Parsed/births.csv", DataFrame)
data_births = data_births[:, 2]

# Read the 'deaths.csv' file and extract the second column
data_deaths = CSV.read("./Dataset/Parsed/deaths.csv", DataFrame)
data_deaths = data_deaths[:, 2]



# Define a neural network that takes in 2 inputs and outputs 2 values
model = Chain(Dense(2, 10, relu), Dense(10, 2))

# Define the loss function
function loss_fn(x, y)
  ŷ = model(x)
    return sum((ŷ .- y).^2)
end

obj_func() = mean([loss_fn(x, y) for (x, y) in data])

# Define the optimizer
opt = Flux.Optimise.ADAM(0.001)

# Define the data
data = [(rand(2), rand(2)) for _ in 1:1000]

# Define the callback function
iter = 0
cb = function () #callback function to observe training
  global iter += 1
  if iter % 50 == 0
    display(obj_func())
  end
end
    display(obj_func())
# Train the model
Flux.train!(obj_func, Flux.params(model), data, opt; cb=cb)


# Make predictions on the 'data_births' data using the trained network
predictions = Flux.predict(model, data_births)

using Plots

# Create a scatter plot of predicted vs actual values
scatter(data_births, data_deaths, label="Actual Values")
scatter!(data_births, predictions, label="Predicted Values")
xlabel!("Births")
ylabel!("Deaths")
title!("Actual vs Predicted Values")
