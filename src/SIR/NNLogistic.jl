using Flux
using Random

# Generate training data
Random.seed!(123)
x = range(-5, stop=5, length=100)
y = 1 ./ (1 .+ exp.(-x))

# Define the neural network
NNLogistic = Chain(x -> [x],
                   Dense(1 => 10, tanh),
                   Dense(10 => 1),
                   first)

# Select a random subset of the first 30% of the data to use as position_data
n = length(x)
k = Int(round(0.3 * n))
idx = randperm(k)
x_sub = x[idx]
y_sub = y[idx]
position_data = (y_sub .- minimum(y_sub)) ./ (maximum(y_sub) - minimum(y_sub))

# Define the Huber loss function with full-size y array
function huber_loss(x, y, δ)
    # Compute the Huber loss on the subset of data with the same size as x
    return sum(abs.(x - y[1:length(x)]) .<= δ ? 0.5 .* abs2(x - y[1:length(x)]) : δ .* (abs.(x - y[1:length(x)]) .- 0.5 .* δ))
end


# Define the loss function as a closure that calls the huber_loss function with the full data
loss() = sum(huber_loss.(NNLogistic.(x), position_data, 0.1))

# Train the model
opt = Flux.Descent(0.01)
data = Iterators.repeated((), 500000)
Flux.train!(loss, Flux.params(NNLogistic), data, opt)

# Evaluate the trained model
x_test = range(-10, stop=10, length=200)
y_test = NNLogistic.(x_test)

# Plot the results
using Plots
plot(x_sub, y_sub, label="Selected data")
plot!(x_test, minimum(y_sub) .+ (maximum(y_sub) - minimum(y_sub)) .* y_test, label="Learned function")
