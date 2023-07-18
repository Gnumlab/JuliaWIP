using Statistics
using SimpleChains

using MLDatasets
using SimpleChains
using NNlib
using Random, Statistics



struct StaticMulticlassLogitCrossEntropyLoss{T<:AbstractVector{<:Integer}} <: SimpleChains.AbstractLoss{T}
    targets::T
end

target(loss::StaticMulticlassLogitCrossEntropyLoss) = loss.targets
(loss::StaticMulticlassLogitCrossEntropyLoss)(::Int) = loss

function calculate_loss(loss::StaticMulticlassLogitCrossEntropyLoss, logits)
    y = loss.targets
    total_loss = zero(eltype(logits))
    for i in eachindex(y)
        y_i = y[i] - 1
        scores = logits[i:i+9]
        correct_score = scores[y_i+1]
        margins = max.(0, scores .- correct_score .+ 1.0)
        margins[y_i+1] = 0.0
        total_loss += sum(margins)
    end
    total_loss
end

function (loss::StaticMulticlassLogitCrossEntropyLoss)(previous_layer_output::AbstractArray{T}, p::Ptr, pu) where {T}
    total_loss = calculate_loss(loss, previous_layer_output)
    total_loss, p, pu
end

function SimpleChains.layer_output_size(::Val{T}, sl::StaticMulticlassLogitCrossEntropyLoss, s::Tuple) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end

function SimpleChains.forward_layer_output_size(::Val{T}, sl::StaticMulticlassLogitCrossEntropyLoss, s) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end

function SimpleChains.chain_valgrad!(
    __,
    previous_layer_output::AbstractArray{T},
    layers::Tuple{StaticMulticlassLogitCrossEntropyLoss},
    _::Ptr,
    pu::Ptr{UInt8},
) where {T}
    loss = getfield(layers, 1)
    total_loss = calculate_loss(loss, previous_layer_output)
    y = loss.targets

    # Store the backpropagated gradient in the previous_layer_output array.
    for i in eachindex(y)
        p = softmax(previous_layer_output[i:i+9])
        y_i = (y[i] - 1) * 10
        y_i_plus_1 = y_i + 1
        previous_layer_output[i:i+9] = p
        previous_layer_output[y_i_plus_1:y_i+10] .-= 1
    end

    return total_loss, previous_layer_output, pu
end



# Define the LeNet-5 architecture with multi-class output
lenet = SimpleChain(
  (static(28), static(28), static(1)),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 6),
  SimpleChains.MaxPool(2, 2),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 16),
  SimpleChains.MaxPool(2, 2),
  Flatten(3),
  TurboDense(SimpleChains.relu, 120),
  TurboDense(SimpleChains.relu, 84),
  TurboDense(identity, 10),
)

# Load the MNIST dataset
xtrain3, ytrain0 = MLDatasets.MNIST.traindata(Float32)
xtest3, ytest0 = MLDatasets.MNIST.testdata(Float32)
xtrain4 = reshape(xtrain3, 28, 28, 1, :)
xtest4 = reshape(xtest3, 28, 28, 1, :)
ytrain1 = UInt32.(ytrain0 .+ 1)
ytest1 = UInt32.(ytest0 .+ 1)



# Set the random seed for reproducibility
Random.seed!(1234)

# Select a random subset of 10% of the examples
n = length(ytrain1)


# Reduce the dataset to the selected examples
xtrain4 = xtrain4[:, :, :, 1:2:1000]
ytrain1 = ytrain1[1:2:1000]

# Print the size of the reduced dataset
println("Size of reduced dataset: $(size(xtrain4, 4)) examples")

# Print examples in ytrain1 that are outside the range of 0 to 9
train_out_of_range = findall(x -> !(1 <= x <= 10), ytrain1)
if !isempty(train_out_of_range)
    println("Examples in ytrain1 that are outside the range of 1 to 10:")
    for i in train_out_of_range
        println("ytrain1[$i] = $(ytrain1[i])")
    end
else
    println("All examples in ytrain1 are within the range of 1 to 10.")
end

# Print examples in ytest1 that are outside the range of 0 to 9
test_out_of_range = findall(x -> !(1 <= x <= 10), ytest1)
if !isempty(test_out_of_range)
    println("Examples in ytest1 that are outside the range of 1 to 10:")
    for i in test_out_of_range
        println("ytest1[$i] = $(ytest1[i])")
    end
else
    println("All examples in ytest1 are within the range of 1 to 10.")
end

# Define the loss function for the model and input data
loss_fn = StaticMulticlassLogitCrossEntropyLoss(ytrain1)

println("loss")

# Add the loss function to the model
lenetloss = SimpleChains.add_loss(lenet, loss_fn)

println("lenetloss")



# Define the loss function for the model and input data
loss_fn = StaticMulticlassLogitCrossEntropyLoss(ytrain1)

# Add the loss function to the model
lenetloss = SimpleChains.add_loss(lenet, loss_fn)

# Initialize the parameters of the model
p = SimpleChains.init_params(lenet, size(xtrain4))

# Create a small test dataset
xtest = xtrain4[:, :, :, 1:5]
ytest = ytrain1[1:5]



# Calculate the expected accuracy
num_cor = 0
for i in 1:length(ytest)
    yhat = argmax(lenet(xtest[:, :, :, i], p))
    if yhat == ytest[i]
        num_cor += 1
    end
end
accuracy = num_cor / length(ytest)


# Calculate the accuracy and loss using the SimpleChains function
G = SimpleChains.alloc_threaded_grad(lenetloss)
SimpleChains.valgrad!(G, lenetloss, xtest, p)
acc, loss_chains = SimpleChains.accuracy_and_loss(lenetloss, xtest, p)

# Print the expected and actual accuracy and loss
println("Expected accuracy = ", accuracy)
println("Actual accuracy = ", acc)
println("Expected loss = ", loss)
println("Actual loss = ", loss_chains)
