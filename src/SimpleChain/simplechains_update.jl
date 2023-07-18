using SimpleChains
using StaticArrays

struct BinaryLogitCrossEntropyLoss{T} <: SimpleChains.AbstractLoss{T}
    target::T
end

target(loss::BinaryLogitCrossEntropyLoss) = loss.target
(loss::BinaryLogitCrossEntropyLoss)(::Int) = loss

function calculate_loss(loss::BinaryLogitCrossEntropyLoss, logits)
    y = loss.target
    p = inv(1 + exp(-logits))
    return -y * log(p) - (1 - y) * log(1 - p)
end

function (loss::BinaryLogitCrossEntropyLoss)(previous_layer_output::AbstractArray{T}, p::Ptr, pu) where {T}
    total_loss = calculate_loss(loss, previous_layer_output[1])
    return total_loss, p, pu
end

function SimpleChains.layer_output_size(::Val{T}, sl::BinaryLogitCrossEntropyLoss, s::Tuple) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end
function SimpleChains.forward_layer_output_size(::Val{T}, sl::BinaryLogitCrossEntropyLoss, s) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end

function SimpleChains.chain_valgrad!(
    __,
    previous_layer_output::AbstractArray{T},
    layers::Tuple{BinaryLogitCrossEntropyLoss},
    _::Ptr,
    pu::Ptr{UInt8},
) where {T}
    loss = getfield(layers, 1)
    total_loss = calculate_loss(loss, previous_layer_output)
    y = loss.targets

    # Store the backpropagated gradient in the previous_layer_output array.
    for i in eachindex(y)
        sign_arg = 2 * y[i] - 1
        # Get the value of the last logit
        logit_i = previous_layer_output[i]
        previous_layer_output[i] = -(sign_arg * inv(1 + exp(sign_arg * logit_i)))
    end

    return total_loss, previous_layer_output, pu
end



# Define the neural network model
model = SimpleChain(
    static(2),
    TurboDense(tanh, 32),
    TurboDense(tanh, 16),
    TurboDense(identity, 1)
)

# Create a data loader for the training data
batch_size = 64
X = rand(Float32, 2, batch_size)
Y = rand(Bool, batch_size)

# Initialize the model parameters
parameters = SimpleChains.init_params(model)

# Create an ADAM optimizer
learning_rate=3e-4
opt = SimpleChains.ADAM(learning_rate)

# Train the model for 10 epochs
num_epochs = 10
for epoch = 1:num_epochs
    # Compute loss and gradients
    loss_fun = SquaredLoss(Y)
    gradients = SimpleChains.alloc_threaded_grad(model)
    model_loss = SimpleChains.add_loss(model, loss_fun);
    SimpleChains.valgrad!(gradients, model_loss, X, parameters)

    # Update the parameters using the ADAM optimizer
    SimpleChains.update!(opt, parameters, gradients)
end

# Predict on new data
X_new = rand(Float32, 2, 10)
Y_pred = model(X_new, parameters)
