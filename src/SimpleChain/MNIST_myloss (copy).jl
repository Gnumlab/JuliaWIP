using SimpleChains

struct BinaryLogitCrossEntropyLoss{T,Y<:AbstractVector{T}} <: SimpleChains.AbstractLoss{T}
    targets::Y
end

target(loss::BinaryLogitCrossEntropyLoss) = loss.targets
(loss::BinaryLogitCrossEntropyLoss)(::Int) = loss

function calculate_loss(loss::BinaryLogitCrossEntropyLoss, logits)
    y = loss.targets
    total_loss = zero(eltype(logits))
    for i in eachindex(y)
        p_i = inv(1 + exp(-logits[i]))
        y_i = y[i]
        total_loss -= y_i * log(p_i) + (1 - y_i) * (1 - log(p_i))
    end
    total_loss
end

function (loss::BinaryLogitCrossEntropyLoss)(previous_layer_output::AbstractArray{T}, p::Ptr, pu) where {T}
    total_loss = calculate_loss(loss, previous_layer_output)
    total_loss, p, pu
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
    gradient::AbstractArray{T},
) where {T}
    loss = getfield(layers, 1)
    y = loss.targets
    gradient .= previous_layer_output
    for i in eachindex(y)
        sign_arg = 2 * y[i] - 1
        logit_i = previous_layer_output[i]
        gradient[i] = -(sign_arg * inv(1 + exp(sign_arg * logit_i)))
    end
    return calculate_loss(loss, previous_layer_output), gradient, pu
end


xtrain3, ytrain0 = MLDatasets.MNIST.traindata()
xtest3, ytest0 = MLDatasets.MNIST.testdata()
xtrain4 = reshape(xtrain3, 28, 28, 1, :)
xtest4 = reshape(xtest3, 28, 28, 1, :)
ytrain1 = UInt32.(ytrain0 .+ 1)
ytest1 = UInt32.(ytest0 .+ 1)

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


lenetloss = SimpleChains.add_loss(lenet, BinaryLogitCrossEntropyLoss(ytrain1))



@time p = SimpleChains.init_params(lenet, size(xtrain4))
G = SimpleChains.alloc_threaded_grad(lenetloss)

@time SimpleChains.train_batched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 1)
SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)
SimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)
