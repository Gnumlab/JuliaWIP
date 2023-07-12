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

xtrain3, ytrain0 = MLDatasets.MNIST.traindata(Float32)
xtest3, ytest0 = MLDatasets.MNIST.testdata(Float32)
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

lenetloss = BinaryLogitCrossEntropyLoss(ytrain1)
lenetloss = SimpleChains.add_loss(lenet, lenetloss)

@time p = SimpleChains.init_params(lenet, size(xtrain4))
G = SimpleChains.alloc_threaded_grad(lenetloss)


function batch(x, y, batch_size, i)
    start_idx = (i - 1) * batch_size + 1
    end_idx = min(length(y), i * batch_size)
    xbatch = x[:, :, :, start_idx:end_idx]
    ybatch = y[start_idx:end_idx]
    if length(ybatch) < batch_size
        pad_size = batch_size - length(ybatch)
        xbatch = cat(xbatch, zeros(size(x, 1), size(x, 2), size(x, 3), pad_size), dims=4)
        ybatch = vcat(ybatch, zeros(UInt32, pad_size))
    end
    return (xbatch, ybatch)
end

batch_size = 128
num_batches = div(length(ytrain1), batch_size)
train_batches = [batch(xtrain4, ytrain1, batch_size, i) for i in 1:num_batches]
train_data = [(x, y) for (x, y) in train_batches]

batch_size = 128
num_batches = div(length(ytrain1), batch_size)
train_batches = [batch(xtrain4, ytrain1, batch_size, i) for i in 1:num_batches]

@time SimpleChains.train!(
    G,
    p,
    lenetloss,
    train_batches,
    SimpleChains.ADAM(3e-4),
    10,
)
SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)
SimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)
