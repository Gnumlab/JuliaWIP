using SimpleChains

struct MyHingeLoss{T,Y<:AbstractVector{T}} <: SimpleChains.AbstractLoss{T}
    targets::Y
end

target(loss::MyHingeLoss) = loss.targets
(loss::MyHingeLoss)(::Int) = loss

function calculate_loss(loss::MyHingeLoss, logits)
    y = loss.targets
    total_loss = zero(eltype(logits))
    for i in eachindex(y)
        total_loss += max(0, 1 - y[i] * logits[i])
    end
    total_loss
end

function (loss::MyHingeLoss)(previous_layer_output::AbstractArray{T}, p::Ptr, pu) where {T}
    total_loss = calculate_loss(loss, previous_layer_output)
    total_loss, p, pu
end

function SimpleChains.layer_output_size(::Val{T}, sl::MyHingeLoss, s::Tuple) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end

function SimpleChains.forward_layer_output_size(::Val{T}, sl::MyHingeLoss, s) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end

function SimpleChains.chain_valgrad!(
    __,
    previous_layer_output::AbstractArray{T},
    layers::Tuple{MyHingeLoss},
    _::Ptr,
    pu::Ptr{UInt8},
) where {T}
    loss = getfield(layers, 1)
    total_loss = calculate_loss(loss, previous_layer_output)
    y = loss.targets

    # Store the backpropagated gradient in the previous_layer_output array.
    for i in eachindex(y)
        if y[i] * previous_layer_output[i] < 1
            previous_layer_output[i] = -y[i]
        else
            previous_layer_output[i] = zero(eltype(previous_layer_output))
        end
    end

    return total_loss, previous_layer_output, pu
end



model = SimpleChain(
    static(2),
    TurboDense(tanh, 32),
    TurboDense(tanh, 16),
    TurboDense(identity, 1)
)

batch_size = 64
X = rand(Float32, 2, batch_size)
Y = rand([-1, 1], batch_size)

parameters = SimpleChains.init_params(model);
gradients = SimpleChains.alloc_threaded_grad(model);

# Add the loss like any other loss type
model_loss = SimpleChains.add_loss(model, MyHingeLoss(Y));


SimpleChains.valgrad!(gradients, model_loss, X, parameters)

