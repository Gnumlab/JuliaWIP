using MLDatasets
using SimpleChains
using NNlib
using Random, Statistics
using Parameters: @unpack

struct MulticlassHingeLoss{T<:AbstractVector{<:Integer}} <: SimpleChains.AbstractLoss{T}
    targets::T
end

SimpleChains.target(loss::MulticlassHingeLoss) = loss.targets
(loss::MulticlassHingeLoss)(t::AbstractVector) = MulticlassHingeLoss(t)

function calculate_loss(loss::MulticlassHingeLoss, logits)
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

function (loss::MulticlassHingeLoss)(previous_layer_output::AbstractArray{T}, p::Ptr, pu) where {T}
    total_loss = calculate_loss(loss, previous_layer_output)
    total_loss, p, pu
end

function SimpleChains.layer_output_size(::Val{T}, sl::MulticlassHingeLoss, s::Tuple) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end

function SimpleChains.forward_layer_output_size(::Val{T}, sl::MulticlassHingeLoss, s) where {T}
    SimpleChains._layer_output_size_no_temp(Val{T}(), sl, s)
end

function SimpleChains.chain_valgrad!(
    __,
    previous_layer_output::AbstractArray{T},
    layers::Tuple{MulticlassHingeLoss},
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








function my_train_unbatched_core!(
  c::SimpleChains.Chain,
  pu::Ptr{UInt8},
  g,
  pX,
  p,
  opt,
  t::AbstractArray,
  mpt
)
  chn = SimpleChains.getchain(c)
  @unpack layers = chn
  pen = SimpleChains.getpenalty(c)
  fl = Base.front(layers)
  ll = last(layers)
  sx = static_size(pX)
  optbuffer, pm = SimpleChains.optmemory(opt, p, pu)
  GC.@preserve p g begin
    for y ∈ t
      layers_y = (fl..., ll(y))
      SimpleChains.update!(g, opt, pX, layers_y, pen, sx, p, pm, optbuffer, mpt)
    end
  end
end







function my_train_unbatched!(
  p::AbstractVector,
  _chn::SimpleChains.Chain,
  X::AbstractArray,
  opt::SimpleChains.AbstractOptimizer,
  t
)
  chn = SimpleChains.getchain(_chn)
  pX = SimpleChains.maybe_static_size_arg(chn.inputdim, X)
  optoff = SimpleChains.optmemsize(opt, p)
  @unpack layers = chn
  glen = _SimpleChains.try_static(numparam(chn), static_length(params))
  numthreads = _numthreads()

  T = Base.promote_eltype(p, X)
  bytes_per_thread, total_bytes = SimpleChains.required_bytes(
    Val{T}(),
    layers,
    static_size(pX),
    optoff + align(glen) * numthreads,
    static(0),
    numthreads
  )
  GC.@preserve X begin
    with_memory(
      my_train_unbatched_core!,
      _chn,
      total_bytes,
      pX,
      p,
      opt,
      t,
      bytes_per_thread
    )
  end
  p
end

function my_train_unbatched_core!(
  c::SimpleChains.Chain,
  pu::Ptr{UInt8},
  pX,
  it,
  p::AbstractVector{T},
  opt,
  mpt
) where {T}
  numthreads = _SimpleChains.numthreads()
  glen = _SimpleChains.try_static(numparam(getchain(c)), static_length(params))
  aligned_glen = SimpleChains.align(glen)
  g = _alloc_grad(Ptr{T}(pu), glen, numthreads, aligned_glen)
  offset = SimpleChains.static_sizeof(T) * aligned_glen * numthreads
  SimpleChains.train_unbatched_core!(c, pu + offset, g, pX, it, p, opt, mpt)
end





function full_train_unbatched_core!(
  model::Flux.Chain,
  pu::Ptr{UInt8},
  g,
  pX,
  p,
  opt,
  iters::Int,
  mpt
)
  chn = SimpleChains.getchain(model)  # Assume SimpleChains.Chain is equivalent to Flux.Chain
  @unpack layers = chn
  pen = SimpleChains.getpenalty(model)  # Assuming this function extracts penalty if any
  sx = SimpleChains.static_size(pX)
  optbuffer, pm = SimpleChains.optmemory(opt, p, pu)
  GC.@preserve p g begin
    for _ ∈ 1:iters
      # println("Third my_train_unbatched_core!", i)
      gs = Flux.gradient(ps) do
        #ŷ = SimpleChains: need evaluate function that evaluetes chn on input x  
        loss(ŷ, y)
      end
      SimpleChains.update!(gs, opt, pX, layers, pen, sx, p, pm, optbuffer, mpt)  # Assuming SimpleChains.update! updates the model weights
    end
  end
end


function my_train_unbatched_core!(
  c::SimpleChains.Chain,
  pu::Ptr{UInt8},
  g,
  pX,
  p,
  opt,
  iters::Int,
  mpt
)
  println("Third my_train_unbatched_core!")
  chn = SimpleChains.getchain(c)
  @unpack layers = chn
  pen = SimpleChains.getpenalty(c)
  sx = SimpleChains.static_size(pX)
  optbuffer, pm = SimpleChains.optmemory(opt, p, pu)
  GC.@preserve p g begin
    for _ ∈ 1:iters
      #println("Third my_train_unbatched_core!", i)
      SimpleChains.update!(g, opt, pX, layers, pen, sx, p, pm, optbuffer, mpt)
    end
  end
end


function my_train_unbatched!(
  g,
  p::AbstractVector,
  _chn::SimpleChains.Chain,
  X,
  opt::SimpleChains.AbstractOptimizer,
  t
)

  println("Second my_train_unbatched!")
  if g isa AbstractMatrix && SimpleChains.static_size(g, static(2)) == 1
    gpb = preserve_buffer(g)
    gv = PtrArray(pointer(g), (length(p),))
    GC.@preserve gpb my_train_unbatched!(gv, p, _chn, X, opt, t)
    return p
  end

  chn = SimpleChains.getchain(_chn)
  pX = SimpleChains.maybe_static_size_arg(chn.inputdim, X)
  optoff = SimpleChains.optmemsize(opt, p)
  @unpack layers = chn
  T = Base.promote_eltype(p, X)
  bytes_per_thread, total_bytes = SimpleChains.required_bytes(
    Val{T}(),
    layers,
    SimpleChains.static_size(pX),
    optoff,
    static(0),
    SimpleChains.static_size(g, static(2))
  )
  GC.@preserve X begin
    SimpleChains.with_memory(
      my_train_unbatched_core!,
      _chn,
      total_bytes,
      g,
      pX,
      p,
      opt,
      t,
      bytes_per_thread
    )
  end
  p
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
xtrain4 = xtrain4[:, :, :, 1:2:2000]
ytrain1 = ytrain1[1:2:2000]

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
loss_fn = MulticlassHingeLoss(ytrain1)

println("loss")

# Add the loss function to the model
lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain1));
#lenetloss = SimpleChains.add_loss(lenet, loss_fn)

println("lenetloss")
# Initialize the parameters of the model
p = SimpleChains.init_params(lenet, size(xtrain4))
G = SimpleChains.alloc_threaded_grad(lenetloss);

println("ini params")

#SimpleChains.valgrad!(G, lenetloss, xtrain4, p)

println("valgrad")

@time my_train_unbatched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 1);
#SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)
SimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)




#SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)

# Train the model using the added loss function
#@time SimpleChains.train_batched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 10)
