using MLDatasets
using SimpleChains
using NNlib
using Random, Statistics
using Parameters: @unpack
using Flux
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Flux.Losses: hinge_loss
using Flux.Data: DataLoader


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
    #println("chain_valgrad! ", total_loss)

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






#Functions taken from SimpleChains

function single_train_unbatched_core!(
  model::SimpleChains.Chain,
  pu::Ptr{UInt8},
  g,
  pX,
  p,
  opt,
  mpt
)


  #println("single_train_unbatched_core!")

  chn = SimpleChains.getchain(model)  # Assume SimpleChains.Chain is equivalent to Flux.Chain
  @unpack layers = chn
  pen = SimpleChains.getpenalty(model)  # SimpleChains.Assuming this function extracts penalty if any
  sx = SimpleChains.static_size(pX)
  optbuffer, pm = SimpleChains.optmemory(opt, p, pu)
  GC.@preserve p g begin
    #println("before update")
#      good_my_update!(g, opt, pX1, layers, pen, sx1, p, pm, optbuffer, mpt) 
    #good_my_update!(g, opt, pX, layers, pen, sx, p, pm, optbuffer, pu) 
    SimpleChains.update!(g, opt, pX, layers, pen, sx, p, pm, optbuffer, mpt) 
    
  end
end
using StaticArrays  # Make sure StaticArrays is imported

function full_train_unbatched_core!(
  model::SimpleChains.Chain,
  pu::Ptr{UInt8},
  g,
  pX,
  pY, #added code
  p,
  opt,
  iters::Int
)

  chn = SimpleChains.getchain(model)  # Assume SimpleChains.Chain is equivalent to Flux.Chain
  @unpack layers = chn
  GC.@preserve p g begin
    for i ∈ 1:iters   #iters are actually the number of epochs 

      actual_num_non_zero_loss_data = 0
       
      pred = model1(pX, p)
      @inbounds for i in 1:n_examples
        #loss_val = Flux.Losses.logitbinarycrossentropy(pred[i], pY[i])
        loss_val = Flux.Losses.hinge_loss(pred[i], pY[i])
        if loss_val > 0.0
          actual_num_non_zero_loss_data += 1
          @inbounds  non_zero_loss_data[:, :, :, actual_num_non_zero_loss_data] .=  @view pX[:, :, :, i:i]
        end
      end

      # Remove unnecessary computations
      println("\t\tnumber non zero = $actual_num_non_zero_loss_data of $n_examples (", actual_num_non_zero_loss_data/(n_examples*1.0), "%).")
      if actual_num_non_zero_loss_data == 0
        return
      end
      #non_zero_loss_data1 =  non_zero_loss_data[:, :, :, 1:actual_num_non_zero_loss_data]
      

      # Create an array `pX` to store the non-zero examples

      pX1 = @view non_zero_loss_data[:, :, :, 1:actual_num_non_zero_loss_data]
      # Assign the slices to the dynamic array
      pX2 = SimpleChains.maybe_static_size_arg(chn.inputdim, pX1)


      optoff = SimpleChains.optmemsize(opt, p)
      @unpack layers = chn
      T = Base.promote_eltype(p, pX2)
      bytes_per_thread, total_bytes = SimpleChains.required_bytes(    
        Val{T}(),
        layers,
        SimpleChains.static_size(pX2),
        optoff,
        static(0),
        SimpleChains.static_size(g, static(2))
      )
   
 
      GC.@preserve pX begin
        SimpleChains.with_memory(
          single_train_unbatched_core!,
          model,
          total_bytes,
          g,
          pX2,
          p,
          opt,
          bytes_per_thread
        )
      end
    end
  end
end




function my_train_unbatched!(
  g,
  p::AbstractVector,
  _chn::SimpleChains.Chain,
  X,
  Y,   #added code
  opt::SimpleChains.AbstractOptimizer,
  t
)

  #println("my_train_unbatched!")
  if g isa AbstractMatrix && SimpleChains.static_size(g, static(2)) == 1
    gpb = SimpleChains.preserve_buffer(g)
    gv = SimpleChains.PtrArray(pointer(g), (length(p),))
    GC.@preserve gpb my_train_unbatched!(gv, p, _chn, X, Y, opt, t) #modified by addding Y
    return p
  end

  chn = SimpleChains.getchain(_chn)
  pX = SimpleChains.maybe_static_size_arg(chn.inputdim, X)

  #pY = Y#SimpleChains.maybe_static_size_arg(chn.inputdim, Y)  #added code

  optoff = SimpleChains.optmemsize(opt, p)
  @unpack layers = chn
  T = Base.promote_eltype(p, X)
  bytes_per_thread, total_bytes = SimpleChains.required_bytes(    #should we do the same for Y?
    Val{T}(),
    layers,
    SimpleChains.static_size(pX),
    2optoff,
    static(0),
    SimpleChains.static_size(g, static(2))
  )
  #println("my_train_unbatched! calling my_train_unbatched_core!")

  
  GC.@preserve X begin
    SimpleChains.with_memory(
      full_train_unbatched_core!,
      _chn,
      total_bytes,
      g,
      pX,
      Y,  #added code for full_train_unbatched_core
      p,
      opt,
      t
    )
  end
  p
end


# Define the LeNet-5 architecture with multi-class output

# Load the MNIST dataset
xtrain3, ytrain0 = MLDatasets.CIFAR10.traindata(Float32)
xtest3, ytest0 = MLDatasets.CIFAR10.testdata(Float32)
xtrain4 = reshape(xtrain3, 32, 32, 3, :)
xtest4 = reshape(xtest3, 32, 32, 3, :)
ytrain1 = UInt32.(ytrain0 .+ 1)
ytest1 = UInt32.(ytest0 .+ 1)

n_examples = 1000
# Reduce the dataset to the selected examples
indices = Random.randperm(size(xtrain4, 4))[1:n_examples]
xtrain4 = @view xtrain4[:, :, :, 1:1:n_examples]
ytrain1 = @view ytrain1[1:1:n_examples]

max_possible_data = 50000
non_zero_loss_data = similar(xtrain4, Float32, 32, 32, 3, n_examples)
#non_zero_loss_data = SMatrix{28, 28, Float32, n_examples}(undef)

pred = similar(ytrain1, Float32, 10, n_examples)



function LeNet5()
  return SimpleChain(
  (static(32), static(32), static(3)),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 6),
  SimpleChains.MaxPool(2, 2),
  SimpleChains.Conv(SimpleChains.relu, (5, 5), 16),
  SimpleChains.MaxPool(2, 2),
  Flatten(3),
  TurboDense(SimpleChains.relu, 120),
  TurboDense(SimpleChains.relu, 84),
  TurboDense(identity, 10)
  ) 
end


using StaticArrays

# Declare a static array with 10 floats
#ypred = SVector{10, Float32}(undef)
# Set the random seed for reproducibility
Random.seed!(1234)

# Select a random subset of 10% of the examples
n = length(ytrain1)




# Print the size of the reduced dataset
println("Size of reduced dataset: $(size(5xtrain4, 4)) examples")

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
    println("Examples in ytest1 that are outside the range of 1 to 10:")ch
    for i in test_out_of_range
        println("ytest1[$i] = $(ytest1[i])")
    end
else
    println("All examples in ytest1 are within the range of 1 to 10.")
end

# Define the loss function for the model and input data
loss_fn = MulticlassHingeLoss(ytrain1)

#println("loss")

# Add the loss function to the model3)
#lenetloss = SimpleChains.add_loss(lenet, loss_fn)



println("ini params")



println("TRAINING WITH $n_examples EXAMPLE\n")
model1 = LeNet5()
for iter in 5:5
  println("Iteration n $iter")
  for i in 1:10
    lenet = LeNet5()
    # Initialize the parameters of the model
    #lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain1))
    lenetloss = SimpleChains.add_loss(lenet, loss_fn)
    p = SimpleChains.init_params(lenet, size(xtrain4))
    G = SimpleChains.alloc_threaded_grad(lenetloss);
    model1 = SimpleChains.remove_loss(lenet)

    @time my_train_unbatched!(G, p, lenetloss, xtrain4, ytrain1, SimpleChains.ADAM(3e-4), iter);
    #@time SimpleChains.train_unbatched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 1);
    println(SimpleChains.accuracy_and_loss(lenetloss, xtest4, p))
    #println(SimpleChains.accuracy_and_loss(lenetloss, xtest4,i,:
  end
  println("\n\n\n\n\n")
#lenet(xtest4, p)
end

println("STANDARD SIMPLECHAINS\n")

for iter in 5:5
  println("Iteration n $iter")
  for i in 1:5

    #@time SimpleChains.train_unbatched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 1);
    #SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)
    #println(SimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p))

    # Initialize the parameters of the model
    lenet1 = LeNet5()
    lenetloss1 = SimpleChains.add_loss(lenet1, loss_fn)
    p = SimpleChains.init_params(lenet1, size(xtrain4))
    G = SimpleChains.alloc_threaded_grad(lenetloss1);

    @time SimpleChains.train_batched!(G, p, lenetloss1, xtrain4, SimpleChains.ADAM(3e-4), iter);
    println(SimpleChains.accuracy_and_loss(lenetloss1, xtest4, p))
  end
  println("\n\n\n\n\n")
#lenet(xtest4, p)
end


#SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)

# Train the model using the added loss function
#@time SimpleChains.train_batched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 10)
