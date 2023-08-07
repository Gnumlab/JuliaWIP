using MLDatasets
import Random
xtrain3, ytrain0 = MLDatasets.MNIST.traindata(Float32);
xtest3, ytest0 = MLDatasets.MNIST.testdata(Float32);  #@deprecated use MNIST(split=:test)[:] instead
println(size(xtest3))
# (28, 28, 60000)
println(extrema(ytrain0)) # digits, 0,...,9
# (0, 9)

xtrain4 = reshape(xtrain3, 28, 28, 1, :);
xtest4 = reshape(xtest3, 28, 28, 1, :);
ytrain1 = UInt32.(ytrain0 .+ 1);
ytest1 = UInt32.(ytest0 .+ 1);

n_examples = 300
# Reduce the dataset to the selected examples
indices = Random.randperm(size(xtrain4, 4))[1:n_examples]
xtrain4 = xtrain4[:, :, :, 1:1:n_examples]
ytrain1 = ytrain1[1:1:n_examples]

println(extrema(ytrain1)) # digits, 0,...,

using SimpleChains

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

lenetloss = SimpleChains.add_loss(lenet, LogitCrossEntropyLoss(ytrain1));


@time p = SimpleChains.init_params(lenet, size(xtrain4));


estimated_num_cores = (Sys.CPU_THREADS ÷ ((Sys.ARCH === :x86_64) + 1));
G = SimpleChains.alloc_threaded_grad(lenetloss);

@time SimpleChains.train_batched!(G, p, lenetloss, xtrain4, SimpleChains.ADAM(3e-4), 10);
SimpleChains.accuracy_and_loss(lenetloss, xtrain4, p)
SimpleChains.accuracy_and_loss(lenetloss, xtest4, ytest1, p)




using SimpleChains
using NNlib
using Random, Statistics
using Parameters: @unpack
using Flux
using Flux: onehotbatch, onecold
using Flux.Losses: logitcrossentropy
using Flux.Losses: hinge_loss
using Flux.Data: DataLoader





batchsize = 0

use_cuda = false #CUDA.functional()
batchsize = 0
device = cpu
batchsize = n_examples#1 * Threads.nthreads()

function create_loader(x, y, batch_size, shuffle)
  y = Flux.onehotbatch(y, 1:10)
  DataLoader(
    (device(x), device(y));
    batchsize = batch_size,
    shuffle = shuffle
  )
end

ytrain1_onehot = Flux.onehotbatch(ytrain1, 1:10)
ytest1_onehot = Flux.onehotbatch(ytest1, 1:10)
println(ytrain1_onehot[1], "   ", ytest1_onehot[1])
train_loader = create_loader(xtrain4, ytrain1, batchsize, true)

pred = lenet(xtest4, p)
println("prediction")
println(pred[:,1])
println(ytest1[1])

m = xtest4[:,:,1,1]
m1 = reshape(m, 28,28,1,:)
pred1 = lenet(m1, p)
println(pred1)
println(size(pred1))
cnl, loss = SimpleChains.split_loss(lenetloss)
loss_target = SimpleChains.target(loss)
#println(loss_target)
class = zeros(Int, size(pred1))
class[ytest1[1]] = 1
println(class)

ytest1_onehot = Flux.onehotbatch(ytest1, 1:10)

# Compute the loss for a single example
loss = Flux.Losses.hinge_loss(pred1, ytest1_onehot[1])
# Print the loss value
println("Loss: $loss")


using StaticArrays

# Preallocate static array for m1
m1 = Array{Float64}(undef, 28, 28, 1)
@time for (x, y) in train_loader
    #println("cycle train_loader")
    #println(size(x))
    
    @inbounds for i in 1:size(x, 4)
        m1 .= x[:, :, 1, i]
        #m1 .= m#reshape(m, 28, 28)
        ypred = lenet(m1, p)
        #println(size(ypred))
        #println(size(y), "  ", y[i])
        loss_val = Flux.Losses.hinge_loss(ypred, y[i])
        #println("Loss: $loss_val")
    end
end

@time for (x, y) in train_loader
  #println("cycle train_loader")
  #println(size(x))
  for i in 1:size(x, 4)
    m = x[:, :, :, i]
    m1 = reshape(m, 28, 28, 1, 1)
    ypred = lenet(m1, p)
    #println(size(ypred))
    #println(size(y), "  ", y[i])
    loss_val = Flux.Losses.hinge_loss(ypred, y[i])
    #println("Loss: $loss_val")
  end
end

@time for (x, y) in train_loader
  #println("cycle train_loader")
  #println(size(x))
  ypredall = lenet(x, p)
  @inbounds for i in 1:size(x, 4)
    #m = x[:, :, :, i]
    #m1 = reshape(m, 28, 28, 1, 1)
    ypred = ypredall[i]
    #println(size(ypred), ypred, " \t\t  ", size(y), y[i])
    #println(size(y), "  ", y[i])
    #loss_val = Flux.Losses.logitbinarycrossentropy(ypred, y[i])
    loss_val = Flux.Losses.hinge_loss(ypred, y[i])

    #println("Loss: $loss_val")
  end
end