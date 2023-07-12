using MLDatasets
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
