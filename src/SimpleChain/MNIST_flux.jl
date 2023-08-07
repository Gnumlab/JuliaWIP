# Comparison of Flux and SimpleChains implementations of classification of
# MNIST dataset with the convolutional neural network known as LeNet5.
# Each implementation is run twice so the second runs will demonstrate
# the performance after compilation has been performed.

import MLDatasets
import Random

# Get MNIST data
function get_data(split)
  x, y = MLDatasets.MNIST(split)[:]
  (reshape(x, 28, 28, 1, :), UInt32.(y .+ 1))
end

xtrain, ytrain = get_data(:train)
img_size = Base.front(size(xtrain))
xtest, ytest = get_data(:test)

# Reduce the dataset to the selected examples
#n_examples = 3000
#xtrain = xtrain[:, :, :, 1:1:n_examples]
#ytrain = ytrain[1:1:n_examples]


n_examples = 500
indices = Random.randperm(size(xtrain, 4))[1:n_examples]
xtrain = xtrain[:, :, :, indices]
ytrain = ytrain[indices]



# Training parameters
num_image_classes = 10
learning_rate = 3e-4
num_epochs = 10

using Printf

function display_loss(accuracy, loss)
  @printf("    training accuracy %.2f, loss %.4f\n", 100 * accuracy, loss)
end



# Flux implementation
begin
  using Flux
  using Flux.Data: DataLoader
  using Flux.Optimise: Optimiser, WeightDecay
  using Flux: onehotbatch, onecold
  using Flux.Losses: logitcrossentropy
  using Flux.Losses: hinge_loss
  using Statistics, Random
  using StaticArrays

  batchsize = 0
 
  use_cuda = false #CUDA.functional()
  batchsize = 0
  device = cpu
  batchsize = 1 * Threads.nthreads()



  function create_loader(x, y, batch_size, shuffle)
    y = onehotbatch(y, 1:num_image_classes)
    DataLoader(
      (device(x), device(y));
      batchsize = batch_size,
      shuffle = shuffle
    )
  end

  train_loader = create_loader(xtrain, ytrain, batchsize, true)
  test_loader = create_loader(xtest, ytest, batchsize, false)

  function LeNet5()
    out_conv_size = (img_size[1] ÷ 4 - 3, img_size[2] ÷ 4 - 3, 16)

    return Chain(
      Flux.Conv((5, 5), img_size[end] => 6, Flux.relu),
      Flux.MaxPool((2, 2)),
      Flux.Conv((5, 5), 6 => 16, Flux.relu),
      Flux.MaxPool((2, 2)),
      Flux.flatten,
      Flux.Dense(prod(out_conv_size), 120, Flux.relu),
      Flux.Dense(120, 84, Flux.relu),
      Flux.Dense(84, num_image_classes)
    ) |> device
  end

  loss(ŷ, y) = logitcrossentropy(ŷ, y)
  #loss(ŷ, y) = hinge_loss(ŷ, y)
  function eval_loss_accuracy(loader, model, device)
    l = 0.0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
      x, y = x |> device, y |> device
      ŷ = model(x)
      l += loss(ŷ, y) * size(x)[end]
      acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
      ntot += size(x)[end]
    end
    return (acc = acc / ntot, loss = l / ntot)
  end
  
  
  # Pre-allocate the array with an upper bound size (e.g., maximum number of data points in the dataset)
  max_possible_data = 10000
  #non_zero_loss_data = Vector{Tuple{AbstractArray, AbstractArray}}(undef, max_possible_data)
  non_zero_loss_data = Vector{Tuple{Array{Float32, 4}, Array{Float32, 2}}}(undef, max_possible_data)


  function fill_non_zero_loss_data(model, train_loader)
    actual_num_non_zero_loss_data = 0
    count = 0
    count2 = 0
    max = 0
    min = 100
    for (x, y) in train_loader
        x = device(x)
        y = device(y)
        #println(model(x))

        loss_val = loss(model(x), y)
        #println(model(x), " --- ", y, " ------- ", typeof(model(x)), " --- ", typeof(y))
        count2 += 1
        if loss_val > 0.05
            actual_num_non_zero_loss_data += 1
            non_zero_loss_data[actual_num_non_zero_loss_data] = (x, y)
            count += 1
        end


        if loss_val > max

          max = loss_val
        end
        if loss_val < min
          min = loss_val
        end
    end
    #println(max, " ", min, " ", count, " ", count2)
    return actual_num_non_zero_loss_data
  end

  function do_gradient(model, ps, non_zero_loss_loader, opt)
    for (x, y) in non_zero_loss_loader
      gs = Flux.gradient(ps) do
          ŷ = model(x)
          loss(ŷ, y)
      end
      Flux.Optimise.update!(opt, ps, gs)
    end
  end

  function my_fast_train!(model, train_loader, opt)
    ps = Flux.params(model)
    
    actual_num_non_zero_loss_data = 0
    for epoch = 1:num_epochs
        actual_num_non_zero_loss_data = fill_non_zero_loss_data(model, train_loader)

        if actual_num_non_zero_loss_data == 0
          println(epoch)
          return
        end
#        if actual_num_non_zero_loss_data == 0
#            println("No non-zero loss data in epoch $epoch.")
#            continue
#        end
#        println(actual_num_non_zero_loss_data)

        # Remove unnecessary computations
        non_zero_loss_data_batched = @view(non_zero_loss_data[1:actual_num_non_zero_loss_data])
        
        # Simplify batching process using loop
        x_batch = cat([non_zero_loss_data_batched[i][1] for i in 1:actual_num_non_zero_loss_data]..., dims=4) |> device
        y_batch = hcat([non_zero_loss_data_batched[i][2] for i in 1:actual_num_non_zero_loss_data]...) |> device

        
        non_zero_loss_loader = DataLoader((x_batch, y_batch), batchsize=batchsize, shuffle=true)

        do_gradient(model, ps, non_zero_loss_loader, opt)
        
    end
  end

  
  
  
  function my_train!(model, train_loader, opt)
    ps = Flux.params(model)
    for epoch = 1:num_epochs
        non_zero_loss_data = []
        for (x, y) in train_loader
            x = device(x)
            y = device(y)
            loss_val = loss(model(x), y)
            if loss_val != 0
                push!(non_zero_loss_data, (x, y))
            end
        end
        if isempty(non_zero_loss_data)
            println("No non-zero loss data in epoch $epoch.")
            continue
        end
        x_batch = cat([non_zero_loss_data[i][1] for i in 1:length(non_zero_loss_data)]..., dims=4)
        y_batch = hcat([non_zero_loss_data[i][2] for i in 1:length(non_zero_loss_data)]...)
        non_zero_loss_loader = DataLoader((x_batch, y_batch), batchsize=batchsize, shuffle=true)
        
        for (x, y) in non_zero_loss_loader
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                loss(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
    end
  end





  function train!(model, train_loader, opt)
    ps = Flux.params(model)
    for _ = 1:num_epochs
      for (x, y) in train_loader
        x = device(x)
        y = device(y)
        gs = Flux.gradient(ps) do
          ŷ = model(x)
          loss(ŷ, y)
        end
        Flux.Optimise.update!(opt, ps, gs)
      end
    end
  end




  println("\n\n\n============================== MY TRAIN ==============================")
  for run = 1:2
    #=
    println("Flux my_train! #$run")
    @time "  create model" model = LeNet5()
    opt = ADAM(learning_rate)
    @time "  train $num_epochs epochs" my_train!(model, train_loader, opt)
    =#

    #@time "  compute training loss" train_acc, train_loss =
    #  eval_loss_accuracy(test_loader, model, device)
    #display_loss(train_acc, train_loss)
    #@time "  compute test loss" test_acc, test_loss =
    #  eval_loss_accuracy(train_loader, model, device)
    #display_loss(test_acc, test_loss)
  end
  
  println("\n\n\n============================== MY FAST TRAIN ==============================")
  for run = 1:2
    println("\n\n\n\n\nFlux my_train! #$run")
    @time "  create model" model = LeNet5()
    opt = ADAM(learning_rate)
    @time "  ========================================\n TRAIN $num_epochs epochs" my_fast_train!(model, train_loader, opt)
    @time "  compute training loss" train_acc, train_loss =
      eval_loss_accuracy(test_loader, model, device)
    display_loss(train_acc, train_loss)
    @time "  compute test loss" test_acc, test_loss =
      eval_loss_accuracy(train_loader, model, device)
    display_loss(test_acc, test_loss)
  end
  
    println("\n\n\n============================== TRAIN ==============================")
  for run = 1:2
    println("\n\n\n\n\nFlux train! #$run")
    @time "  create model" model = LeNet5()
    opt = ADAM(learning_rate)
    @time " ========================================\n TRAIN $num_epochs epochs" train!(model, train_loader, opt)
    @time "  compute training loss" train_acc, train_loss =
      eval_loss_accuracy(test_loader, model, device)
    display_loss(train_acc, train_loss)
    @time "  compute test loss" test_acc, test_loss =
      eval_loss_accuracy(train_loader, model, device)
    display_loss(test_acc, test_loss)
  end
  
end






















