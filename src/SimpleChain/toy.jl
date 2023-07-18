using SimpleChains

# Define the neural network model
model = SimpleChain(
    static(2),
    TurboDense(tanh, 32),
    TurboDense(tanh, 16),
    TurboDense(identity, 1)
)

# Initialize the model parameters
parameters = SimpleChains.init_params(model)

# Generate some random data
batch_size = 64
X = rand(Float32, 2, batch_size)
Y = rand(Bool, batch_size)

# Add the loss to the model with an L2 penalty
penalty_strength = 0.001
l2_penalty = L2Penalty(penalty_strength)
loss_function = SimpleChains.add_loss(model, SquaredLoss(Y))

# Create an optimizer object
learning_rate=3e-4
opt = SimpleChains.ADAM(learning_rate)

# Train the model for one iteration
gradients = SimpleChains.alloc_threaded_grad(model)
SimpleChains.valgrad!(gradients, loss_function, X, parameters)
SimpleChains.update!(gradients, opt, X, parameters)
