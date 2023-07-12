using SimpleChains, Flux, Plots

# One input variable with 200 observations
x = rand(200)

# One response variable with 200 observations (-1 or 1)
y = sign.(randn(200))

schain = SimpleChain(
  static(1), # input dimension (optional)
  TurboDense{true}(tanh, 8), # dense layer with bias that maps to 8 outputs and applies `tanh` activation
  SimpleChains.Dropout(0.2), # dropout layer
  TurboDense{false}(identity, 1), # dense layer without bias that maps to 1 output and `identity` activation
  (x, y) -> Flux.hinge_loss(TurboDense{false}(identity, 1)(TurboDense{true}(tanh, 8)(x)), y)
)

# Initialize the parameters
params = SimpleChains.init_params(schain)

# Define the loss function, optimizer, and training data
loss(x, y) = Flux.hinge_loss(TurboDense{false}(identity, 1)(TurboDense{true}(tanh, 8)(x)), y)
opt = ADAM()
data = [(x, y)]

# Train the neural network
Flux.train!(loss, params, data, opt)

# Make predictions on a grid of input values
xgrid = range(-3, 3, length=100)
ygrid = [applychain(schain, [x]) for x in xgrid]
yhat = Flux.sign.(ygrid)

# Plot the true and predicted outputs
plot(xgrid, ygrid, label="Predicted")
scatter!(x, y, label="True")
plot!(xgrid, yhat, label="Predicted (Discrete)", linestyle=:dash)
params = Flux.params(schain)
nparams = length(params)

# Entirely in place evaluation
#@benchmark valgrad!($g, $schain, $x, $params) # dropout active
