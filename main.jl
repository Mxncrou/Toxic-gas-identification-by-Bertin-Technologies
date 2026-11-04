using Lux, Optimisers, Zygote, Random
using CSV, DataFrames, Plots
include("src/model.jl")
include("src/train.jl")
include("utils/preprocess.jl")
include("utils/register_runs.jl")


X  = CSV.read("data/x_train_T9QMMVq.csv", DataFrame)
Y  = CSV.read("data/y_train_R0MqWmu.csv", DataFrame)

Xtrain, Ytrain, Xtest, Ytest = pre_process(X, Y)

θ, state, train_losses, test_losses = train(network(), Xtrain, Ytrain, Xtest, Ytest; epochs=1000)
register_run(train_losses, test_losses; model="256 neurons 3 layers MLP tanh/tanh/sigmoid")