using Lux, Optimisers, Zygote, Random
using CSV, DataFrames, Plots
include("src/model.jl")
include("src/train.jl")
include("utils/preprocess.jl")
include("utils/register_runs.jl")


X  = CSV.read("data/x_train_T9QMMVq.csv", DataFrame)
Y  = CSV.read("data/y_train_R0MqWmu.csv", DataFrame)

Xtrain, Ytrain, Xtest, Ytest = pre_process(X, Y, 0.0)

model = HumidityFilmnet()
θ, state, train_losses, test_losses = train(model, Xtrain, Ytrain, Xtest, Ytest; epochs=1000, for_sub=true)
# register_run(train_losses, test_losses; model = "HumidityFilmnet; d1=256, d2=32, Dropout=0.1 all but Humidity MLP")

Xs   = CSV.read("data/x_test_9F13O5s.csv", DataFrame)
Xsub = pre_process_submission(Xs)

Ŷ, _  = model(Xsub, θ, Lux.testmode(state))

sub = DataFrame(permutedims(Ŷ), TARGETS)
sub.c07 = sub.c20 = sub.c02
sub.c04 = sub.c22 = sub.c03
sub.c06 = sub.c21 = sub.c05
sub.c15 = zeros(Float32, nrow(sub))
insertcols!(sub, 1, :ID => Xs.ID)

CSV.write("data/submission.csv", sub[:, ["ID"; ["c" * lpad(i, 2, '0') for i in 1:23]]])