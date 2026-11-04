using Lux, Optimisers, Zygote, Random
using CSV, DataFrames, Plots
include("src/model.jl")
include("src/train.jl")
include("src/evaluate.jl")


x  = CSV.read("data/x_train_T9QMMVq.csv", DataFrame)
y  = CSV.read("data/y_train_R0MqWmu.csv", DataFrame)
df = innerjoin(x, y, on=:ID)

const FEATURES = ["M4", "M5", "M6", "M7", "M12", "M13", "M14", "M15", "S1", "S2", "S3", "R", "Humidity"]
const TARGETS  = ["c01", "c02", "c03", "c05", "c08", "c09", "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c18", "c19", "c23"] 

X = Float32.(permutedims(Matrix(df[:, FEATURES])))
Y = Float32.(permutedims(Matrix(df[:, TARGETS])))

perm  = shuffle(Xoshiro(42), 1:size(X, 2))
n_val = size(X, 2) ÷ 10
validation, training = perm[1:n_val], perm[n_val+1:end]

θ, state, losses = train(network(), X[:, training], Y[:, training]; epochs=100)
print("Validation loss: ", evaluate(network(), θ, state, X[:, validation], Y[:, validation]))
plot(losses; xlabel="epoch", ylabel="training loss", legend=false)