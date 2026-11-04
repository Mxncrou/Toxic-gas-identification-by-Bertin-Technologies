using Statistics


const FEATURES = ["M4", "M5", "M6", "M7", "M12", "M13", "M14", "M15", "S1", "S2", "S3", "R", "Humidity"]
const TARGETS  = ["c01", "c02", "c03", "c05", "c08", "c09", "c10", "c11", "c12", "c13", "c14", "c16", "c17", "c18", "c19", "c23"]

inverse_hyperbolic_sine(x, θ) = asinh(x * θ) / θ

function pre_process(X, Y)
    df = innerjoin(X, Y, on=:ID)
    
    θ1 = 1 / 1.0563292914209657 # 1/std of cluster M12–M15
    θ2 = 1 / 1.551442317682796  # 1/std of cluster M4–M7

    Humidity           = permutedims(df.Humidity)
    Sensors_cluster_1  = inverse_hyperbolic_sine.(permutedims(Matrix(df[:, ["M12", "M13", "M14", "M15"]])), θ1)
    Sensors_cluster_2  = inverse_hyperbolic_sine.(permutedims(Matrix(df[:, ["M4", "M5", "M6", "M7"]])),     θ2)
    R                  = log.(permutedims(df.R))
    Additional_sensors = log.(permutedims(Matrix(df[:, ["S1", "S2", "S3"]])))

    X = Float32.(vcat(Humidity, Sensors_cluster_1, Sensors_cluster_2, R, Additional_sensors))
    Y = Float32.(permutedims(Matrix(df[:, TARGETS])))

    shuffled = shuffle(Xoshiro(42), 1:size(X, 2))
    n_test   = size(X, 2) ÷ 10
    testing  = shuffled[1:n_test]
    training = shuffled[n_test+1:end]

    return X[:, training], Y[:, training], X[:, testing], Y[:, testing]
end