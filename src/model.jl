using Lux

const xshape = 13
const yshape = 16
const d1     = 256
const d2     = 32

MLP() =
    Chain(
        Dense(xshape => d1,     tanh,    init_weight = orthogonal(; gain = 5f0/3), init_bias = zeros32), Dropout(0.1),
        Dense(    d1 => d1,     tanh,    init_weight = orthogonal(; gain = 5f0/3), init_bias = zeros32), Dropout(0.1),
        Dense(    d1 => yshape, sigmoid, init_weight = orthogonal(; gain = 1f0),   init_bias = zeros32)
    )

HumidityFilmnet() = @compact(
    encoder = Dense(xshape -1 => d1, init_weight = orthogonal(; gain = 5f0/3), init_bias = zeros32), # tanh applied in do X
    humiditynet = Chain(
        Dense(1  => d2, tanh, init_weight = orthogonal(; gain = 5f0/3), init_bias = zeros32),
        Dense(d2 => 2 * d1;   init_weight = zeros32,                    init_bias = zeros32)
    ),
    head = Chain(
        Dropout(0.1),
        Dense(d1 => d1,     tanh,    init_weight = orthogonal(; gain = 5f0/3), init_bias = zeros32), Dropout(0.1),
        Dense(d1 => yshape, sigmoid, init_weight = orthogonal(; gain = 1f0),   init_bias = zeros32)
    )
) do X
    humidity, sensors = X[1:1, :], X[2:end, :]
    x  = encoder(sensors)
    γβ = humiditynet(humidity)
    x  = tanh.(x .* (1 .+ γβ[1:d1, :]) .+ γβ[d1+1:end, :])
    @return head(x) 
end