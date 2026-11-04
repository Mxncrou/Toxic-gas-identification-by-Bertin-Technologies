using Lux

const xshape = 13
const yshape = 16

network() = Chain(    
    Dense(xshape => 64, relu),
    Dense(64 => 64, relu),
    Dense(64 => yshape, sigmoid),
)