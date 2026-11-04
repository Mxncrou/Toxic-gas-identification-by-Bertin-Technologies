using Lux

const xshape = 13
const yshape = 16

network() =
    Chain(
        Dense(xshape => 128,    tanh), 
        Dense(   128 => 128,    tanh),
        Dense(   128 => yshape, sigmoid)
    )