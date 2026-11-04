using Lux

const xshape = 13
const yshape = 16

network() =
    Chain(
        Dense(xshape => 256,    tanh), 
        Dense(   256 => 256,    tanh),
        Dense(   256 => yshape, sigmoid)
    )