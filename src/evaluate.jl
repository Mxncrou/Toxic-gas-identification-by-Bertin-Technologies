using Lux

function evaluate(model, θ, state, X, Y)
    Ŷ, _ = model(X, θ, Lux.testmode(state))
    return (loss = sqrt(challenge_mse(Ŷ, Y)),)
end