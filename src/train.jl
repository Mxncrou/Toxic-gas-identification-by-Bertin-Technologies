using Lux, Optimisers, Zygote, Random

function challenge_mse(ŷ, y)
    f = ifelse.(y .>= 0.5f0, 1.2f0, 1f0)
    f = sqrt.(f)
    return MSELoss()(f .* ŷ, f .* y)
end

function train(model, Xtrain, Ytrain, Xtest, Ytest; epochs, batch_size=1024, lr=5.0f-4, seed=42)
    rng = Xoshiro(seed)
    
    θ, state   = Lux.setup(rng, model)
    optimiser_state = Optimisers.setup(Adam(lr), θ)
    N               = size(Xtrain, 2)

    training_losses = Float32[]
    testing_losses  = Float32[]

    for epoch in 1:epochs
        loss = 0f0

        for idx in Iterators.partition(shuffle(rng, 1:N), batch_size)

            x, y = Xtrain[:, idx], Ytrain[:, idx]
            (𝓛, state), (∇𝓛,)  = Zygote.withgradient(θ) do θ
                ŷ, st = model(x, θ, state)
                challenge_mse(ŷ, y), st
            end
                
            optimiser_state, θ = Optimisers.update(optimiser_state, θ, ∇𝓛)
            loss += 𝓛 * length(idx) # for println the loss
        end
        
        loss = sqrt(loss / N)

        push!(training_losses, loss)
        
        evaluation = evaluate(model, θ, state, Xtest, Ytest).loss
        push!(testing_losses, evaluation)
        println("Epoch: $epoch, Loss: $(loss), Evaluated loss: $(evaluation)")
    end
    
    return θ, state, training_losses, testing_losses
end