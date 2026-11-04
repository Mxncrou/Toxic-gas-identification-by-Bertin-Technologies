using Lux, Optimisers, Zygote, Random

function bce(ŷ, y; ε=1f-7)
    ŷ = clamp.(ŷ, ε, 1 - ε) # Avoid numerical issues like log(0)
    return -sum(y .* log.(ŷ) .+ (1 .- y) .* log.(1 .- ŷ)) / length(y)
end

function train(model, X, Y; epochs, batch_size=1024, lr=2.5f-4, seed=42)
    rng = Xoshiro(seed)
    
    θ, state   = Lux.setup(rng, model)
    optimiser_state = Optimisers.setup(Adam(lr), θ)
    N               = size(X, 2)

    losses = Float32[]

    for epoch in 1:epochs
        loss = 0f0

        for idx in Iterators.partition(shuffle(rng, 1:N), batch_size)

            x, y = X[:, idx], Y[:, idx]
            𝓛, (∇𝓛,)  = Zygote.withgradient(θ -> bce(model(x, θ, state)[1], y), θ)
                
            optimiser_state, θ = Optimisers.update(optimiser_state, θ, ∇𝓛)
            loss += 𝓛 * length(idx) # for println the loss
        end
        
        push!(losses, loss / N)
        println("Epoch: $epoch, Loss: $(loss / N)")
    end
    
    return θ, state, losses
end