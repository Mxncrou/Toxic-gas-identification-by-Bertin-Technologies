using DataFrames, CSV, Dates

function register_run(training_losses, testing_losses; model::String, path::String = "runs/losses.csv")
    n  = length(training_losses)
    df = DataFrame(run   = fill(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), n),
                   model = fill(model, n),
                   epoch = 1:n,
                   train_loss = training_losses,
                   test_loss  = testing_losses)
    mkpath(dirname(path))
    CSV.write(path, df; append = isfile(path) && filesize(path) > 0)
    return nothing
end