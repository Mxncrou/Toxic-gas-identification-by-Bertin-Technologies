using CSV, DataFrames, Plots

function show_runs(path::String = "runs/losses.csv")
    df = CSV.read(path, DataFrame)
    p  = plot(xlabel = "epoch", ylabel = "loss")

    for (i, g) in enumerate(groupby(df, [:run, :model]))
        label = "$(g.model[1]) ($(g.run[1]))"
        plot!(p, g.epoch, g.train_loss; color = i, label = label * " train")
        plot!(p, g.epoch, g.test_loss;  color = i, linestyle = :dash, label = label * " test")
    end

    display(plot(p, plot(p; yscale = :log10); layout = (2, 1), size = (1200, 1000), margin = 5Plots.mm))
end

show_runs()