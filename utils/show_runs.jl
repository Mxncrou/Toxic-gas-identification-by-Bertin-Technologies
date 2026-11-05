using CSV, DataFrames, Plots

function show_runs(path::String = "runs/losses.csv")
    df = CSV.read(path, DataFrame)
    p  = plot(xlabel = "epoch", ylabel = "loss",
              yscale = :log10, size = (1300, 800), margin = 5Plots.mm)

    for (i, g) in enumerate(groupby(df, [:run, :model]))
        label = "$(g.model[1]) ($(g.run[1]))"
        plot!(p, g.epoch, g.train_loss; color = i, label = label * " train")
        plot!(p, g.epoch, g.test_loss;  color = i, linestyle = :dash, label = label * " test")
    end

    display(p)
    return p
end

show_runs()