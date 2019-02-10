module StatsBase
    import Statistics: mean, mean!, var, varm, varm!, std, stdm, cov, covm,
                       median, middle
    export
    ConvergenceException
    include("common.jl")
    include("scalarstats.jl")
end # module
