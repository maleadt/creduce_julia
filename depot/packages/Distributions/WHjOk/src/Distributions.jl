module Distributions
import StatsBase: kurtosis, skewness, entropy, mode, modes,
                  params, params!
export
    Univariate,
    Continuous,
    wsample, wsample!      # weighted sampling from a source array
include("common.jl")
include("utils.jl")
end # module
