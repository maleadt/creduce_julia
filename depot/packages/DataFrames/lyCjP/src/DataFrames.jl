module DataFrames
using Reexport, StatsBase, SortingAlgorithms, Compat, Statistics, Unicode, Printf
export AbstractDataFrame,
       DataFrame,
       permutecols!
if VERSION >= v"1.1.0-DEV.792"
end
include("abstractdataframe/abstractdataframe.jl")
include("groupeddataframe/grouping.jl")
include("abstractdataframe/iteration.jl")
include("abstractdataframe/io.jl")
end # module DataFrames
