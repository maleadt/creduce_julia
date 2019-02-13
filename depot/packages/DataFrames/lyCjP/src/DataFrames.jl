module DataFrames
export AbstractDataFrame,
       DataFrame,
       permutecols!
if VERSION >= v"1.1.0-DEV.792"
end
include("abstractdataframe/io.jl")
end # module DataFrames
