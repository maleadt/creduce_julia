__precompile__()
module DataValues
using Dates
export DataValue, DataValueException, NA
export DataValueArray, DataValueVector, DataValueMatrix
export isna, hasvalue, dropna, dropna!, padna!, padna
include("scalar/core.jl")
include("scalar/operations.jl")
include("array/typedefs.jl")
include("array/constructors.jl")
include("array/indexing.jl")
include("array/datavaluevector.jl")
include("array/primitives.jl")
include("array/reduce.jl")
include("array/promotion.jl")
end
