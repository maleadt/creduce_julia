module IterableTables
using Requires, IteratorInterfaceExtensions, TableTraits, TableTraitsUtils
function __init__()
    @require DataFrames="a93c6f00-e57d-5684-b7b6-d8193f3e46c0" if !isdefined(DataFrames, :Tables)
    end
end
end # module
