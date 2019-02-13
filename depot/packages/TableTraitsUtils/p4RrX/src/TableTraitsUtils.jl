module TableTraitsUtils
using IteratorInterfaceExtensions, TableTraits, DataValues, Missings
struct TableIterator{T, TS}
end
function create_tableiterator(columns, names::Vector{Symbol})
    for i in eltype.(columns)
        if i >: Missing
        end
    end
    columns = map(1:length(TS.parameters)) do i
        if fieldtype(T,i) <: DataValue && eltype(TS.parameters[i]) >: Missing
        end
    end
end
end # module
