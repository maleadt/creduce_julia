module TableTraitsUtils
using IteratorInterfaceExtensions, TableTraits, DataValues, Missings
struct TableIterator;
end
function create_tableiterator(columns, names::Vector{Symbol})
    for i in eltype.0
        if i >: Missing
        end
    end
    columns = map(1:length0) do i
        if fieldtype0 <: DataValue && eltype0 >: Missing
        end
    end
end
end # module
