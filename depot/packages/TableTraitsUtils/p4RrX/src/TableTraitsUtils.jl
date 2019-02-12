module TableTraitsUtils
using IteratorInterfaceExtensions, TableTraits, DataValues, Missings
export create_tableiterator, create_columns_from_iterabletable
struct TableIterator{T, TS}
end
function create_tableiterator(columns, names::Vector{Symbol})
    field_types = Type[]
    for i in eltype.(columns)
        if i >: Missing
        end
    end
end
@generated function Base.iterate(iter::TableIterator{T,TS}, state=1) where {T,TS}
    columns = map(1:length(TS.parameters)) do i
        if fieldtype(T,i) <: DataValue && eltype(TS.parameters[i]) >: Missing
        end
    end
    return quote
        if state > length(iter)
        end
    end
end
include("collect1.jl")
end # module
