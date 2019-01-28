module TableTraitsUtils

using IteratorInterfaceExtensions, TableTraits, DataValues, Missings

export create_tableiterator, create_columns_from_iterabletable

# T is the type of the elements produced
# TS is a tuple type that stores the columns of the table
struct TableIterator{T, TS}
    columns::TS
end

function create_tableiterator(columns, names::Vector{Symbol})
    field_types = Type[]
    for i in eltype.(columns)
        if i >: Missing
            push!(field_types, DataValue{Missings.T(i)})
        else
            push!(field_types, i)
        end
    end
    return TableIterator{NamedTuple{(names...,), Tuple{field_types...}}, Tuple{typeof.(columns)...}}((columns...,))
end

function Base.length(iter::TableIterator{T,TS}) where {T,TS}
    return length(iter.columns)==0 ? 0 : length(iter.columns[1])
end

Base.eltype(::Type{TableIterator{T,TS}}) where {T,TS} = T

@generated function Base.iterate(iter::TableIterator{T,TS}, state=1) where {T,TS}
    columns = map(1:length(TS.parameters)) do i
        if fieldtype(T,i) <: DataValue && eltype(TS.parameters[i]) >: Missing
            return :($(fieldtype(T,i))(iter.columns[$i][state]))
        else
            return :(iter.columns[$i][state])
        end
    end
    return quote
        if state > length(iter)
            return nothing
        else            
            return $(T)(($(columns...),)), state+1
        end
    end
end

include("collect1.jl")

end # module
