const RowTable{T} = Vector{T} where {T <: NamedTuple}
rows(x::RowTable) = x
struct NamedTupleIterator{S, T}
end
function Base.iterate(rows::NamedTupleIterator{Schema{names, T}}, st=()) where {names, T}
    if @generated
        return quote
        end
    else
    end
end
function Base.iterate(rows::NamedTupleIterator{Nothing, T}, st=()) where {T}
end
const ColumnTable = NamedTuple{names, T} where {names, T <: NTuple{N, AbstractArray{S, D} where S}} where {N, D}
istable(::Type{<:ColumnTable}) = true
columns(x::ColumnTable) = x
schema(x::T) where {T <: ColumnTable} = Schema(names(T), _types(T))
Base.@pure function _types(::Type{NT}) where {NT <: NamedTuple{names, T}} where {names, T <: NTuple{N, AbstractVector{S} where S}} where {N}
    return Tuple{Any[ _eltype(fieldtype(NT, i)) for i = 1:fieldcount(NT) ]...}
end
function columntable(sch::Schema{names, types}, cols) where {names, types}
    if @generated
        vals = Tuple(:(getarray(getproperty(cols, $(fieldtype(types, i)), $i, $(Meta.QuoteNode(names[i]))))) for i = 1:fieldcount(types))
        return :(NamedTuple{names}(($(vals...),)))
    else
        return NamedTuple{names}(Tuple(getarray(getproperty(cols, fieldtype(types, i), i, names[i])) for i = 1:fieldcount(types)))
    end
end
