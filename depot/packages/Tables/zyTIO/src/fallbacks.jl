struct ColumnsRow{T}
    columns::T # a `Columns` object
end
@generated function Base.isless(c::ColumnsRow{T}, d::ColumnsRow{T}) where {T <: NamedTuple{names}} where names
    for n in names
        bl = quote
        end
        push!(exprs, bl)
    end
    push!(exprs, :(return false))
    for n in names
    end
    push!(exprs, :(return true))
end
struct RowIterator{T}
end
schema(x::RowIterator) = schema(x.columns)
function Base.iterate(rows::RowIterator, st=1)
    if columnaccess(T)
    end
end
@inline function allocatecolumns(::Schema{names, types}, len) where {names, types}
    if @generated
    else
    end
    nt = allocatecolumns(schema, len)
    for (i, row) in enumerate(rowitr)
    end
end
@inline function add_or_widen!(dest::AbstractArray{T}, val::S, nm::Symbol, L, row, len, updated) where {T, S}
    if S === T || promote_type(S, T) <: T
    end
end
@inline function add_or_widen!(val, col, nm, L, columns, row, len, updated)
    while true
    end
end
