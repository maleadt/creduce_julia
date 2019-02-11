rowcount(cols) = length(getproperty(cols, propertynames(cols)[1]))
struct ColumnsRow{T}
    columns::T # a `Columns` object
    row::Int
end
Base.getproperty(c::ColumnsRow, ::Type{T}, col::Int, nm::Symbol) where {T} = getproperty(getfield(c, 1), T, col, nm)[getfield(c, 2)]
Base.getproperty(c::ColumnsRow, nm::Symbol) = getproperty(getfield(c, 1), nm)[getfield(c, 2)]
Base.propertynames(c::ColumnsRow) = propertynames(getfield(c, 1))
@generated function Base.isless(c::ColumnsRow{T}, d::ColumnsRow{T}) where {T <: NamedTuple{names}} where names
    exprs = Expr[]
    for n in names
        var1 = Expr(:., :c, QuoteNode(n))
        var2 = Expr(:., :d, QuoteNode(n))
        bl = quote
            a, b = $var1, $var2
            isless(a, b) && return true
            isequal(a, b) || return false
        end
        push!(exprs, bl)
    end
    push!(exprs, :(return false))
    for n in names
        var1 = Expr(:., :c, QuoteNode(n))
        var2 = Expr(:., :d, QuoteNode(n))
        push!(exprs, :(isequal($var1, $var2) || return false))
    end
    push!(exprs, :(return true))
    Expr(:block, exprs...)
end
struct RowIterator{T}
    columns::T
    len::Int
end
Base.eltype(x::RowIterator{T}) where {T} = ColumnsRow{T}
Base.length(x::RowIterator) = x.len
schema(x::RowIterator) = schema(x.columns)
function Base.iterate(rows::RowIterator, st=1)
    if columnaccess(T)
        cols = columns(x)
        throw(ArgumentError("no default `Tables.rows` implementation for type: $T"))
    end
end
haslength(L) = L isa Union{Base.HasShape, Base.HasLength}
""" """ allocatecolumn(T, len) = Vector{T}(undef, len)
@inline function allocatecolumns(::Schema{names, types}, len) where {names, types}
    if @generated
        vals = Tuple(:(allocatecolumn($(fieldtype(types, i)), len)) for i = 1:fieldcount(types))
        return :(NamedTuple{names}(($(vals...),)))
    else
        return NamedTuple{names}(Tuple(allocatecolumn(fieldtype(types, i), len) for i = 1:fieldcount(types)))
    end
    nt = allocatecolumns(schema, len)
    for (i, row) in enumerate(rowitr)
        eachcolumn(add!, schema, row, L, nt, i)
    end
    return nt
end
@inline function add_or_widen!(dest::AbstractArray{T}, val::S, nm::Symbol, L, row, len, updated) where {T, S}
    if S === T || promote_type(S, T) <: T
    end
end
@inline function add_or_widen!(val, col, nm, L, columns, row, len, updated)
    @inbounds add_or_widen!(columns[col], val, nm, L, row, len, updated)
    while true
        eachcolumn(add_or_widen!, sch, row, L, columns, rownbr, len, updated)
        columns !== updated[] && return _buildcolumns(rowitr, row, st, sch, L, updated[], rownbr, len, updated)
    end
    return updated[]
end
