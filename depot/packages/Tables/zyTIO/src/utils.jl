names(::Type{NamedTuple{nms, T}}) where {nms, T} = nms
types(::Type{NamedTuple{nms, T}}) where {nms, T} = T
"helper function to calculate a run-length encoding of a tuple type"
Base.@pure function runlength(::Type{T}) where {T <: Tuple}
    rle = Tuple{Type, Int}[]
    fieldcount(T) == 0 && return rle
    curT = fieldtype(T, 1)
    prevT = curT
    len = 1
    for i = 2:fieldcount(T)
        @inbounds curT = fieldtype(T, i)
        if curT === prevT
            len += 1
        else
            push!(rle, (prevT, len))
            prevT = curT
            len = 1
        end
    end
    push!(rle, (curT, len))
    return rle
end
Base.getproperty(x, ::Type{T}, i::Int, nm::Symbol) where {T} = getproperty(x, nm)
""" """ function eachcolumn end
@inline function eachcolumn(f::Base.Callable, sch::Schema{names, types}, row, args...) where {names, types}
    if @generated
        rle = runlength(types)
        if length(rle) < 100
            block = Expr(:block, Expr(:meta, :inline))
            i = 1
            for (T, len) in rle
                push!(block.args, quote
                    for j = 0:$(len-1)
                        @inbounds f(getproperty(row, $T, $i + j, names[$i + j]), $i + j, names[$i + j], args...)
                    end
                end)
                i += len
            end
            b = block
        else
            b = quote
                $(Expr(:meta, :inline))
                for (i, nm) in enumerate(names)
                    f(getproperty(row, fieldtype(types, i), i, nm), i, nm, args...)
                end
                return
            end
        end
        return b
    else
        for (i, nm) in enumerate(names)
            f(getproperty(row, fieldtype(types, i), i, nm), i, nm, args...)
        end
        return
    end
end
@inline function eachcolumn(f::Base.Callable, sch::Schema{names, nothing}, row, args...) where {names}
    if @generated
        if length(names) < 100
            b = Expr(:block, Any[:(f(getproperty(row, $(Meta.QuoteNode(names[i]))), $i, $(Meta.QuoteNode(names[i])), args...)) for i = 1:length(names)]...)
        else
            b = quote
                for (i, nm) in enumerate(names)
                    f(getproperty(row, nm), i, nm, args...)
                end
                return
            end
        end
        return b
    else
        for (i, nm) in enumerate(names)
            f(getproperty(row, nm), i, nm, args...)
        end
        return
    end
end
struct EachColumn{T}
    source::T
end
Base.length(e::EachColumn) = length(propertynames(e.source))
Base.IteratorEltype(::Type{<:EachColumn}) = Base.EltypeUnknown()
function Base.iterate(e::EachColumn, (idx, props)=(1, propertynames(e.source)))
    idx > length(props) && return nothing
    return getproperty(e.source, props[idx]), (idx + 1, props)
end
eachcolumn(c) = EachColumn(c)
"given names and a Symbol `name`, compute the index (1-based) of the name in names"
Base.@pure function columnindex(names::Tuple{Vararg{Symbol}}, name::Symbol)
    i = 1
    for nm in names
        nm === name && return i
        i += 1
    end
    return 0
end
"given tuple type and a Symbol `name`, compute the type of the name in the tuples types"
Base.@pure function columntype(names::Tuple{Vararg{Symbol}}, ::Type{T}, name::Symbol) where {T <: Tuple}
    i = 1
    for nm in names
        nm === name && return fieldtype(T, i)
        i += 1
    end
    return Union{}
end
