""" """ function Base.mapreduce(f, op::Function, X::T; skipna::Bool = false) where {N,S<:DataValue,T<:AbstractArray{S,N}}
    if skipna
        return DataValue(mapreduce(f, op, dropna(X)))
    else
        return Base._mapreduce(f, op, IndexStyle(X), X)
    end
end
""" """ function Base.reduce(op, X::T; skipna::Bool = false) where {N,S<:DataValue,T<:AbstractArray{S,N}}
    return mapreduce(identity, op, X; skipna = skipna)
end
for (fn, op) in ((:(Base.sum), +),
                 (:(Base.prod), *),
                 (:(Base.minimum), min),
                 (:(Base.maximum), max))
    @eval begin
        function $fn(f::Union{Function,Type},X::T; skipna::Bool=false) where {N,S<:DataValue,T<:AbstractArray{S,N}}
            return mapreduce(f, $op, X; skipna = skipna)
        end
        function $fn(X::T; skipna::Bool=false) where {N,S<:DataValue,T<:AbstractArray{S,N}}
            return mapreduce(identity, $op, X; skipna = skipna)
        end
    end
end
for op in (Base.min, Base.max)
    @eval begin
        function Base._mapreduce(::typeof(identity), ::$(typeof(op)),
                                    X::DataValueArray{T}, missingdata) where {T}
            missingdata && return DataValue{T}()
            DataValue(Base._mapreduce(identity, $op, X.values))
        end
    end
end
function Base.extrema(X::T; skipna::Bool = false) where {N,T2,T<:DataValueArray{T2,N}}
    length(X) > 0 || throw(ArgumentError("collection must be non-empty"))
    vmin = DataValue{T2}()
    vmax = DataValue{T2}()
    @inbounds for i in 1:length(X)
        x = X.values[i]
        missing = X.isna[i]
        if skipna && missing
            continue
        elseif missing
            return (DataValue{T2}(), DataValue{T2}())
        elseif isna(vmax) # Equivalent to isna(vmin)
            vmax = vmin = DataValue(x)
        elseif x > vmax.value
            vmax = DataValue(x)
        elseif x < vmin.value
            vmin = DataValue(x)
        end
    end
    return (vmin, vmax)
end
function Base.extrema(X::T; skipna::Bool = false) where {N,T2,S<:DataValue{T2},T<:AbstractArray{S,N}}
    length(X) > 0 || throw(ArgumentError("collection must be non-empty"))
    vmin = DataValue{T2}()
    vmax = DataValue{T2}()
    @inbounds for i in 1:length(X)
        missing = isna(X[i])
        if skipna && missing
            continue
        elseif missing
            return (DataValue{T2}(), DataValue{T2}())
        else
            x = get(X[i])
            if isna(vmax) # Equivalent to isna(vmin)
                vmax = vmin = DataValue(x)
            elseif x > vmax.value
                vmax = DataValue(x)
            elseif x < vmin.value
                vmin = DataValue(x)
            end
        end
    end
    return (vmin, vmax)
end