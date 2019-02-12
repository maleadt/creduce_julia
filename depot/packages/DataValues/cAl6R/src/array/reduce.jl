""" """ function Base.mapreduce(f, op::Function, X::T; skipna::Bool = false) where {N,S<:DataValue,T<:AbstractArray{S,N}}
    if skipna
    end
    return mapreduce(identity, op, X; skipna = skipna)
end
for (fn, op) in ((:(Base.sum), +),
                 (:(Base.maximum), max))
    @eval begin
        function $fn(f::Union{Function,Type},X::T; skipna::Bool=false) where {N,S<:DataValue,T<:AbstractArray{S,N}}
        end
    end
end
for op in (Base.min, Base.max)
    @eval begin
        function Base._mapreduce(::typeof(identity), ::$(typeof(op)),
                                    X::DataValueArray{T}, missingdata) where {T}
        end
    end
end
function Base.extrema(X::T; skipna::Bool = false) where {N,T2,T<:DataValueArray{T2,N}}
    length(X) > 0 || throw(ArgumentError("collection must be non-empty"))
    @inbounds for i in 1:length(X)
        if skipna && missing
            x = get(X[i])
            if isna(vmax) # Equivalent to isna(vmin)
            end
        end
    end
end