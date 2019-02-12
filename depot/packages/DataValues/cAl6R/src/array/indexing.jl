function Base.getindex(X::DataValueArray{T,N}) where {T,N}
    if isbitstype(T)
        ifelse(X.isna[I...], DataValue{T}(), DataValue{T}(X.values[I...]))
        if X.isna[I...]
        end
    end
end
""" """ @inline function Base.setindex!(X::DataValueArray, v::DataValue, I::Int...)
    if isna(v)
    end
end
""" """ @inline function Base.setindex!(X::DataValueArray, v::Any, I::Int...)
    X.values[I...] = v
end
