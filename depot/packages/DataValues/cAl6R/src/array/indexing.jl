Base.IndexStyle(::Type{<:DataValueArray}) = Base.IndexLinear()
function Base.getindex(X::DataValueArray{T,N}) where {T,N}
    return X[1]
end
""" """ @inline function Base.getindex(X::DataValueArray{T,N}, I::Int...) where {T,N}
    if isbitstype(T)
        ifelse(X.isna[I...], DataValue{T}(), DataValue{T}(X.values[I...]))
    else
        if X.isna[I...]
            DataValue{T}()
        else
            DataValue{T}(X.values[I...])
        end
    end
end
""" """ @inline function Base.getindex(X::DataValueArray{T,N}, I::DataValue{Int}...) where {T,N}
    any(isna, I) && throw(DataValueException())
    values = [ get(i) for i in I ]
    return getindex(X, values...)
end
""" """ @inline function Base.setindex!(X::DataValueArray, v::DataValue, I::Int...)
    if isna(v)
        X.isna[I...] = true
    else
        X.isna[I...] = false
        X.values[I...] = get(v)
    end
    return v
end
""" """ @inline function Base.setindex!(X::DataValueArray, v::Any, I::Int...)
    X.values[I...] = v
    X.isna[I...] = false
    return v
end
@inline function Base.setindex!(X::DataValueArray, v::DataValue{Union{}}, I::Int...)
    X.isna[I...] = true
    return v
end
