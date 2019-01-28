isna(X::DataValueArray, I::Int...) = X.isna[I...]
Base.values(X::DataValueArray, I::Int...) = X.values[I...]
"""
    size(X::DataValueArray, [d::Real])
Return a tuple containing the lengths of each dimension of `X`, or if `d` is
specific, the length of `X` along dimension `d`.
"""
Base.size(X::DataValueArray) = size(X.values)
function Base.similar(x::AbstractArray, ::Type{DataValue{T}}, dims::Dims) where {T}
    return DataValueArray{T}(dims)
end
function Base.similar(x::Array, ::Type{DataValue{T}}, dims::Dims) where {T}
    return DataValueArray{T}(dims)
end
function Base.similar(x::SubArray, ::Type{DataValue{T}}, dims::Dims) where {T}
    return DataValueArray{T}(dims)
end
"""
    copy(X::DataValueArray)
Return a shallow copy of `X`; the outer structure of `X` will be copied, but
all elements will be identical to those of `X`.
"""
function Base.copy(X::DataValueArray{T}) where {T}
    return Base.copyto!(similar(X, DataValue{T}), X)
end
"""
    copyto!(dest::DataValueArray, src::DataValueArray)
Copy the initialized values of a source DataValueArray into the respective
indices of the destination DataValueArray. If an entry in `src` is null, then
this method nullifies the respective entry in `dest`.
"""
function Base.copyto!(dest::DataValueArray, src::DataValueArray)
    if isbitstype(eltype(dest)) && isbitstype(eltype(src))
        copyto!(dest.values, src.values)
    else
        dest_values = dest.values
        src_values = src.values
        length(dest_values) >= length(src_values) || throw(BoundsError())
        for i in 1:length(src_values)
            @inbounds !(src.isna[i]) && (dest.values[i] = src.values[i])
        end
    end
    copyto!(dest.isna, src.isna)
    return dest
end
"""
    fill!(X::DataValueArray, x::DataValue)
Fill `X` with the value `x`. If `x` is empty, then `fill!(X, x)` nullifies each
entry of `X`. Otherwise, `fill!(X, x)` fills `X.values` with the value of `x`
and designates each entry of `X` as present.
"""
function Base.fill!(X::DataValueArray, x::DataValue)
    if isna(x)
        fill!(X.isna, true)
    else
        fill!(X.values, get(x))
        fill!(X.isna, false)
    end
    return X
end
"""
    fill!(X::DataValueArray, x::DataValue)
Fill `X` with the value `x` and designate each entry as present. If `x` is an
object reference, all elements will refer to the same object. Note that
`fill!(X, Foo())` will return `X` filled with the result of evaluating `Foo()`
once.
"""
function Base.fill!(X::DataValueArray, x::Any)
    fill!(X.values, x)
    fill!(X.isna, false)
    return X
end
"""
    Base.deepcopy(X::DataValueArray)
Return a `DataValueArray` object whose internal `values` and `isna` fields are
deep copies of `X.values` and `X.isna` respectively.
"""
function Base.deepcopy(X::DataValueArray)
    return DataValueArray(deepcopy(X.values), deepcopy(X.isna))
end
"""
    resize!(X::DataValueVector, n::Int)
Resize a one-dimensional `DataValueArray` `X` to contain precisely `n` elements.
If `n` is greater than the current length of `X`, then each new entry will be
designated as null.
"""
function Base.resize!(X::DataValueArray{T,1}, n::Int) where {T}
    resize!(X.values, n)
    oldn = length(X.isna)
    resize!(X.isna, n)
    X.isna[oldn+1:n] .= true
    return X
end
function Base.reshape(X::DataValueArray, dims::Dims)
    return DataValueArray(reshape(X.values, dims), reshape(X.isna, dims))
end
"""
    ndims(X::DataValueArray)
Returns the number of dimensions of `X`.
"""
Base.ndims(X::DataValueArray) = ndims(X.values)
"""
    length(X::DataValueArray)
Returns the maximum index `i` for which `getindex(X, i)` is valid.
"""
Base.length(X::DataValueArray) = length(X.values)
"""
    lastindex(X::DataValueArray)
Returns the last entry of `X`.
"""
Base.lastindex(X::DataValueArray) = lastindex(X.values)
"""
    dropna(X::AbstractVector)
Return a vector containing only the non-missing entries of `X`,
unwrapping `DataValue` entries. A copy is always returned, even when
`X` does not contain any missing values.
"""
function dropna(X::AbstractVector{T}) where {T}
    if !(DataValue <: T) && !(T <: DataValue)
        return copy(X)
    else
        Y = filter(x->!isna(x), X)
        res = similar(Y, eltype(T))
        for i in eachindex(Y, res)
            @inbounds res[i] = isa(Y[i], DataValue) ? Y[i].value : Y[i]
        end
        return res
    end
end
dropna(X::DataValueVector) = X.values[(!).(X.isna)]
"""
    dropna!(X::AbstractVector)
Remove missing entries of `X` in-place and return a `Vector` view of the
unwrapped `DataValue` entries. If no missing values are present, this is a no-op
and `X` is returned.
"""
function dropna!(X::AbstractVector{T}) where {T}                 # -> AbstractVector
    if !(DataValue <: T) && !(T <: DataValue)
        return X
    else
        deleteat!(X, findall(isna, X))
        res = similar(X, eltype(T))
        for i in eachindex(X, res)
            @inbounds res[i] = isa(X[i], DataValue) ? X[i].value : X[i]
        end
        return res
    end
end
"""
    dropna!(X::DataValueVector)
Remove missing entries of `X` in-place and return a `Vector` view of the
unwrapped `DataValue` entries.
"""
dropna!(X::DataValueVector) = deleteat!(X, (LinearIndices(X.isna))[findall(X.isna)]).values # -> Vector
"""
    convert(T, X::DataValueArray)
Convert `X` to an `AbstractArray` of type `T`. Note that if `X` contains any
null entries then calling `convert` without supplying a replacement value for
null entries will result in an error.
Currently supported return type arguments include: `Array`, `Array{T}`,
`Vector`, `Matrix`.
    convert(T, X::DataValueArray, replacement)
Convert `X` to an `AbstractArray` of type `T` and replace all null entries of
`X` with `replacement` in the result.
"""
function Base.convert(::Type{Array{S, N}}, X::DataValueArray{T, N}) where {S,T,N}
    if any(isna, X)
        throw(DataValueException())
    else
        return convert(Array{S, N}, X.values)
    end
end
function Base.convert(::Type{Array{S}}, X::DataValueArray{T, N}) where {S,T,N} # -> Array{S, N}
    return convert(Array{S, N}, X)
end
function Base.convert(::Type{Vector}, X::DataValueVector{T}) where {T} # -> Vector{T}
    return convert(Array{T, 1}, X)
end
function Base.convert(::Type{Matrix}, X::DataValueMatrix{T}) where {T} # -> Matrix{T}
    return convert(Array{T, 2}, X)
end
function Base.convert(::Type{Array},
                            X::DataValueArray{T, N}) where {T,N} # -> Array{T, N}
    return convert(Array{T, N}, X)
end
function Base.convert(::Type{Array{S, N}},
                               X::DataValueArray{T, N},
                               replacement::Any) where {S,T,N} # -> Array{S, N}
    replacementS = convert(S, replacement)
    res = Array{S}(undef, size(X))
    for i in 1:length(X)
        if X.isna[i]
            res[i] = replacementS
        else
            res[i] = X.values[i]
        end
    end
    return res
end
function Base.convert(::Type{Vector},
                         X::DataValueVector{T},
                         replacement::Any) where {T} # -> Vector{T}
    return convert(Array{T, 1}, X, replacement)
end
function Base.convert(::Type{Matrix},
                         X::DataValueMatrix{T},
                         replacement::Any) where {T} # -> Matrix{T}
    return convert(Array{T, 2}, X, replacement)
end
function Base.convert(::Type{Array},
                            X::DataValueArray{T, N},
                            replacement::Any) where {T,N} # -> Array{T, N}
    return convert(Array{T, N}, X, replacement)
end
Base.any(::typeof(isna), X::DataValueArray) = Base.any(X.isna)
Base.all(::typeof(isna), X::DataValueArray) = Base.all(X.isna)
