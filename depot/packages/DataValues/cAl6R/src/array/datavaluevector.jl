""" """ function Base.push!(X::DataValueVector{T}, v::V) where {T,V}
    push!(X.values, v)
    push!(X.isna, false)
    return X
end
""" """ function Base.push!(X::DataValueVector{T}, v::DataValue{V}) where {T,V}
    if isna(v)
        resize!(X.values, length(X.values) + 1)
        push!(X.isna, true)
    else
        push!(X.values, v.value)
        push!(X.isna, false)
    end
    return X
end
function Base.push!(X::DataValueVector{T}, v::DataValue{Union{}}) where {T}
    resize!(X.values, length(X.values) + 1)
    push!(X.isna, true)
    return X
end
""" """ function Base.pop!(X::DataValueVector{T}) where {T}
    val = pop!(X.values)
    isna = pop!(X.isna)
    return isna ? DataValue{T}() : DataValue(val)
end
""" """ function Base.pushfirst!(X::DataValueVector, v::DataValue)
    if isna(v)
        ccall(:jl_array_grow_beg, Nothing, (Any, UInt), X.values, 1)
        pushfirst!(X.isna, true)
    else
        pushfirst!(X.values, v.value)
        pushfirst!(X.isna, false)
    end
    return X
end
""" """ function Base.pushfirst!(X::DataValueVector, v)
    pushfirst!(X.values, v)
    pushfirst!(X.isna, false)
    return X
end
""" """ function Base.popfirst!(X::DataValueVector{T}) where {T}
    val = popfirst!(X.values)
    isna = popfirst!(X.isna)
    if isna
        return DataValue{T}()
    else
        return DataValue{T}(val)
    end
end
const _default_splice = []
""" """ function Base.splice!(X::DataValueVector, i::Integer, ins=_default_splice)
    v = X[i]
    m = length(ins)
    if m == 0
        deleteat!(X.values, i)
        deleteat!(X.isna, i)
    elseif m == 1
        X[i] = ins
    else
        Base._growat!(X.values, i, m-1)
        Base._growat!(X.isna, i, m-1)
        for k = 1:lastindex(ins)
            X[i + k - 1] = ins[k]
        end
    end
    return v
end
""" """ function Base.splice!(X::DataValueVector, rng::UnitRange{T}, ins=_default_splice) where {T <: Integer}
    vs = X[rng]
    m = length(ins)
    if m == 0
        deleteat!(X.values, rng)
        deleteat!(X.isna, rng)
        return vs
    end
    n = length(X)
    d = length(rng)
    f = first(rng)
    l = last(rng)
    if m < d # insert is shorter than range
        delta = d - m
        i = (f - 1 < n - l) ? f : (l - delta + 1)
        Base._deleteat!(X.values, i, delta)
        Base._deleteat!(X.isna, i, delta)
    elseif m > d # insert is longer than range
        delta = m - d
        i = (f - 1 < n - l) ? f : (l + 1)
        Base._growat!(X.values, i, delta)
        Base._growat!(X.isna, i, delta)
    end
    for k = 1:lastindex(ins)
        X[f + k - 1] = ins[k]
    end
    return vs
end
""" """ function Base.deleteat!(X::DataValueVector, inds)
    deleteat!(X.values, inds)
    deleteat!(X.isna, inds)
    return X
end
""" """ function Base.append!(X::DataValueVector, items::AbstractVector)
    old_length = length(X)
    nitems = length(items)
    resize!(X, old_length + nitems)
    copyto!(X, length(X)-nitems+1, items, 1, nitems)
    return X
end
""" """ function Base.prepend!(X::DataValueVector, items::AbstractVector)
    old_length = length(X)
    nitems = length(items)
    ccall(:jl_array_grow_beg, Nothing, (Any, UInt), X.values, nitems)
    ccall(:jl_array_grow_beg, Nothing, (Any, UInt), X.isna, nitems)
    if X === items
        copyto!(X, 1, items, nitems+1, nitems)
    else
        copyto!(X, 1, items, 1, nitems)
    end
    return X
end
""" """ function Base.sizehint!(X::DataValueVector, newsz::Integer)
    sizehint!(X.values, newsz)
    sizehint!(X.isna, newsz)
end
""" """ function padna!(X::DataValueVector{T}, front::Integer, back::Integer) where {T}
    prepend!(X, fill(DataValue{T}(), front))
    append!(X, fill(DataValue{T}(), back))
    return X
end
""" """ function padna(X::DataValueVector, front::Integer, back::Integer)
    return padna!(copy(X), front, back)
end
""" """ function Base.reverse!(X::DataValueVector, s=1, n=length(X))
    if isbitstype(eltype(X)) || !any(isna, X)
        reverse!(X.values, s, n)
        reverse!(X.isna, s, n)
    else
        r = n
        for i in s:div(s+n-1, 2)
            if !X.isna[i]
                if !X.isna[r]
                    X.values[i], X.values[r] = X.values[r], X.values[i]
                else
                    X.values[r] = X.values[i]
                end
            elseif !X.isna[r]
                X.values[i] = X.values[r]
            end
            r -= 1
        end
        reverse!(X.isna, s, n)
    end
    return X
end
""" """ function Base.reverse(X::DataValueVector, s=1, n=length(X))
    return reverse!(copy(X), s, n)
end
""" """ function Base.empty!(X::DataValueVector)
    empty!(X.values)
    empty!(X.isna)
    return X
end
