"""
    push!{T,V}(X::DataValueVector{T}, v::V)

Insert `v` at the end of `X`, which registers `v` as a present value.
"""
function Base.push!(X::DataValueVector{T}, v::V) where {T,V}
    push!(X.values, v)
    push!(X.isna, false)
    return X
end

"""
    push!{T,V}(X::DataValueVector{T}, v::DataValue{V})

Insert a value at the end of `X` from a `DataValue` value `v`. If `v` is null
then this method adds a null entry at the end of `X`. Returns `X`.
"""
function Base.push!(X::DataValueVector{T}, v::DataValue{V}) where {T,V}
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

"""
    pop!{T}(X::DataValueVector{T})

Remove the last entry from `X` and return it. If the value in that entry is
missing, then this method returns `DataValue{T}()`.
"""
function Base.pop!(X::DataValueVector{T}) where {T}
    val = pop!(X.values)
    isna = pop!(X.isna)
    return isna ? DataValue{T}() : DataValue(val)
end

"""
    pushfirst!(X::DataValueVector, v::DataValue)

Insert a value at the beginning of `X` from a `DataValue` value `v`. If `v` is
null then this method inserts a null entry at the beginning of `X`. Returns `X`.
"""
function Base.pushfirst!(X::DataValueVector, v::DataValue)
    if isna(v)
        ccall(:jl_array_grow_beg, Nothing, (Any, UInt), X.values, 1)
        pushfirst!(X.isna, true)
    else
        pushfirst!(X.values, v.value)
        pushfirst!(X.isna, false)
    end
    return X
end

"""
    pushfirst!(X::DataValueVector, v)

Insert a value `v` at the beginning of `X` and return `X`.
"""
function Base.pushfirst!(X::DataValueVector, v)
    pushfirst!(X.values, v)
    pushfirst!(X.isna, false)
    return X
end

"""
    popfirst!{T}(X::DataValueVector{T})

Remove the first entry from `X` and return it as a `DataValue` object.
"""
function Base.popfirst!(X::DataValueVector{T}) where {T}
    val = popfirst!(X.values)
    isna = popfirst!(X.isna)
    if isna
        return DataValue{T}()
    else
        return DataValue{T}(val)
    end
end

const _default_splice = []

"""
    splice!(X::DataValueVector, i::Integer, [ins])

Remove the item at index `i` and return the removed item. Subsequent items
are shifted down to fill the resulting gap. If specified, replacement values from
an ordered collection will be spliced in place of the removed item.
"""
function Base.splice!(X::DataValueVector, i::Integer, ins=_default_splice)
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

"""
    splice!{T<:Integer}(X::DataValueVector, rng::UnitRange{T}, [ins])

Remove items in the specified index range, and return a collection containing
the removed items. Subsequent items are shifted down to fill the resulting gap.
If specified, replacement values from an ordered collection will be spliced in
place of the removed items.

To insert `ins` before an index `n` without removing any items, use
`splice!(X, n:n-1, ins)`.
"""
function Base.splice!(X::DataValueVector, rng::UnitRange{T}, ins=_default_splice) where {T <: Integer}
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

"""
    deleteat!(X::DataValueVector, inds)

Delete the entry at `inds` from `X` and then return `X`. Note that `inds` may
be either a single scalar index or a collection of sorted, pairwise unique
indices. Subsequent items after deleted entries are shifted down to fill the
resulting gaps.
"""
function Base.deleteat!(X::DataValueVector, inds)
    deleteat!(X.values, inds)
    deleteat!(X.isna, inds)
    return X
end

"""
    append!(X::DataValueVector, items::AbstractVector)

Add the elements of `items` to the end of `X`.

Note that `append!(X, [1, 2, 3])` is equivalent to `push!(X, 1, 2, 3)`,
where the items to be added to `X` are passed individually to `push!` and as a
collection to `append!`.
"""
function Base.append!(X::DataValueVector, items::AbstractVector)
    old_length = length(X)
    nitems = length(items)
    resize!(X, old_length + nitems)
    copyto!(X, length(X)-nitems+1, items, 1, nitems)
    return X
end

"""
    prepend!(X::DataValueVector, items::AbstractVector)

Add the elements of `items` to the beginning of `X`.

Note that `prepend!(X, [1, 2, 3])` is equivalent to `pushfirst!(X, 1, 2, 3)`,
where the items to be added to `X` are passed individually to `pushfirst!` and as a
collection to `prepend!`.
"""
function Base.prepend!(X::DataValueVector, items::AbstractVector)
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

"""
    sizehint!(X::DataValueVector, newsz::Integer)

Suggest that collection `X` reserve capacity for at least `newsz` elements.
This can improve performance.
"""
function Base.sizehint!(X::DataValueVector, newsz::Integer)
    sizehint!(X.values, newsz)
    sizehint!(X.isna, newsz)
end

"""
    padna!(X::DataValueVector, front::Integer, back::Integer)

Insert `front` null entries at the beginning of `X` and add `back` null entries
at the end of `X`. Returns `X`.
"""
function padna!(X::DataValueVector{T}, front::Integer, back::Integer) where {T}
    prepend!(X, fill(DataValue{T}(), front))
    append!(X, fill(DataValue{T}(), back))
    return X
end

"""
    padna(X::DataValueVector, front::Integer, back::Integer)

return a copy of `X` with `front` null entries inserted at the beginning of
the copy and `back` null entries inserted at the end.
"""
function padna(X::DataValueVector, front::Integer, back::Integer)
    return padna!(copy(X), front, back)
end

"""
    reverse!(X::DataValueVector, [s], [n])

Modify `X` by reversing the first `n` elements starting at index `s`
(inclusive). If unspecified, `s` and `n` will default to `1` and `length(X)`,
respectively.
"""
function Base.reverse!(X::DataValueVector, s=1, n=length(X))
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

"""
    reverse(X::DataValueVector, [s], [n])

Return a copy of `X` with the first `n` elements starting at index `s`
(inclusive) reversed. If unspecified, `s` and `n` will default to `1` and
`length(X)`, respectively.
"""
function Base.reverse(X::DataValueVector, s=1, n=length(X))
    return reverse!(copy(X), s, n)
end

"""
    empty!(X::DataValueVector) -> DataValueVector

Remove all elements from a `DataValueVector`. Returns `DataValueVector{T}()`,
where `T` is the `eltype` of `X`.
"""
function Base.empty!(X::DataValueVector)
    empty!(X.values)
    empty!(X.isna)
    return X
end
