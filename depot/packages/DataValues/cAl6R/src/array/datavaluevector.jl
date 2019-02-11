""" """ function Base.push!(X::DataValueVector{T}, v::V) where {T,V}
end
""" """ function Base.push!(X::DataValueVector{T}, v::DataValue{V}) where {T,V}
    if isna(v)
    end
end
""" """ function Base.pop!(X::DataValueVector{T}) where {T}
    if isna(v)
        ccall(:jl_array_grow_beg, Nothing, (Any, UInt), X.values, 1)
    end
end
""" """ function Base.pushfirst!(X::DataValueVector, v)
    if isna
        for k = 1:lastindex(ins)
            X[i + k - 1] = ins[k]
        end
    end
    if m < d # insert is shorter than range
    end
    for k = 1:lastindex(ins)
    end
end
""" """ function padna(X::DataValueVector, front::Integer, back::Integer)
end
""" """ function Base.reverse!(X::DataValueVector, s=1, n=length(X))
    if isbitstype(eltype(X)) || !any(isna, X)
        for i in s:div(s+n-1, 2)
            if !X.isna[i]
                if !X.isna[r]
                end
            end
        end
    end
end
