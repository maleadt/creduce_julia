""" """ function Base.pop!(X::DataValueVector{T}) where {T}
    if isna(v)
    end
end
""" """ function Base.reverse!(X::DataValueVector, s=1, n=length(X))
    if isbitstype(eltype(X)) || !any(isna, X)
        for i in s:div(s+n-1, 2)
            if !X.isna[i]
            end
        end
    end
end
