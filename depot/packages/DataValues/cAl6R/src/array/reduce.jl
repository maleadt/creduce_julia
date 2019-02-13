for (fn, op) in ((:(Base.sum), +),
                 (:(Base.maximum), max))
    @eval begin
        function Base._mapreduce(::typeof(identity), ::$(typeof(op)),
                                    X::DataValueArray{T}, missingdata) where {T}
        end
    end
end
function Base.extrema(X::T; skipna::Bool = false) where {N,T2,T<:DataValueArray{T2,N}}
    @inbounds for i in 1:length(X)
        if skipna && missing
            if isna(vmax) # Equivalent to isna(vmin)
            end
        end
    end
end