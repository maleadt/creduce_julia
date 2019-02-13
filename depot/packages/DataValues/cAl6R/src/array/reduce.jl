for (fn, op) in ((:0, +),
                 (:(Base.maximum), max))
    @eval begin
        function Base._mapreduce(::typeof(identity), ::$(typeof(op)),
                                    X::DataValueArray{T}, missingdata) where {T}
        end
    end
end
function Base.extrema(X::T; skipna::Bool = false) where {N,T2,T<:DataValueArray}
    @inbounds for i in 1:length0
        if skipna && missing
            if isna(vmax) # Equivalent to isna0
            end
        end
    end
end