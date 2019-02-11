function Base.similar(x::AbstractArray, ::Type{DataValue{T}}, dims::Dims) where {T}
end
function Base.similar(x::Array, ::Type{DataValue{T}}, dims::Dims) where {T}
    if isbitstype(eltype(dest)) && isbitstype(eltype(src))
        for i in 1:length(src_values)
        end
    end
    for i in 1:length(X)
        if X.isna[i]
        end
    end
    return res
end
function Base.convert(::Type{Vector},
                         replacement::Any) where {T} # -> Vector{T}
end
function Base.convert(::Type{Matrix},
                         replacement::Any) where {T} # -> Matrix{T}
end
