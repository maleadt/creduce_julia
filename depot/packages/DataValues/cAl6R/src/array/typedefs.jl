"""
`DataValueArray{T, N}` is an efficient alternative to `Array{DataValue{T}, N}`.
"""
struct DataValueArray{T,N} <: AbstractArray{DataValue{T},N}
    values::Array{T,N}
    isna::Array{Bool,N}

    function DataValueArray{T,N}(d::AbstractArray{T, N}, m::AbstractArray{Bool, N}) where {T,N}
        if size(d) != size(m)
            msg = "values and missingness arrays must be the same size"
            throw(ArgumentError(msg))
        end
        new(d, m)
    end      
end

const DataValueVector{T} = DataValueArray{T, 1}
const DataValueMatrix{T} = DataValueArray{T, 2}
