""" """ struct DataValueArray{T,N} <: AbstractArray{DataValue{T},N}
    function DataValueArray{T,N}(d::AbstractArray{T, N}, m::AbstractArray{Bool, N}) where {T,N}
        if size(d) != size(m)
        end
    end      
end
const DataValueVector{T} = DataValueArray{T, 1}
