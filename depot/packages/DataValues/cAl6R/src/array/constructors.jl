
# ----- Outer Constructors -------------------------------------------------- #

# The following provides an outer constructor whose argument signature matches
# that of the inner constructor provided in typedefs.jl: constructs a DataValueArray
# from an AbstractArray of values and an AbstractArray{Bool} mask.
function DataValueArray(d::AbstractArray{T,N}, m::AbstractArray{Bool,N}) where {T,N}
    return DataValueArray{T,N}(d, m)
end

function DataValueArray{T}(d::NTuple{N,Int}) where {T,N}
    return DataValueArray{T,N}(Array{T,N}(undef, d), fill(true, d))    
end

function DataValueArray{T,N}(d::NTuple{N,Int}) where {T,N}
    return DataValueArray{T,N}(Array{T,N}(undef, d), fill(true, d))    
end

function DataValueArray{T}(d::Vararg{Int,N}) where {T,N}
    return DataValueArray{T,N}(Array{T,N}(undef, d), fill(true, d))    
end

function DataValueArray{T,N}(d::Vararg{Int,N}) where {T,N}
    return DataValueArray{T,N}(Array{T,N}(undef, d), fill(true, d))    
end

function DataValueArray{T, N}(::UndefInitializer, args...) where {T, N}
    return DataValueArray(Array{T, N}(undef, args...), fill(true, args...))
end

function DataValueArray{T}(::UndefInitializer, args...) where {T}
    return DataValueArray(Array{T}(undef, args...), fill(true, args...))
end

function DataValueArray(data::AbstractArray{T,N}) where {T<:DataValue,N}
    S = eltype(eltype(data))
    new_array = DataValueArray{S,N}(Array{S}(undef, size(data)), Array{Bool}(undef, size(data)))
    for i in eachindex(data)
        new_array[i] = data[i]
    end
    return new_array
end

function DataValueArray{S}(data::AbstractArray{T,N}) where {S,T<:DataValue,N}
    new_array = DataValueArray{S,N}(Array{S}(undef, size(data)), Array{Bool}(undef, size(data)))
    for i in eachindex(data)
        new_array[i] = data[i]
    end
    return new_array
end

# The following method allows for the construction of zero-element
# DataValueArrays by calling the parametrized type on zero arguments.
function DataValueArray{T,N}() where {T,N}
    return DataValueArray{T}(ntuple(i->0, N))
end

function DataValueArray{T,N}(data::AbstractArray{S,N}) where {S,T,N}
    convert(DataValueArray{T,N}, data)
end

function DataValueArray{T}(data::AbstractArray{S,N}) where {S,T,N}
    convert(DataValueArray{T,N}, data)
end

function DataValueArray(data::AbstractArray{T,N}) where {T,N}
    return convert(DataValueArray{T,N}, data)
end

# ----- Conversion to DataValueArrays ---------------------------------------- #
# Also provides constructors from arrays via the fallback mechanism.

#----- Conversion from arrays (of non-DataValues) -----------------------------#
function Base.convert(::Type{DataValueArray{T,N}}, A::AbstractArray{S,N}) where {S,T,N}
    return DataValueArray{T,N}(convert(Array{T,N}, A), fill(false, size(A)))
end

function Base.convert(::Type{DataValueArray{T}}, A::AbstractArray{S,N}) where {S,T,N}
    return convert(DataValueArray{T,N}, A)
end

function Base.convert(::Type{DataValueArray}, A::AbstractArray{T,N}) where {T,N}
    return convert(DataValueArray{T,N}, A)
end

#----- Conversion from arrays of DataValues -----------------------------------#
function Base.convert(::Type{DataValueArray{T,N}}, A::AbstractArray{S,N}) where {S<:DataValue,T,N}
    new_array = DataValueArray{T,N}(Array{T}(undef, size(A)), Array{Bool}(undef, size(A)))
    for i in eachindex(A)
        new_array[i] = A[i]
    end
    return new_array
end

#----- Conversion from DataValueArrays of a different type --------------------#
function Base.convert(::Type{DataValueArray}, X::DataValueArray{T,N}) where {T,N}
    return X
end

function Base.convert(::Type{DataValueArray{T}}, A::AbstractArray{DataValue{S},N}) where {S,T,N}
    return convert(DataValueArray{T, N}, A)
end

function Base.convert(::Type{DataValueArray}, A::AbstractArray{DataValue{T},N}) where {T,N}
    return convert(DataValueArray{T,N}, A)
end

function Base.convert(::Type{DataValueArray}, A::AbstractArray{DataValue, N}) where {N}
    return convert(DataValueArray{Any,N}, A)
end

function Base.convert(::Type{DataValueArray{T,N}}, A::DataValueArray{S,N}) where {S,T,N}
    return DataValueArray(convert(Array{T,N}, A.values), A.isna)
end
