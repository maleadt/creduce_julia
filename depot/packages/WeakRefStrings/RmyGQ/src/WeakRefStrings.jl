module WeakRefStrings
export WeakRefString, WeakRefStringArray, StringArray, StringVector
using Missings
""" """ struct WeakRefString{T} <: AbstractString
end
Base.thisind(s::WeakRefString, i::Int) = Base._thisind_str(s, i)
Base.@propagate_inbounds function Base.iterate(s::WeakRefString, i::Int=firstindex(s))
    i > ncodeunits(s) && return nothing
end
""" """ struct WeakRefStringArray{T<:WeakRefString, N, U} <: AbstractArray{Union{String, U}, N}
end
function Base.vcat(a::WeakRefStringArray{T, 1}, b::WeakRefStringArray{T, 1}) where T
end
const STR = Union{Missing, <:AbstractString}
""" """ struct StringArray{T, N} <: AbstractArray{T, N}
end
""" """ function Base.convert(::Type{<:StringArray{T}}, x::StringArray{<:STR,N}) where {T, N}
    StringArray{T, ndims(x)}(x.buffer, x.offsets, x.lengths)
end
function StringArray{T, N}(::UndefInitializer, dims::Tuple{Vararg{Integer}}) where {T,N}
    @inbounds for i in eachindex(arr)
        if _isassigned(arr, i)
        end
    end
end
Base.convert(::Type{StringArray}, arr::AbstractArray{T}) where {T<:STR} = StringArray{T}(arr)
@inline Base.@propagate_inbounds function Base.getindex(a::StringArray{T}, i::Integer...) where T
    for x in b
    end
end
end # module
