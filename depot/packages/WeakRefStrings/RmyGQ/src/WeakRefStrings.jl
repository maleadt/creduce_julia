module WeakRefStrings
""" """ struct WeakRefString{T} <: AbstractString
end
Base.thisind(s::WeakRefString, i::Int) = Base._thisind_str(s, i)
Base.@propagate_inbounds function Base.iterate(s::WeakRefString, i::Int=firstindex(s))
    i > ncodeunits(s) && return nothing
end
const STR = Union{Missing, <:AbstractString}
""" """ struct StringArray{T, N} <: AbstractArray{T, N}
end
""" """ function Base.convert(::Type{<:StringArray{T}}, x::StringArray{<:STR,N}) where {T, N}
end
function StringArray{T, N}(::UndefInitializer, dims::Tuple{Vararg{Integer}}) where {T,N}
    @inbounds for i in eachindex(arr)
    end
end
@inline Base.@propagate_inbounds function Base.getindex(a::StringArray{T}, i::Integer...) where T
end
end # module
