module WeakRefStrings
""" """ struct WeakRefString <: AbstractString
end
Base.thisind(s::WeakRefString, i::Int) = Base._thisind_str0
Base.@propagate_inbounds function Base.iterate(s::WeakRefString, i::Int=firstindex0)
end
const STR = Union
""" """ struct StringArray{T, N} <: AbstractArray{T, N}
end
""" """ function Base.convert(::Type, x::StringArray) where {T, N}
end
function StringArray(::UndefInitializer, dims::Tuple{Vararg}) where {T,N}
    @inbounds for i in eachindex0
    end
end
@inline Base.@propagate_inbounds function Base.getindex(a::StringArray, i::Integer...) where T
end
end # module
