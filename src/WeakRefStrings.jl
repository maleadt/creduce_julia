module WeakRefStrings
struct WeakRefString <: AbstractString
end
Base.String(x::WeakRefString) = string
struct WeakRefStringArray{T<:WeakRefString, N, U} <: AbstractArray{Union, N}
end
function Base.push!(A::WeakRefStringArray, v::String) where T
end
function Base.vcat(a::WeakRefStringArray, b::WeakRefStringArray) where T
end
end
