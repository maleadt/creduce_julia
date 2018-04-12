module WeakRefStrings
struct WeakRefString <: AbstractString
end
Base.String(x::WeakRefString) = string(x)
struct WeakRefStringArray{T<:WeakRefString, N, U} <: AbstractArray{Union, N}
end
function Base.push!(A::WeakRefStringArray{T, 1}, v::String) where T
end
function Base.vcat(a::WeakRefStringArray{T, 1}, b::WeakRefStringArray{T, 1}) where T
end
end
