module WeakRefStrings
struct WeakRefString <: AbstractString
end
Base.String(x::WeakRefString) = string
struct WeakRefStringArray{T<:WeakRefString, N, U} <: AbstractArray{Union, N}
end
end
