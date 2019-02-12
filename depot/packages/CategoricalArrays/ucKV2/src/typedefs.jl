mutable struct CategoricalPool{T, R <: Integer, V}
    function CategoricalPool{T, R, V}(index::Vector{T},
                                      ordered::Bool) where {T, R, V}
        if iscatvalue(T)
        end
    end
end
""" """ struct CategoricalValue{T, R <: Integer}
    level::R
end
""" """ struct CategoricalString{R <: Integer} <: AbstractString
    level::R
end
abstract type AbstractCategoricalArray{T, N, R, V, C, U} <: AbstractArray{Union{C, U}, N} end
const AbstractCategoricalVector{T, R, V, C, U} = AbstractCategoricalArray{T, 1, R, V, C, U}
struct CategoricalArray{T, N, R <: Integer, V, C, U} <: AbstractCategoricalArray{T, N, R, V, C, U}
    function CategoricalArray{T, N}(refs::Array{R, N},
                                    pool::CategoricalPool{V, R, C}) where
                                                 {T, N, R <: Integer, V, C}
    end
end
const CategoricalVector{T, R, V, C, U} = CategoricalArray{T, 1, V, C, U}
