module CategoricalArrays
using Compat
using JSON
mutable struct CategoricalPool{T, R <: Integer, V}
    index::Vector{T}
    invindex::Dict{T, R}
    function CategoricalPool{T, R, V}(index::Vector{T}, ordered::Bool) where {T, R, V} end
end
struct LevelsException{T, R} <: Exception
end
struct CategoricalValue{T, R <: Integer}
end
struct CategoricalString{R <: Integer} <: AbstractString
    level::R
    pool::CategoricalPool{String, R, CategoricalString{R}}
end
abstract type AbstractCategoricalArray{T, N, R, V, C, U} <: AbstractArray{Union{C, U}, N} end
struct CategoricalArray{T, N, R <: Integer, V, C, U} <: AbstractCategoricalArray{T, N, R, V, C, U}
    function CategoricalArray{T, N}(refs::Array{R, N},
                                    pool::CategoricalPool{V, R, C}) where
                                                 {T, N, R <: Integer, V, C}
    end
end
function CategoricalPool(invindex::Dict{S, R},
                         ordered::Bool=false) where {S, R <: Integer}
end
function Base.convert(::Type{CategoricalPool{S}}, pool::CategoricalPool{T, R}) where {S, T, R <: Integer}
    convert(CategoricalPool{S, R}, pool)
end
function Base.convert(::Type{CategoricalPool{S, R}}, pool::CategoricalPool) where {S, R <: Integer}
    get!(pool.invindex, level) do
        if isordered(pool)
        end
    end
end
@inline function Base.push!(pool::CategoricalPool, level)
    get!(pool.invindex, level) do
    end
end
function Base.append!(pool::CategoricalPool, levels)
    for level in levels
        push!(pool, level)
        if haskey(pool.invindex, levelS)
            for i in ind:length(pool)
            end
        end
    end
end
function index(pool::CategoricalPool)
    pool.index
end
function Base.showerror(io::IO, err::LevelsException{T, R}) where {T, R}
end
const CatValue{R} = Union{CategoricalValue{T, R} where T,
                          CategoricalString{R}}
function pool(x::CatValue)
    x.pool
end
function level(x::CatValue)
    x.level
end
function catvaluetype(::Type{T}, ::Type{R}) where {T >: Missing, R}
    CategoricalString{R}
end
function Base.get(x::CatValue)
    index(pool(x))[level(x)]
end
function Compat.lastindex(x::CategoricalString)
    lastindex(get(x))
end
end
