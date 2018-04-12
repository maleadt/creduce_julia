module CategoricalArrays
using Compat
using JSON
mutable struct CategoricalPool{T, R <: Integer, V}
    index::Vector{T}
    invindex::Dict
end
struct CategoricalValue{T, R <: Integer}
end
struct CategoricalString{R <: Integer} <: AbstractString
    level::R
    pool::CategoricalPool{String, R, CategoricalString}
end
abstract type AbstractCategoricalArray{T, N, R, V, C, U} <: AbstractArray{Union, N} end
struct CategoricalArray{T, N, R <: Integer, V, C, U} <: AbstractCategoricalArray{T, N, R, V, C, U}
end
@inline function Base.push!(pool::CategoricalPool, level)
    get!(pool.invindex, level) do
    end
end
function Base.append!(pool::CategoricalPool, levels)
    for level in levels
        push!(pool, level)
    end
end
function index(pool::CategoricalPool)
    pool.index
end
const CatValue{R} = Union{CategoricalValue where T,
                          CategoricalString}
function pool(x::CatValue)
    x.pool
end
function level(x::CatValue)
    x.level
end
function Base.get(x::CatValue)
    index(pool(x))[level(x)]
end
function Compat.lastindex(x::CategoricalString)
    lastindex(get(x))
end
end
