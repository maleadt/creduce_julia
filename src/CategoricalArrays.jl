module CategoricalArrays
using Compat
using JSON
mutable struct CategoricalPool{T, R <: Integer, V}
    index::Vector{T}
    invindex::Dict
end
struct CategoricalValue;
end
struct CategoricalString{R <: Integer} <: AbstractString
    pool::CategoricalPool{String, R, CategoricalString}
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
const CatValue = Union{CategoricalValue where T,
                          CategoricalString}
function pool(x::CatValue)
    x.pool
end
function Base.get(x::CatValue)
    index(pool(x))[]
end
function Compat.lastindex(x::CategoricalString)
    lastindex(get(x))
end
end
