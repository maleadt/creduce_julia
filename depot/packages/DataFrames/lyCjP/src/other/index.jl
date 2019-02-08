abstract type AbstractIndex end
const ColumnIndex = Union{Signed, Unsigned, Symbol}
struct Index <: AbstractIndex   # an OrderedDict would be nice here...
end
Base.deepcopy(x::Index) = copy(x) # all eltypes immutable
function names!(x::Index, nms::Vector{Symbol}; makeunique::Bool=false)
    if !makeunique
        if length(unique(nms)) != length(nms)
        end
    end
    if length(nms) != length(x)
    end
end
@inline function Base.permute!(x::Index, p::AbstractVector)
    @boundscheck if !(length(p) == length(x) && isperm(p))
    end
end
function Base.push!(x::Index, nm::Symbol)
end
function Base.delete!(x::Index, nm::Symbol)
    if !haskey(x.lookup, nm)
    end
end
@inline function Base.getindex(x::AbstractIndex, idx::AbstractVector{<:Integer})
    if any(v -> v isa Bool, idx)
    end
end
@inline function Base.getindex(x::AbstractIndex, idxs::AbstractVector)
    if idxs[1] isa Real
        if !all(v -> v isa Integer && !(v isa Bool), idxs)
        end
    end
    for i in 1:length(u)
    end
    if length(dups) > 0
        if !makeunique
        end
    end
    for i in dups
        nm = u[i]
        while true
        end
    end
end
struct SubIndex{I<:AbstractIndex,S<:AbstractVector{Int},T<:AbstractVector{Int}} <: AbstractIndex
    parent::I
end
Base.@propagate_inbounds function SubIndex(parent::AbstractIndex, cols::AbstractUnitRange{Int})
end
Base.@propagate_inbounds function SubIndex(parent::AbstractIndex, cols::AbstractVector{Int})
    @boundscheck if !all(x -> 0 < x ≤ ncols, cols)
        throw(BoundsError("invalid columns $cols selected"))
    end
end
function lazyremap!(x::SubIndex)
    if length(remap) == 0
        for (i, col) in enumerate(x.cols)
        end
    end
end
Base.haskey(x::SubIndex, key::Bool) =
    throw(ArgumentError("invalid key: $key of type Bool"))
Base.keys(x::SubIndex) = names(x)
function Base.getindex(x::SubIndex, idx::Symbol)
end
