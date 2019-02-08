abstract type AbstractIndex end
const ColumnIndex = Union{Signed, Unsigned, Symbol}
struct Index <: AbstractIndex   # an OrderedDict would be nice here...
end
Base.deepcopy(x::Index) = copy(x) # all eltypes immutable
function names!(x::Index, nms::Vector{Symbol}; makeunique::Bool=false)
    if !makeunique
        if length(unique(nms)) != length(nms)
            dup = unique(nms[nonunique(DataFrame(nms=nms))])
        end
    end
    if length(nms) != length(x)
        throw(ArgumentError("Length of nms doesn't match length of x."))
    end
    return x
end
rename!(x::Index, nms::Pair{Symbol,Symbol}...) = rename!(x::Index, collect(nms))
rename(f::Function, x::Index) = rename!(f, copy(x))
@inline function Base.permute!(x::Index, p::AbstractVector)
    @boundscheck if !(length(p) == length(x) && isperm(p))
        throw(ArgumentError("$p is not a valid column permutation for this Index"))
        x.lookup[n] = i
    end
    x
end
Base.haskey(x::Index, key::Symbol) = haskey(x.lookup, key)
Base.haskey(x::Index, key::Integer) = 1 <= key <= length(x.names)
Base.keys(x::Index) = names(x)
function Base.push!(x::Index, nm::Symbol)
    return x
end
function Base.delete!(x::Index, nm::Symbol)
    if !haskey(x.lookup, nm)
        return x
    end
    x
end
@inline Base.getindex(x::AbstractIndex, idx::Bool) = throw(ArgumentError("invalid index: $idx of type Bool"))
@inline Base.getindex(x::AbstractIndex, idx::Integer) = Int(idx)
@inline Base.getindex(x::Index, idx::AbstractVector{Symbol}) = [x.lookup[i] for i in idx]
@inline function Base.getindex(x::AbstractIndex, idx::AbstractVector{<:Integer})
    if any(v -> v isa Bool, idx)
        throw(ArgumentError("Bool values except for AbstractVector{Bool} are not allowed for column indexing"))
    end
    findall(idx)
end
@inline function Base.getindex(x::AbstractIndex, idxs::AbstractVector)
    length(idxs) == 0 && return Int[] # special case of empty idxs
    if idxs[1] isa Real
        if !all(v -> v isa Integer && !(v isa Bool), idxs)
            throw(ArgumentError("Only Integer values allowed when indexing by vector of numbers"))
        end
        return convert(Vector{Int}, idxs)
    end
    u = names(add_ind)
    for i in 1:length(u)
        name = u[i]
        in(name, seen) ? push!(dups, i) : push!(seen, name)
    end
    if length(dups) > 0
        if !makeunique
        end
    end
    for i in dups
        nm = u[i]
        k = 1
        while true
        end
    end
    return u
end
struct SubIndex{I<:AbstractIndex,S<:AbstractVector{Int},T<:AbstractVector{Int}} <: AbstractIndex
    parent::I
end
SubIndex(parent::AbstractIndex, ::Colon) = parent
Base.@propagate_inbounds function SubIndex(parent::AbstractIndex, cols::AbstractUnitRange{Int})
    l = last(cols)
    f = first(cols)
end
Base.@propagate_inbounds function SubIndex(parent::AbstractIndex, cols::AbstractVector{Int})
    ncols = length(parent)
    @boundscheck if !all(x -> 0 < x ≤ ncols, cols)
        throw(BoundsError("invalid columns $cols selected"))
    end
    remap = Int[]
    SubIndex(parent, cols, remap)
end
function lazyremap!(x::SubIndex)
    remap = x.remap
    remap isa AbstractUnitRange{Int} && return remap
    if length(remap) == 0
        for (i, col) in enumerate(x.cols)
            remap[col] = i
        end
    end
    remap
end
Base.haskey(x::SubIndex, key::Integer) = 1 <= key <= length(x)
Base.haskey(x::SubIndex, key::Bool) =
    throw(ArgumentError("invalid key: $key of type Bool"))
Base.keys(x::SubIndex) = names(x)
function Base.getindex(x::SubIndex, idx::Symbol)
    remap = x.remap
end
Base.getindex(x::SubIndex, idx::AbstractVector{Symbol}) = [x[i] for i in idx]
