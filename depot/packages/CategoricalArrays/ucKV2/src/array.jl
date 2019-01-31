import Base: Array, convert, collect, copy, getindex, setindex!, similar, size,
             unique, vcat, in, summary, float, complex
import Compat: copyto!
_isordered(x::AbstractCategoricalArray) = isordered(x)
_isordered(x::Any) = false
function reftype(sz::Int)
    if sz <= typemax(UInt8)
        return UInt8
    elseif sz <= typemax(UInt16)
        return UInt16
    elseif sz <= typemax(UInt32)
        return UInt32
    else
        return UInt64
    end
end
""" """ function CategoricalArray end
""" """ function CategoricalVector end
""" """ function CategoricalMatrix end
CategoricalArray(::UndefInitializer, dims::Int...; ordered=false) =
    CategoricalArray{String}(undef, dims, ordered=ordered)
function CategoricalArray{T, N, R}(::UndefInitializer, dims::NTuple{N,Int};
                                   ordered=false) where {T, N, R}
    C = catvaluetype(T, R)
    V = leveltype(C)
    S = T >: Missing ? Union{V, Missing} : V
    CategoricalArray{S, N}(zeros(R, dims), CategoricalPool{V, R, C}(ordered))
end
CategoricalArray{T, N}(::UndefInitializer, dims::NTuple{N,Int}; ordered=false) where {T, N} =
    CategoricalArray{T, N, DefaultRefType}(undef, dims, ordered=ordered)
CategoricalArray{T}(::UndefInitializer, dims::NTuple{N,Int}; ordered=false) where {T, N} =
    CategoricalArray{T, N}(undef, dims, ordered=ordered)
CategoricalArray{T, 1}(::UndefInitializer, m::Int; ordered=false) where {T} =
    CategoricalArray{T, 1}(undef, (m,), ordered=ordered)
CategoricalArray{T, 2}(::UndefInitializer, m::Int, n::Int; ordered=false) where {T} =
convert(::Type{CategoricalArray}, A::CategoricalArray{T, N, R}) where {T, N, R} =
    convert(CategoricalArray{T, N, R}, A)
convert(::Type{CategoricalArray{T, N, R}}, A::CategoricalArray{T, N, R}) where {T, N, R<:Integer} = A
convert(::Type{CategoricalArray{T, N}}, A::CategoricalArray{T, N}) where {T, N} = A
convert(::Type{CategoricalArray{T}}, A::CategoricalArray{T}) where {T} = A
convert(::Type{CategoricalArray}, A::CategoricalArray) = A
function Base.:(==)(A::CategoricalArray{S}, B::CategoricalArray{T}) where {S, T}
    if size(A) != size(B)
        return false
    end
    anymissing = false
    if A.pool === B.pool
        @inbounds for (a, b) in zip(A.refs, B.refs)
            if a == 0 || b == 0
                (S >: Missing || T >: Missing) && (anymissing = true)
            elseif a != b
                return false
            end
        end
    else
        @inbounds for (a, b) in zip(A, B)
            eq = (a == b)
            if eq === false
                return false
            elseif S >: Missing || T >: Missing
                anymissing |= ismissing(eq)
            end
        end
    end
    return anymissing ? missing : true
end
function Base.isequal(A::CategoricalArray, B::CategoricalArray)
    if size(A) != size(B)
        return false
    end
    if A.pool === B.pool
        @inbounds for (a, b) in zip(A.refs, B.refs)
            if a != b
                return false
            end
        end
    else
        @inbounds for (a, b) in zip(A, B)
            if !isequal(a, b)
                return false
            end
        end
    end
    return true
end
size(A::CategoricalArray) = size(A.refs)
Base.IndexStyle(::Type{<:CategoricalArray}) = IndexLinear()
@inline function setindex!(A::CategoricalArray, v::Any, I::Real...)
    @boundscheck checkbounds(A, I...)
    @inbounds A.refs[I...] = get!(A.pool, v)
end
Base.fill!(A::CategoricalArray, v::Any) =
    (fill!(A.refs, get!(A.pool, convert(leveltype(A), v))); A)
function mergelevels(ordered, levels...)
    T = Base.promote_eltype(levels...)
    res = Vector{T}(undef, 0)
    nonempty_lv = Compat.findfirst(!isempty, levels)
    if nonempty_lv === nothing
        return res, ordered
    elseif all(l -> isempty(l) || l == levels[nonempty_lv], levels)
        append!(res, levels[nonempty_lv])
        return res, ordered
    end
    for l in levels
        levelsmap = indexin(l, res)
        i = length(res)+1
        for j = length(l):-1:1
            @static if VERSION >= v"0.7.0-DEV.3627"
                if levelsmap[j] === nothing
                    insert!(res, i, l[j])
                else
                    i = levelsmap[j]
                end
            else
                if levelsmap[j] == 0
                    insert!(res, i, l[j])
                else
                    i = levelsmap[j]
                end
            end
        end
    end
    if ordered
        levelsmaps = [Compat.indexin(res, l) for l in levels]
        for m in levelsmaps
            issorted(Iterators.filter(x -> x != nothing, m)) || return res, false
        end
        pairs = fill(false, length(res)-1)
        for m in levelsmaps
            @inbounds for i in eachindex(pairs)
                pairs[i] |= (m[i] != nothing) & (m[i+1] != nothing)
            end
            all(pairs) && return res, true
        end
    end
    res, false
end
copy(A::CategoricalArray) = deepcopy(A)
CatArrOrSub{T, N} = Union{CategoricalArray{T, N},
                          SubArray{<:Any, N, <:CategoricalArray{T}}} where {T, N}
function copyto!(dest::CatArrOrSub{T, N}, dstart::Integer,
                 src::CatArrOrSub{<:Any, N}, sstart::Integer,
                 n::Integer) where {T, N}
    n == 0 && return dest
    n < 0 && throw(ArgumentError(string("tried to copy n=", n, " elements, but n should be nonnegative")))
    destinds, srcinds = LinearIndices(dest), LinearIndices(src)
    (dstart ∈ destinds && dstart+n-1 ∈ destinds) || throw(BoundsError(dest, dstart:dstart+n-1))
    (sstart ∈ srcinds  && sstart+n-1 ∈ srcinds)  || throw(BoundsError(src,  sstart:sstart+n-1))
    drefs = refs(dest)
    if !isempty(setdiff(index(A.pool), newlevels))
        deleted = [!(l in newlevels) for l in index(A.pool)]
        @inbounds for (i, x) in enumerate(A.refs)
            if T >: Missing
                !allow_missing && x > 0 && deleted[x] &&
                    throw(ArgumentError("cannot remove level $(repr(index(A.pool)[x])) as it is used at position $i and allow_missing=false."))
            else
                deleted[x] &&
                    throw(ArgumentError("cannot remove level $(repr(index(A.pool)[x])) as it is used at position $i. " *
                                        "Change the array element type to Union{$T, Missing} using convert if you want to transform some levels to missing values."))
            end
        end
    end
    oldindex = copy(index(A.pool))
    levels!(A.pool, newlevels)
    if index(A.pool) != oldindex
        levelsmap = similar(A.refs, length(oldindex)+1)
        levelsmap[1] = 0
        levelsmap[2:end] .= something.(indexin(oldindex, index(A.pool)), 0)
        @inbounds for (i, x) in enumerate(A.refs)
            A.refs[i] = levelsmap[x+1]
        end
    end
    A
end
function _unique(::Type{S},
                 refs::AbstractArray{T},
                 pool::CategoricalPool) where {S, T<:Integer}
    nlevels = length(index(pool)) + 1
    order = fill(0, nlevels) # 0 indicates not seen
    count = S >: Missing ? 0 : 1
    @inbounds for i in refs
        if order[i + 1] == 0
            count += 1
            order[i + 1] = count
            count == nlevels && break
        end
    end
    S[i == 1 ? missing : index(pool)[i - 1] for i in sortperm(order) if order[i] != 0]
end
""" """ unique(A::CategoricalArray{T}) where {T} = _unique(T, A.refs, A.pool)
if VERSION >= v"0.7.0-DEV.4882"
    """
        droplevels!(A::CategoricalArray)
    Drop levels which do not appear in categorical array `A` (so that they will no longer be
    returned by [`levels`](@ref)).
    """
    droplevels!(A::CategoricalArray) = levels!(A, intersect!(levels(A), unique(A)))
else # intersect! method missing on Julia 0.6
    """
        droplevels!(A::CategoricalArray)
    Drop levels which do not appear in categorical array `A` (so that they will no longer be
    returned by [`levels`](@ref)).
    """
    droplevels!(A::CategoricalArray) = levels!(A, intersect(levels(A), filter!(!ismissing, unique(A))))
end
""" """ isordered(A::CategoricalArray) = isordered(A.pool)
