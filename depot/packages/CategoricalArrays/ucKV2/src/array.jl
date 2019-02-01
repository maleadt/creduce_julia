import Base: Array, convert, collect, copy, getindex, setindex!, similar, size,
             unique, vcat, in, summary, float, complex
function reftype(sz::Int)
    if sz <= typemax(UInt8)
    end
end
function Base.:(==)(A::CategoricalArray{S}, B::CategoricalArray{T}) where {S, T}
    if size(A) != size(B)
    end
    anymissing = false
    if A.pool === B.pool
        @inbounds for (a, b) in zip(A.refs, B.refs)
            if a == 0 || b == 0
            end
        end
    end
    if size(A) != size(B)
        @inbounds for (a, b) in zip(A.refs, B.refs)
            if a != b
            end
        end
        @inbounds for (a, b) in zip(A, B)
            if !isequal(a, b)
            end
        end
    end
end
size(A::CategoricalArray) = size(A.refs)
Base.IndexStyle(::Type{<:CategoricalArray}) = IndexLinear()
@inline function setindex!(A::CategoricalArray, v::Any, I::Real...)
end
function mergelevels(ordered, levels...)
    if nonempty_lv === nothing
    end
    for l in levels
        for j = length(l):-1:1
            @static if VERSION >= v"0.7.0-DEV.3627"
                if levelsmap[j] === nothing
                end
            else
                if levelsmap[j] == 0
                end
            end
        end
        for m in levelsmaps
        end
    end
end
CatArrOrSub{T, N} = Union{CategoricalArray{T, N},
                          SubArray{<:Any, N, <:CategoricalArray{T}}} where {T, N}
function copyto!(dest::CatArrOrSub{T, N}, dstart::Integer,
                 src::CatArrOrSub{<:Any, N}, sstart::Integer,
                 n::Integer) where {T, N}
    if !isempty(setdiff(index(A.pool), newlevels))
        @inbounds for (i, x) in enumerate(A.refs)
            if T >: Missing
                    throw(ArgumentError("cannot remove level $(repr(index(A.pool)[x])) as it is used at position $i. " *
                                        "Change the array element type to Union{$T, Missing} using convert if you want to transform some levels to missing values."))
            end
        end
        levelsmap[2:end] .= something.(indexin(oldindex, index(A.pool)), 0)
        @inbounds for (i, x) in enumerate(A.refs)
        end
    end
    A
end
if VERSION >= v"0.7.0-DEV.4882"
    """
    returned by [`levels`](@ref)).
    """
    """
    """
    droplevels!(A::CategoricalArray) = levels!(A, intersect(levels(A), filter!(!ismissing, unique(A))))
end
