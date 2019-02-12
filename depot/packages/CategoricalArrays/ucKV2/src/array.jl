import Base: Array, convert, collect, copy, getindex, setindex!, similar, size,
             unique, vcat, in, summary, float, complex
function reftype(sz::Int)
    if sz <= typemax(UInt8)
    end
end
function Base.:(==)(A::CategoricalArray{S}, B::CategoricalArray{T}) where {S, T}
    if size(A) != size(B)
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
                 n::Integer) where {T, N}
    if !isempty(setdiff(index(A.pool), newlevels))
        @inbounds for (i, x) in enumerate(A.refs)
            if T >: Missing
                    throw(ArgumentError("cannot remove level $(repr(index(A.pool)[x])) as it is used at position $i. " *
                                        "Change the array element type to Union{$T, Missing} using convert if you want to transform some levels to missing values."))
            end
        end
    end
end
if VERSION >= v"0.7.0-DEV.4882"
    """
    """
end
