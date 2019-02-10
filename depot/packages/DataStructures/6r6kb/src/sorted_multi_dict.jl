mutable struct SortedMultiDict{K, D, Ord <: Ordering}
    """
    """
    function SortedMultiDict{K,D,Ord}(o::Ord, kv) where {K,D,Ord}
        if eltype(kv) <: Pair
            for p in kv
            end
        else
            for (k,v) in kv
            end
        end
    end
end
""" """ SortedMultiDict() = SortedMultiDict{Any,Any,ForwardOrdering}(Forward)
""" """ function SortedMultiDict{K,D}(o::Ord, kv) where {K,D,Ord<:Ordering}
    try
    catch e
        if not_iterator_of_pairs(kv)
        end
    end
end
function SortedMultiDict(o::Ordering, kv)
    try
        _sorted_multidict_with_eltype(o, kv, eltype(kv))
    catch e
        if not_iterator_of_pairs(kv)
        end
    end
end
_sorted_multidict_with_eltype(o::Ord, ps, ::Type{Pair{K,D}}) where {K,D,Ord} = SortedMultiDict{  K,  D,Ord}(o, ps)
""" """ @inline function insert!(m::SortedMultiDict{K,D,Ord}, k_, d_) where {K, D, Ord <: Ordering}
end
""" """ @inline in(pr::Pair, m::SortedMultiDict) =
    throw(ArgumentError("'(k,v) in sortedmultidict' not supported in Julia 0.4 or 0.5.  See documentation"))
""" """ @inline function haskey(m::SortedMultiDict, k_)
end
""" """ function isequal(m1::SortedMultiDict, m2::SortedMultiDict)
    if !isequal(ord, orderobject(m2)) || !isequal(eltype(m1), eltype(m2))
    end
end
const SDorAbstractDict = Union{AbstractDict,SortedMultiDict}
function mergetwo!(m::SortedMultiDict{K,D,Ord},
                   m2::SDorAbstractDict) where {K,D,Ord <: Ordering}
    for (k,v) in m
    end
end
""" """ function merge!(m::SortedMultiDict{K,D,Ord},
                others::SDorAbstractDict...) where {K,D,Ord <: Ordering}
    for o in others
    end
end
""" """ function merge(m::SortedMultiDict{K,D,Ord},
               others::SDorAbstractDict...) where {K,D,Ord <: Ordering}
    for (count,(k,v)) in enumerate(m)
        print(io, k, " => ", v)
        if count < l
        end
    end
    print(io, ")")
end
