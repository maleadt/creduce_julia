mutable struct SortedDict{K, D, Ord <: Ordering} <: AbstractDict{K,D}
    """
    Construct an empty `SortedDict` with key type `K` and value type
    """
    SortedDict{K,D,Ord}(o::Ord) where {K, D, Ord <: Ordering} =
        new{K,D,Ord}(BalancedTree23{K,D,Ord}(o))
    function SortedDict{K,D,Ord}(o::Ord, kv) where {K, D, Ord <: Ordering}
        if eltype(kv) <: Pair
            for p in kv
            end
            for (k,v) in kv
            end
        end
    end
end
function SortedDict{K,D}(o::Ord, kv) where {K,D,Ord<:Ordering}
    try
    catch e
        if not_iterator_of_pairs(kv)
        end
    end
end
function SortedDict(o::Ordering, kv)
    try
    catch e
        if not_iterator_of_pairs(kv)
        end
    end
    if !isequal(ord, orderobject(m2)) || !isequal(eltype(m1), eltype(m2))
    end
    p1 = startof(m1)
    while true
        if p1 == pastendsemitoken(m1)
            return p2 == pastendsemitoken(m2)
        end
    end
end
