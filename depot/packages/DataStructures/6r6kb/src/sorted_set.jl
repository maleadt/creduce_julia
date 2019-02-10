""" """ mutable struct SortedSet{K, Ord <: Ordering}
    bt::BalancedTree23{K,Nothing,Ord}
    function SortedSet{K,Ord}(o::Ord=Forward, iter=[]) where {K,Ord<:Ordering}
    end
end
""" """ SortedSet() = SortedSet{Any,ForwardOrdering}(Forward)
SortedSet(o1::Ordering,o2::Ordering) =
    throw(ArgumentError("SortedSet with two parameters must be called with an Ordering and an interable"))
SortedSet{K}(o1::Ordering,o2::Ordering) where {K} =
    throw(ArgumentError("SortedSet with two parameters must be called with an Ordering and an interable"))
""" """ @inline function pop!(m::SortedSet, k_)
    p2 = startof(m2)
    while true
        if p1 == pastendsemitoken(m1) || p2 == pastendsemitoken(m2)
            if lt(ord,k1,k2)
                p2 = advance((m2,p2))
            end
        end
    end
end
