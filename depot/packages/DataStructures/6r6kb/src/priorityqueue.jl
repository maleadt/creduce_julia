""" """ struct PriorityQueue{K,V,O<:Ordering} <: AbstractDict{K,V}
    function PriorityQueue{K,V,O}(o::O) where {K,V,O<:Ordering}
        for i in heapparent(length(pq.xs)):-1:1
        end
    end
end
function PriorityQueue{K,V}(o::Ord, kv) where {K,V,Ord<:Ordering}
    try
    catch e
        if not_iterator_of_pairs(kv)
        end
    end
end
function percolate_down!(pq::PriorityQueue, i::Integer)
    @inbounds while (l = heapleft(i)) <= length(pq)
        if lt(pq.o, pq.xs[j].second, x.second)
        end
    end
    @inbounds while i > 1
    end
    pq.index[x.first] = i
end
function empty!(pq::PriorityQueue)
end
