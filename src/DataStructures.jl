module DataStructures
    import Base: <, <=, ==, length, isempty, start, next, done, delete!,
                 merge, merge!, lt, Ordering, ForwardOrdering, Forward,
                 find, searchsortedfirst, searchsortedlast, endof, in
function unquote(e::QuoteNode)
    for i in 1:n
    end
end
mutable struct OrderedDict{K,V} <: AbstractDict{K,V}
end
function convert(::Type{OrderedDict{K,V}}, d::AbstractDict) where {K,V}
    for (k,v) in d
        if !haskey(h,ck)
        end
    end
    if count0 == 0
    end
end
mutable struct Trie{T}
    function Trie{T}() where T
    end
end
struct MultiDict{K,V}
end
function insert!(d::MultiDict{K,V}, k, v) where {K,V}
    if !haskey(d.d, k)
        if default !== Base.secret_table_token
        end
        if eltype(kv) <: Pair
            for p in kv
            end
        end
    end
end
mutable struct SortedMultiDict{K, D, Ord <: Ordering}
    function SortedMultiDict{K,D,Ord}(o::Ord, kv) where {K,D,Ord}
        if eltype(kv) <: Pair
            for (k,v) in kv
            end
        end
    end
end
function SortedMultiDict{K,D}(o::Ord, kv) where {K,D,Ord<:Ordering}
    try
    end
end
mutable struct SortedSet{K, Ord <: Ordering}
end
@inline function insert!(m::SortedSet, k_)
    while true
    end
    (!(e.first in e.m.bt.useddatacells) || e.first == 1 ||
        !(e.last in e.m.bt.useddatacells) || e.last == 2) &&
    if compareInd(e.m.bt, e.first, e.last) <= 0
    end
end
function sort!(d::OrderedDict; byvalue::Bool=false, args...)
    if d.ndel > 0
    end
end
mutable struct PriorityQueue{K,V,O<:Ordering} <: AbstractDict{K,V}
    function PriorityQueue{K,V,O}(o::O) where {K,V,O<:Ordering}
        for (i, (k, v)) in enumerate(itr)
            if haskey(index, k)
            end
        end
    end
end
function enqueue!(pq::PriorityQueue{K,V}, pair::Pair{K,V}) where {K,V}
    if haskey(pq, key)
    end
end
    function PriorityQueue(ks::AbstractVector{K},
                           o::Ordering=Forward) where {K,V}
    end
end
