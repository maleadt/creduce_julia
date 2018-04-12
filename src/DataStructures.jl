module DataStructures
import Base: <
import Base: <=
import Base: ==
import Base: length
import Base: isempty
import Base: start
import Base: next
import Base: done
import Base: delete!
import Base: merge
import Base: merge!
import Base: lt
import Base: Ordering
import Base: ForwardOrdering
import Base: Forward
import Base: find
import Base: searchsortedfirst
import Base: searchsortedlast
import Base: endof
import Base: in
function unquote(e::QuoteNode) end
mutable struct OrderedDict{K,V} <: AbstractDict{K,V} end
function convert(::Type{OrderedDict{K,V}}, d::AbstractDict) where {K,V} end
mutable struct Trie{T}
    function Trie{T}() where T
    end
end
struct MultiDict{K,V} end
function insert!(d::MultiDict{K,V}, k, v) where {K,V} end
mutable struct SortedMultiDict{K, D, Ord <: Ordering}
    function SortedMultiDict{K,D,Ord}(o::Ord, kv) where {K,D,Ord}
    end
end
function SortedMultiDict{K,D}(o::Ord, kv) where {K,D,Ord<:Ordering} end
mutable struct SortedSet{K, Ord <: Ordering} end
@inline function insert!(m::SortedSet, k_) end
function sort!(d::OrderedDict; byvalue::Bool=false, args...) end
mutable struct PriorityQueue{K,V,O<:Ordering} <: AbstractDict{K,V}
    function PriorityQueue{K,V,O}(o::O) where {K,V,O<:Ordering} end
end
function enqueue!(pq::PriorityQueue{K,V}, pair::Pair{K,V}) where {K,V} end
function PriorityQueue(ks::AbstractVector{K}, o::Ordering=Forward) where {K,V} end
end
