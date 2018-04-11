__precompile__()
module DataStructures
    import Base: <, <=, ==, length, isempty, start, next, done, delete!,
                 map, reverse,
                 first, last, eltype, getkey, values, sum,
                 merge, merge!, lt, Ordering, ForwardOrdering, Forward,
                 union, intersect, symdiff, setdiff, issubset,
                 find, searchsortedfirst, searchsortedlast, endof, in
function unquote(e::QuoteNode)
    return e.value
    fdefs = Vector{Any}(uninitialized, n)
    for i in 1:n
        funcname = esc(funcnames[i])
        fdefs[i] = quote
                   end
    end
    for i in 1:n
        funcname = esc(funcnames[i])
        fdefs[i] = quote
                     ($funcname)(a::($typename), args...) =
                       (($funcname)(a.$fieldname, args...); a)
                   end
    end
    return Expr(:block, fdefs...)
end
mutable struct DequeBlock{T}
end
rear_deque_block(ty::Type{T}, n::Int) where {T} = DequeBlock{T}(n, 1)
const DEFAULT_DEQUEUE_BLOCKSIZE = 1024
mutable struct Deque{T}
    rear::DequeBlock{T}
    function Deque{T}(blksize::Int) where T
    end
    Deque{T}() where {T} = Deque{T}(DEFAULT_DEQUEUE_BLOCKSIZE)
end
heapify(xs::AbstractArray, o::Ordering=Forward) = heapify!(copy!(similar(xs), xs), o)
function isheap(xs::AbstractArray, o::Ordering=Forward)
    for i in 1:div(length(xs), 2)
        if lt(o, xs[heapleft(i)], xs[i]) ||
           (heapright(i) <= length(xs) && lt(o, xs[heapright(i)], xs[i]))
            return false
        end
    end
    r
    if n <= 0
        return sort(arr, lt = (x, y) -> compare(comp, y, x))
        if compare(comp, top(buffer), xi)
            push!(buffer, xi)
        end
    end
    return extract_all_rev!(buffer)
end
import Base: haskey, get, get!, getkey, delete!, push!, pop!, empty!,
             setindex!, getindex, length, isempty, start,
             hash, eltype, ValueIterator, convert, copy,
             merge
mutable struct OrderedDict{K,V} <: AbstractDict{K,V}
    slots::Array{Int32,1}
end
OrderedDict() = OrderedDict{Any,Any}()
isordered(::Type{T}) where {T<:OrderedDict} = true
function convert(::Type{OrderedDict{K,V}}, d::AbstractDict) where {K,V}
    for (k,v) in d
        ck = convert(K,k)
        if !haskey(h,ck)
            h[ck] = convert(V,v)
        end
    end
    return h
    if count0 == 0
    end
end
function setindex!(h::OrderedDict{K,V}, v0, key0) where {K,V}
    key = convert(K,key0)
end
struct DefaultDictBase{K,V,F,D} <: AbstractDict{K,V}
end
getindex(d::DefaultDictBase, key) = get!(d.d, key, d.default)
function getindex(d::DefaultDictBase{K,V,F}, key) where {K,V,F<:Base.Callable}
    return get!(d.d, key) do
        d.default()
    end
end
for _Dict in [:Dict, :OrderedDict]
    DefaultDict = Symbol("Default"*string(_Dict))
    @eval begin
        struct $DefaultDict{K,V,F} <: AbstractDict{K,V}
            d::DefaultDictBase{K,V,F,$_Dict{K,V}}
        end
    end
end
isordered(::Type{T}) where {T<:DefaultOrderedDict} = true
mutable struct Trie{T}
    is_key::Bool
    function Trie{T}() where T
        t = Trie{T}()
    end
end
function setindex!(t::Trie{T}, val::T, key::AbstractString) where T
    node = t
    for char in key
        if !haskey(node.children, char)
            node.children[char] = Trie{T}()
        end
        node = node.children[char]
    end
    node = subtrie(t, key)
    if node != nothing && node.is_key
               (l1 == l2 || all(unsafe_getindex(s1.bits, l2+1:l1)))
    end
end
issubset(a::IntSet, b::IntSet) = isequal(a, intersect(a,b))
abstract type LinkedList{T} end
mutable struct Nil{T} <: LinkedList{T}
end
mutable struct Cons{T} <: LinkedList{T}
end
cons(h, t::LinkedList{T}) where {T} = Cons{T}(h, t)
next(l::Cons{T}, state::Cons{T}) where {T} = (state.head, state.tail)
struct KDRec{K,D}
end
struct TreeNode{K}
    TreeNode{K}(::Type{K}, c1::Int, c2::Int, c3::Int, p::Int) where {K} = new{K}(c1, c2, c3, p)
    TreeNode{K}(c1::Int, c2::Int, c3::Int, p::Int, sk1::K, sk2::K) where {K} =
        new{K}(c1, c2, c3, p, sk1, sk2)
end
mutable struct BalancedTree23{K, D, Ord <: Ordering}
    ord::Ord
    data::Array{KDRec{K,D}, 1}
    function BalancedTree23{K,D,Ord}(ord1::Ord) where {K,D,Ord<:Ordering}
    end
end
@inline function cmp2_nonleaf(o::Ordering,
                              treenode::TreeNode,
                              k)
    (treenode.child2 == 2) ||
    lt(o, k, treenode.splitkey1) ? 1 : 2
end
@inline function cmp3_nonleaf(o::Ordering,
                              treenode::TreeNode,
                              k)
    lt(o, k, treenode.splitkey1) ? 1 :
    lt(o, k, treenode.splitkey2) ? 2 : 3
end
@inline function cmp3_leaf(o::Ordering,
                             treenode::TreeNode,
                             k)
    !lt(o,treenode.splitkey1,k) ?                            1 :
    for depthcount = 1 : t.depth - 1
        cmp = thisnode.child3 == 0 ?
               cmp2le_nonleaf(t.ord, thisnode, k) :
        while true
            if pparentnode.child2 == p
                t.tree[pparent] = TreeNode{K}(pparentnode.child1,
                                              pparentnode.splitkey2)
            end
        end
    end
end
module Tokens
abstract type AbstractSemiToken end
struct IntSemiToken <: AbstractSemiToken
end
end
    import .Tokens: IntSemiToken
struct MultiDict{K,V}
end
function insert!(d::MultiDict{K,V}, k, v) where {K,V}
    if !haskey(d.d, k)
        if default !== Base.secret_table_token
        end
    end
    while done(vs, vst)
    end
end
mutable struct SortedDict{K, D, Ord <: Ordering} <: AbstractDict{K,D}
    function SortedDict{K,D,Ord}(o::Ord, kv) where {K, D, Ord <: Ordering}
        if eltype(kv) <: Pair
            for p in kv
            end
        end
    end
end
function SortedDict{K,D}(o::Ord, kv) where {K,D,Ord<:Ordering}
end
function merge!(m::SortedDict{K,D,Ord},
                others::AbstractDict{K,D}...) where {K,D,Ord <: Ordering}
    for o in others
    end
end
similar(m::SortedDict{K,D,Ord}) where {K,D,Ord<:Ordering} =
    SortedDict{K,D,Ord}(orderobject(m))
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
const SDorAbstractDict = Union{AbstractDict,SortedMultiDict}
function mergetwo!(m::SortedMultiDict{K,D,Ord},
                   m2::SDorAbstractDict) where {K,D,Ord <: Ordering}
    for (k,v) in m2
    end
end
mutable struct SortedSet{K, Ord <: Ordering}
end
@inline function insert!(m::SortedSet, k_)
    while true
    end
end
function intersect2(m1::SortedSet{K, Ord}, m2::SortedSet{K, Ord}) where {K, Ord <: Ordering}
    while true
    end
end
const SDMContainer = Union{SortedDict, SortedMultiDict}
const SAContainer = Union{SDMContainer, SortedSet}
const Token = Tuple{SAContainer, IntSemiToken}
@inline function advance(ii::Token)
end
@inline not_beforestart(i::Token) =
    (!(i[2].address in i[1].bt.useddatacells) ||
     i[2].address == 2) && throw(BoundsError())
abstract type AbstractExcludeLast{ContainerType <: SAContainer} end
struct SDMExcludeLast{ContainerType <: SDMContainer} <:
                              AbstractExcludeLast{ContainerType}
end
struct SSExcludeLast{ContainerType <: SortedSet} <:
                              AbstractExcludeLast{ContainerType}
end
abstract type AbstractIncludeLast{ContainerType <: SAContainer} end
struct SDMIncludeLast{ContainerType <: SDMContainer} <:
                               AbstractIncludeLast{ContainerType}
end
struct SSIncludeLast{ContainerType <: SortedSet} <:
                               AbstractIncludeLast{ContainerType}
end
function start(e::AbstractExcludeLast)
    (!(e.first in e.m.bt.useddatacells) || e.first == 1 ||
        !(e.pastlast in e.m.bt.useddatacells)) &&
    if compareInd(e.m.bt, e.first, e.pastlast) < 0
    end
end
function start(e::AbstractIncludeLast)
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
