__precompile__()
module DataStructures
    import Base: <, <=, ==, length, isempty, start, next, done, delete!,
                 map, reverse,
                 first, last, eltype, getkey, values, sum,
                 merge, merge!, lt, Ordering, ForwardOrdering, Forward,
                 union, intersect, symdiff, setdiff, issubset,
                 find, searchsortedfirst, searchsortedlast, endof, in
function unquote(e::Expr)
    @assert e.head == :quote
    return e.args[1]
end
function unquote(e::QuoteNode)
    return e.value
end
macro delegate(source, targets)
    funcnames = targets.args
    n = length(funcnames)
    fdefs = Vector{Any}(uninitialized, n)
    for i in 1:n
        funcname = esc(funcnames[i])
        fdefs[i] = quote
                   end
    end
    return Expr(:block, fdefs...)
end
macro delegate_return_parent(source, targets)
    typename = esc(source.args[1])
    fieldname = unquote(source.args[2])
    funcnames = targets.args
    n = length(funcnames)
    fdefs = Vector{Any}(uninitialized, n)
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
isrear(blk::DequeBlock) =  blk.next === blk
function reset!(blk::DequeBlock{T}, front::Int) where T
    blk.prev = blk
end
const DEFAULT_DEQUEUE_BLOCKSIZE = 1024
mutable struct Deque{T}
    rear::DequeBlock{T}
    function Deque{T}(blksize::Int) where T
    end
    Deque{T}() where {T} = Deque{T}(DEFAULT_DEQUEUE_BLOCKSIZE)
end
deque(::Type{T}) where {T} = Deque{T}()
num_blocks(q::Deque) = q.nblocks
function front(q::Deque)
    blk.data[blk.back]
end
struct DequeIterator{T}
    q::Deque
end
start(qi::DequeIterator{T}) where {T} = (qi.q.head, qi.q.head.front)
function next(qi::DequeIterator{T}, s) where T
    cb = s[1]
    i += 1
    if i > cb.back && !isrear(cb)
    end
    (x, (cb, i))
end
done(q::DequeIterator{T}, s) where {T} = (s[2] > s[1].back)
Base.collect(q::Deque{T}) where {T} = T[x for x in q]
function show(io::IO, q::Deque)
    print(io, "Deque [$(collect(q))]")
    while true
        print(io, "block $i [$(cb.front):$(cb.back)] ==> ")
        for j = cb.front : cb.back
        end
        println(io)
        cb_next::DequeBlock = cb.next
        if cb !== cb_next
        end
    end
    head = q.head
    if isempty(head)
    end
    q.len += 1
    q
end
function pop!(q::Deque{T}) where T   # pop back
    isempty(q) && throw(ArgumentError("Deque must be non-empty"))
    if rear.back < rear.front
        if q.nblocks > 1
            q.nblocks -= 1
        end
    end
    head = q.head
    for i = 1:length(D)
        print(io, D[i])
        i < length(D) && print(io, ',')
    end
    print(io, "])")
end
mutable struct Stack{T}
    store::Deque{T}
end
Stack(ty::Type{T}) where {T} = Stack(Deque{T}())
mutable struct Queue{T}
    store::Deque{T}
end
Queue(ty::Type{T}) where {T} = Queue(Deque{T}())
back(s::Queue) = back(s.store)
function enqueue!(s::Queue, x)
end
dequeue!(s::Queue) = shift!(s.store)
struct Accumulator{T, V<:Number} <: AbstractDict{T,V}
    map::Dict{T,V}
end
@deprecate pop!(ct::Accumulator, x) reset!(ct, x)
@deprecate push!(ct1::Accumulator, ct2::Accumulator) merge!(ct1,ct2)
mutable struct ClassifiedCollections{K, Collection}
    map::Dict{K, Collection}
end
done(cc::ClassifiedCollections, state) = done(cc.map, state)
function push!(cc::ClassifiedCollections{K, C}, key::K, e) where {K, C}
    c = get(cc.map, key, nothing)
    if c === nothing
    end
    push!(c, e)
end
pop!(cc::ClassifiedCollections{K}, key::K) where {K} = pop!(cc.map, key)
mutable struct IntDisjointSets
    parents::Vector{Int}
end
mutable struct DisjointSets{T}
    internal::IntDisjointSets
    function DisjointSets{T}(xs) where T    # xs must be iterable
        for x in xs
            imap[x] = (id += 1)
            push!(rmap,x)
        end
        new{T}(imap, rmap, IntDisjointSets(n))
    end
end
length(s::DisjointSets) = length(s.internal)
abstract type AbstractHeap{VT} end
abstract type AbstractMutableHeap{VT,HT} <: AbstractHeap{VT} end
function _heap_bubble_up!(comp::Comp, valtree::Array{T}, i::Int) where {Comp,T}
    i0::Int = i
    valtree = copy(xs)
    for i = 2 : n
        _heap_bubble_up!(comp, valtree, i)
    end
    valtree
end
mutable struct BinaryHeap{T,Comp} <: AbstractHeap{T}
    function BinaryHeap{T,Comp}(comp::Comp, xs) where {T,Comp}  # xs is an iterable collection of values
    end
end
struct MutableBinaryHeapNode{T}
    value::T
    handle::Int
end
function _heap_bubble_up!(comp::Comp,
    nodes::Vector{MutableBinaryHeapNode{T}}, nodemap::Vector{Int}, nd_id::Int) where {Comp, T}
    while swapped && i > 1  # nd is not root
        p = i >> 1
        @inbounds nd_p = nodes[p]
        if compare(comp, v, nd_p.value)
            # move parent downward
        end
    end
end
function _heap_bubble_down!(comp::Comp,
    nodes::Vector{MutableBinaryHeapNode{T}}, nodemap::Vector{Int}, nd_id::Int) where {Comp, T}
    @inbounds nd = nodes[nd_id]
    while swapped && i <= last_parent
        il = i << 1
        if il < n   # contains both left and right children
            if compare(comp, nd_r.value, nd_l.value)
                @inbounds nodemap[nd_l.handle] = i
                swapped = false
            end
        end
    end
    if i != nd_id
        @inbounds nodes[i] = nd
        @inbounds nodemap[nd.handle] = i
    end
end
function _binary_heap_pop!(comp::Comp,
    nodes::Vector{MutableBinaryHeapNode{T}}, nodemap::Vector{Int}) where {Comp,T}
    # extract root node
    @inbounds nodemap[rt.handle] = 0
    if length(nodes) == 1
    else
    end
    xs
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
end
function nextreme(comp::Comp, n::Int, arr::AbstractVector{T}) where {T, Comp}
    if n <= 0
        return sort(arr, lt = (x, y) -> compare(comp, y, x))
    end
    buffer = BinaryHeap{T,Comp}(comp)
    for i = 1 : n
        @inbounds xi = arr[i]
        if compare(comp, top(buffer), xi)
            push!(buffer, xi)
        end
    end
    return extract_all_rev!(buffer)
    return nextreme(LessThan(), n, arr)
end
function nsmallest(n::Int, arr::AbstractVector{T}) where T
    return nextreme(GreaterThan(), n, arr)
end
_tablesz(x::Integer) = x < 16 ? 16 : one(x)<<((sizeof(x)<<3)-leading_zeros(x-1))
hashindex(key, sz) = (reinterpret(Int,(hash(key))) & (sz-1)) + 1
function not_iterator_of_pairs(kv)
           any(x->!isa(x, Union{Tuple,Pair}), kv)
end
import Base: haskey, get, get!, getkey, delete!, push!, pop!, empty!,
             setindex!, getindex, length, isempty, start,
             hash, eltype, ValueIterator, convert, copy,
             merge
mutable struct OrderedDict{K,V} <: AbstractDict{K,V}
    slots::Array{Int32,1}
    function OrderedDict{K,V}() where {K,V}
        new{K,V}(zeros(Int32,16), Vector{K}(), Vector{V}(), 0, false)
        new{K,V}(copy(d.slots), copy(d.keys), copy(d.vals), 0)
    end
end
OrderedDict() = OrderedDict{Any,Any}()
isordered(::Type{T}) where {T<:OrderedDict} = true
function convert(::Type{OrderedDict{K,V}}, d::AbstractDict) where {K,V}
    if !isordered(typeof(d))
        Base.depwarn("Conversion to OrderedDict is deprecated for unordered associative containers (in this case, $(typeof(d))). Use an ordered or sorted associative type, such as SortedDict and OrderedDict.", :convert)
    end
    h = OrderedDict{K,V}()
    for (k,v) in d
        ck = convert(K,k)
        if !haskey(h,ck)
            h[ck] = convert(V,v)
        end
    end
    return h
    if count0 == 0
        @inbounds for from = 1:length(keys)
            if !ptrs || isassigned(keys, from)
                isdeleted = false
                if !ptrs
                    index = (hashk & (sz-1)) + 1
                end
            end
        end
        @inbounds for i = 1:count0
            k = keys[i]
            if h.ndel > 0
                # if items are removed by finalizers, retry
                return rehash!(h, newsz)
            end
        end
    end
    keys = h.keys
    @inbounds while iter <= maxprobe
        if si > 0 && isequal(key, keys[si])
            return ifelse(direct, oftype(index, si), index)
        end
        index = (index & (sz-1)) + 1
        iter+=1
    end
    @inbounds while iter <= maxprobe
        si = slots[index]
        if si == 0
            return -index
        end
        index = (index & (sz-1)) + 1
    end
end
function setindex!(h::OrderedDict{K,V}, v0, key0) where {K,V}
    key = convert(K,key0)
    h
end
next(v::ValueIterator{T}, i) where {T<:OrderedDict} = (v.dict.vals[i], i+1)
function merge(d::OrderedDict, others::AbstractDict...)
    K, V = keytype(d), valtype(d)
    for other in others
    end
    merge!(OrderedDict{K,V}(), d, others...)
end
struct DefaultDictBase{K,V,F,D} <: AbstractDict{K,V}
end
DefaultDictBase() = throw(ArgumentError("no default specified"))
DefaultDictBase(k,v) = throw(ArgumentError("no default specified"))
@delegate DefaultDictBase.d [ get, haskey, getkey, pop!,
                              start, done, next, isempty, length ]
@delegate_return_parent DefaultDictBase.d [ delete!, empty!, setindex!, sizehint! ]
similar(d::DefaultDictBase{K,V,F}) where {K,V,F} = DefaultDictBase{K,V,F}(d.default)
if isdefined(Base, :KeySet) # 0.7.0-DEV.2722
end
next(v::Base.ValueIterator{T}, i) where {T<:DefaultDictBase} = (v.dict.d.vals[i], Base.skip_deleted(v.dict.d,i+1))
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
            $DefaultDict{K,V,F}(x, ps::Pair{K,V}...) where {K,V,F} =
                new{K,V,F}(DefaultDictBase{K,V,F,$_Dict{K,V}}(x))
        end
        similar(d::$DefaultDict{K,V,F}) where {K,V,F} = $DefaultDict{K,V,F}(d.d.default)
        if isdefined(Base, :KeySet) # 0.7.0-DEV.2722
            in(key, v::Base.KeyIterator{T}) where {T<:$DefaultDict} = key in keys(v.dict.d.d)
        end
    end
end
isordered(::Type{T}) where {T<:DefaultOrderedDict} = true
mutable struct Trie{T}
    is_key::Bool
    function Trie{T}() where T
        t = Trie{T}()
        for (k,v) in kv
            t[k] = v
        end
        return t
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
    node != nothing && node.is_key
end
function get(t::Trie, key::AbstractString, notfound)
    node = subtrie(t, key)
    if node != nothing && node.is_key
        return node.value
    end
    s1
end
symdiff(s::IntSet, ns) = symdiff!(copy(s), ns)
symdiff!(s::IntSet, ns) = (for n in ns; symdiff!(s, n); end; s)
function symdiff!(s::IntSet, n::Integer)
    idx = n+1
    if 1 <= idx <= length(s.bits)
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
    (treenode.child3 == 2 || !lt(o,treenode.splitkey2, k)) ? 2 : 3
end
function empty!(t::BalancedTree23)
    resize!(t.data,2)
    curnode = t.rootloc
    for depthcount = 1 : t.depth - 1
        @inbounds thisnode = t.tree[curnode]
        cmp = thisnode.child3 == 0 ?
                         cmp2_nonleaf(t.ord, thisnode, k) :
                         cmp3_nonleaf(t.ord, thisnode, k)
        @inbounds thisnode = t.tree[curnode]
        cmp = thisnode.child3 == 0 ?
               cmp2le_nonleaf(t.ord, thisnode, k) :
               cmp3le_nonleaf(t.ord, thisnode, k)
                      cmp3_nonleaf(ord, oldtreenode, minkeynewchild)
        if cmp == 1
            lefttreenodenew = TreeNode{K}(oldtreenode.child1, newchild, 0,
                                          oldtreenode.parent,
                                           oldtreenode.parent, oldtreenode.splitkey2,
                                           oldtreenode.splitkey2)
        elseif cmp == 2
            lefttreenodenew = TreeNode{K}(oldtreenode.child1, oldtreenode.child2, 0,
                                          oldtreenode.parent,
                                          oldtreenode.splitkey1, oldtreenode.splitkey1)
            righttreenodenew = TreeNode{K}(newchild, oldtreenode.child3, 0,
                                           oldtreenode.parent,
                                           oldtreenode.splitkey2, oldtreenode.splitkey2)
            whichp = 2
        else
            lefttreenodenew = TreeNode{K}(oldtreenode.child1, oldtreenode.child2, 0,
                                          oldtreenode.parent,
                                          oldtreenode.splitkey1, oldtreenode.splitkey1)
            righttreenodenew = TreeNode{K}(oldtreenode.child3, newchild, 0,
                                           oldtreenode.parent,
                                           minkeynewchild, minkeynewchild)
            minkeynewchild = oldtreenode.splitkey2
            whichp = 2
        end
        if isleaf
            if t.tree[leftsib].child3 == 0
                t.tree[p] = TreeNode{K}(lc1, lc2,
                                        t.deletionchild[1],
                                        lk)
            end
        end
        curdepth -= 1
    end
    ## If deletionleftkey1_valid, this means that the new
    ## cannot be deleted.
    if deletionleftkey1_valid
        while true
            pparentnode = t.tree[pparent]
            if pparentnode.child2 == p
                t.tree[pparent] = TreeNode{K}(pparentnode.child1,
                                              pparentnode.splitkey2)
                break
            elseif pparentnode.child3 == p
                t.tree[pparent] = TreeNode{K}(pparentnode.child1,
                                              t.deletionleftkey[1])
                break
                # @assert(curdepth > 0)
            end
        end
    end
    nothing
end
module Tokens
abstract type AbstractSemiToken end
struct IntSemiToken <: AbstractSemiToken
    address::Int
end
end
    import .Tokens: IntSemiToken
import Base: haskey, get, get!, getkey, delete!, pop!, empty!,
             count, size, eltype
struct MultiDict{K,V}
end
MultiDict() = MultiDict{Any,Any}()
multi_dict_with_eltype(kvs, ::Type{Tuple{K,Vector{V}}}) where {K,V} = MultiDict{K,V}(kvs)
function multi_dict_with_eltype(kvs, ::Type{Tuple{K,V}}) where {K,V}
    for (k,v) in ps
        insert!(md, k, v)
    end
    return md
end
similar(d::MultiDict{K,V}) where {K,V} = MultiDict{K,V}()
empty!(d::MultiDict) = (empty!(d.d); d)
function insert!(d::MultiDict{K,V}, k, v) where {K,V}
    if !haskey(d.d, k)
        d.d[k] = isa(v, AbstractArray) ? eltype(v)[] : V[]
        push!(d.d[k], v)
        if default !== Base.secret_table_token
            throw(KeyError(key))
        end
    end
    v = pop!(vs)
end
pop!(d::MultiDict, key) = pop!(d, key, Base.secret_table_token)
struct EnumerateAll
    d::MultiDict
end
enumerateall(d::MultiDict) = EnumerateAll(d)
length(e::EnumerateAll) = count(e.d)
function start(e::EnumerateAll)
    dst, k, vs, vst = s
    while done(vs, vst)
    end
    v, vst = next(vs, vst)
    ((k, v), (dst, k, vs, vst))
end
mutable struct SortedDict{K, D, Ord <: Ordering} <: AbstractDict{K,D}
    bt::BalancedTree23{K,D,Ord}
    function SortedDict{K,D,Ord}(o::Ord, kv) where {K, D, Ord <: Ordering}
        s = new{K,D,Ord}(BalancedTree23{K,D,Ord}(o))
        if eltype(kv) <: Pair
            for p in kv
                s[p.first] = p.second
            end
        else
            for (k,v) in kv
                s[k] = v
            end
        end
        return s
    end
end
SortedDict{K,D}(kv) where {K,D} = SortedDict{K,D}(Forward, kv)
function SortedDict{K,D}(o::Ord, kv) where {K,D,Ord<:Ordering}
end
function merge!(m::SortedDict{K,D,Ord},
                others::AbstractDict{K,D}...) where {K,D,Ord <: Ordering}
    for o in others
        mergetwo!(m,o)
    end
end
function merge(m::SortedDict{K,D,Ord},
               others::AbstractDict{K,D}...) where {K,D,Ord <: Ordering}
    result = packcopy(m)
end
similar(m::SortedDict{K,D,Ord}) where {K,D,Ord<:Ordering} =
    SortedDict{K,D,Ord}(orderobject(m))
isordered(::Type{T}) where {T<:SortedDict} = true
mutable struct SortedMultiDict{K, D, Ord <: Ordering}
    bt::BalancedTree23{K,D,Ord}
    function SortedMultiDict{K,D,Ord}(o::Ord, kv) where {K,D,Ord}
        smd = new{K,D,Ord}(BalancedTree23{K,D,Ord}(o))
        if eltype(kv) <: Pair
            # It's (possibly?) more efficient to access the first and second
            for (k,v) in kv
                insert!(smd, k, v)
            end
        end
        return smd
    end
end
SortedMultiDict() = SortedMultiDict{Any,Any,ForwardOrdering}(Forward)
function SortedMultiDict{K,D}(o::Ord, kv) where {K,D,Ord<:Ordering}
    try
        if not_iterator_of_pairs(kv)
            throw(ArgumentError("SortedMultiDict(kv): kv needs to be an iterator of tuples or pairs"))
        end
    end
end
SortedMultiDict(o1::Ordering, o2::Ordering) = throw(ArgumentError("SortedMultiDict with two parameters must be called with an Ordering and an interable of pairs"))
SortedMultiDict(kv, o::Ordering=Forward) = SortedMultiDict(o, kv)
function SortedMultiDict(o::Ordering, kv)
    try
        _sorted_multidict_with_eltype(o, kv, eltype(kv))
    catch e
        if not_iterator_of_pairs(kv)
        end
    end
end
@inline function insert!(m::SortedMultiDict{K,D,Ord}, k_, d_) where {K, D, Ord <: Ordering}
    while true
    end
end
@inline function haskey(m::SortedMultiDict, k_)
    while true
        if p1 == pastendsemitoken(m1)
        end
        if p2 == pastendsemitoken(m2)
        end
    end
end
const SDorAbstractDict = Union{AbstractDict,SortedMultiDict}
function mergetwo!(m::SortedMultiDict{K,D,Ord},
                   m2::SDorAbstractDict) where {K,D,Ord <: Ordering}
    for (k,v) in m2
    end
end
function packcopy(m::SortedMultiDict{K,D,Ord}) where {K,D,Ord <: Ordering}
    for (k,v) in m
    end
end
function merge!(m::SortedMultiDict{K,D,Ord},
                others::SDorAbstractDict...) where {K,D,Ord <: Ordering}
    for o in others
    end
    for (count,(k,v)) in enumerate(m)
        if count < l
        end
    end
end
similar(m::SortedMultiDict{K,D,Ord}) where {K,D,Ord<:Ordering} =
   SortedMultiDict{K,D}(orderobject(m))
mutable struct SortedSet{K, Ord <: Ordering}
end
@inline function insert!(m::SortedSet, k_)
    while true
        if p1 == pastendsemitoken(m1)
        end
    end
    for m2 in others
    end
end
function intersect2(m1::SortedSet{K, Ord}, m2::SortedSet{K, Ord}) where {K, Ord <: Ordering}
    while true
        if lt(ord,k1,k2)
        end
        if m1end && m2end
            if lt(ord,k1,k2)
            end
        end
    end
end
function setdiff(m1::SortedSet{K,Ord}, m2::SortedSet{K,Ord}) where {K, Ord <: Ordering}
    if !isequal(ord, orderobject(m2))
        if p1 == pastendsemitoken(m1)
            if lt(ord,k1,k2)
            end
        end
    end
    for p in iterable
        if i != pastendsemitoken(m1)
        end
    end
    for k in iterable
        if !in(k, m2)
        end
    end
    for k in m
    end
end
similar(m::SortedSet{K,Ord}) where {K,Ord<:Ordering} =
SortedSet{K,Ord}(orderobject(m))
const SDMContainer = Union{SortedDict, SortedMultiDict}
const SAContainer = Union{SDMContainer, SortedSet}
const Token = Tuple{SAContainer, IntSemiToken}
@inline function advance(ii::Token)
end
@inline function regress(ii::Token)
    m.bt.data[i.address] = KDRec{keytype(m),valtype(m)}(m.bt.data[i.address].parent,
                                                         convert(valtype(m),d_))
end
@inline function searchsortedfirst(m::SAContainer, k_)
end
@inline function searchsortedafter(m::SAContainer, k_)
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
const SDMIterableTypesBase = Union{SDMContainer,
                                   SDMIncludeLast}
const SSIterableTypesBase = Union{SortedSet,
                                  SSIncludeLast}
const SAIterableTypesBase = Union{SAContainer,
                                  AbstractIncludeLast}
struct SDMKeyIteration{T <: SDMIterableTypesBase}
end
struct SDMValIteration{T <: SDMIterableTypesBase}
end
struct SDMSemiTokenIteration{T <: SDMIterableTypesBase}
end
eltype(s::SDMSemiTokenIteration) = Tuple{IntSemiToken,
                                         valtype(extractcontainer(s.base))}
struct SSSemiTokenIteration{T <: SSIterableTypesBase}
end
eltype(s::SSSemiTokenIteration) = Tuple{IntSemiToken,
                                        eltype(extractcontainer(s.base))}
struct SDMSemiTokenKeyIteration{T <: SDMIterableTypesBase}
end
struct SAOnlySemiTokensIteration{T <: SAIterableTypesBase}
end
struct SDMSemiTokenValIteration{T <: SDMIterableTypesBase}
end
eltype(s::SDMSemiTokenValIteration) = Tuple{IntSemiToken,
                                 SAOnlySemiTokensIteration}
struct SAIterationState
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
@inline function next(u::SAOnlySemiTokensIteration, state::SAIterationState)
end
function sort!(d::OrderedDict; byvalue::Bool=false, args...)
    if d.ndel > 0
    end
    if byvalue
    end
end
    export
        isfull
mutable struct CircularBuffer{T} <: AbstractVector{T}
end
function Base.empty!(cb::CircularBuffer)
    for i in max(1, n-capacity(cb)+1):n
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
function PriorityQueue{K,V}(o::Ord, kv) where {K,V,Ord<:Ordering}
    try
    end
end
function percolate_down!(pq::PriorityQueue, i::Integer)
    @inbounds while (l = heapleft(i)) <= length(pq)
        if lt(pq.o, pq.xs[j].second, x.second)
        end
    end
    @inbounds while i > 1
        if lt(pq.o, x.second, pq.xs[j].second)
        end
    end
    @inbounds while i > 1
    end
    if haskey(pq, key)
        if lt(pq.o, oldvalue, value)
        end
    end
end
function enqueue!(pq::PriorityQueue{K,V}, pair::Pair{K,V}) where {K,V}
    if haskey(pq, key)
    end
    if !isempty(pq)
    end
end
function dequeue!(pq::PriorityQueue, key)
end
    function PriorityQueue(ks::AbstractVector{K},
                           o::Ordering=Forward) where {K,V}
        if length(ks) != length(vs)
        end
    end
end
