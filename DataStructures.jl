__precompile__()
module DataStructures
    import Base: <, <=, ==, length, isempty, start, next, done, delete!,
                 show, dump, empty!, getindex, setindex!, get, get!,
                 in, haskey, keys, merge, copy, cat,
                 push!, pop!, shift!, unshift!, insert!,
                 union!, delete!, similar, sizehint!,
                 isequal, hash,
                 map, reverse,
                 first, last, eltype, getkey, values, sum,
                 merge, merge!, lt, Ordering, ForwardOrdering, Forward,
                 ReverseOrdering, Reverse, Lt,
                 isless,
                 union, intersect, symdiff, setdiff, issubset,
                 find, searchsortedfirst, searchsortedlast, endof, in
    using Compat: uninitialized, Nothing, Cvoid, AbstractDict
    export Deque, Stack, Queue, CircularDeque
    export deque, enqueue!, dequeue!, dequeue_pair!, update!, reverse_iter
    export capacity, num_blocks, front, back, top, top_with_handle, sizehint!
    export Accumulator, counter, reset!, inc!, dec!
    export ClassifiedCollections
    export classified_lists, classified_sets, classified_counters
    export IntDisjointSets, DisjointSets, num_groups, find_root, in_same_set, root_union!
    export AbstractHeap, compare, extract_all!
    export BinaryHeap, binary_minheap, binary_maxheap, nlargest, nsmallest
    export MutableBinaryHeap, mutable_binary_minheap, mutable_binary_maxheap
    export heapify!, heapify, heappop!, heappush!, isheap
    export OrderedDict, OrderedSet
    export DefaultDict, DefaultOrderedDict
    export Trie, subtrie, keys_with_prefix, path
    export LinkedList, Nil, Cons, nil, cons, head, tail, list, filter, cat,
           reverse
    export SortedDict, SortedMultiDict, SortedSet
    export SDToken, SDSemiToken, SMDToken, SMDSemiToken
    export SetToken, SetSemiToken
    export startof
    export pastendsemitoken, beforestartsemitoken
    export searchsortedafter, searchequalrange
    export packcopy, packdeepcopy
    export exclusive, inclusive, semitokens
    export orderobject, ordtype, Lt, compare, onlysemitokens
    export MultiDict, enumerateall
    import Base: eachindex, keytype, valtype
function unquote(e::Expr)
    @assert e.head == :quote
    return e.args[1]
end
function unquote(e::QuoteNode)
    return e.value
end
macro delegate(source, targets)
    typename = esc(source.args[1])
    fieldname = unquote(source.args[2])
    funcnames = targets.args
    n = length(funcnames)
    fdefs = Vector{Any}(uninitialized, n)
    for i in 1:n
        funcname = esc(funcnames[i])
        fdefs[i] = quote
                     ($funcname)(a::($typename), args...) =
                       ($funcname)(a.$fieldname, args...)
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
    data::Vector{T}  # only data[front:back] is valid
    capa::Int
    front::Int
    back::Int
    prev::DequeBlock{T}  # ref to previous block
    next::DequeBlock{T}  # ref to next block
    function DequeBlock{T}(capa::Int, front::Int) where T
        data = Vector{T}(uninitialized, capa)
        blk = new{T}(data, capa, front, front-1)
        blk.prev = blk
        blk.next = blk
        blk
    end
end
rear_deque_block(ty::Type{T}, n::Int) where {T} = DequeBlock{T}(n, 1)
head_deque_block(ty::Type{T}, n::Int) where {T} = DequeBlock{T}(n, n+1)
capacity(blk::DequeBlock) = blk.capa
length(blk::DequeBlock) = blk.back - blk.front + 1
isempty(blk::DequeBlock) = blk.back < blk.front
ishead(blk::DequeBlock) = blk.prev === blk
isrear(blk::DequeBlock) =  blk.next === blk
function reset!(blk::DequeBlock{T}, front::Int) where T
    blk.front = front
    blk.back = front - 1
    blk.prev = blk
    blk.next = blk
end
function show(io::IO, blk::DequeBlock)  # avoids recursion into prev and next
    x = blk.data[blk.front:blk.back]
    print(io, "$(typeof(blk))(capa = $(blk.capa), front = $(blk.front), back = $(blk.back)): $x")
end
const DEFAULT_DEQUEUE_BLOCKSIZE = 1024
mutable struct Deque{T}
    nblocks::Int
    blksize::Int
    len::Int
    head::DequeBlock{T}
    rear::DequeBlock{T}
    function Deque{T}(blksize::Int) where T
        head = rear = rear_deque_block(T, blksize)
        new{T}(1, blksize, 0, head, rear)
    end
    Deque{T}() where {T} = Deque{T}(DEFAULT_DEQUEUE_BLOCKSIZE)
end
"""
    deque(T)
Create a deque of type `T`.
"""
deque(::Type{T}) where {T} = Deque{T}()
isempty(q::Deque) = q.len == 0
length(q::Deque) = q.len
num_blocks(q::Deque) = q.nblocks
"""
    front(q::Deque)
Returns the first element of the deque `q`.
"""
function front(q::Deque)
    isempty(q) && throw(ArgumentError("Deque must be non-empty"))
    blk = q.head
    blk.data[blk.front]
end
"""
    back(q::Deque)
Returns the last element of the deque `q`.
"""
function back(q::Deque)
    isempty(q) && throw(ArgumentError("Deque must be non-empty"))
    blk = q.rear
    blk.data[blk.back]
end
struct DequeIterator{T}
    q::Deque
end
start(qi::DequeIterator{T}) where {T} = (qi.q.head, qi.q.head.front)
function next(qi::DequeIterator{T}, s) where T
    cb = s[1]
    i = s[2]
    x = cb.data[i]
    i += 1
    if i > cb.back && !isrear(cb)
        cb = cb.next
        i = 1
    end
    (x, (cb, i))
end
done(q::DequeIterator{T}, s) where {T} = (s[2] > s[1].back)
struct ReverseDequeIterator{T}
    q::Deque
end
start(qi::ReverseDequeIterator{T}) where {T} = (qi.q.rear, qi.q.rear.back)
done(qi::ReverseDequeIterator{T}, s) where {T} = (s[2] < s[1].front)
function next(qi::ReverseDequeIterator{T}, s) where T
    cb = s[1]
    i = s[2]
    x = cb.data[i]
    i -= 1
    # If we're past the beginning of a block, go to the previous one
    if i < cb.front && !ishead(cb)
        cb = cb.prev
        i = cb.back
    end
    (x, (cb, i))
end
reverse_iter(q::Deque{T}) where {T} = ReverseDequeIterator{T}(q)
start(q::Deque{T}) where {T} = start(DequeIterator{T}(q))
next(q::Deque{T}, s) where {T} = next(DequeIterator{T}(q), s)
done(q::Deque{T}, s) where {T} = done(DequeIterator{T}(q), s)
Base.length(qi::DequeIterator{T}) where {T} = qi.q.len
Base.length(qi::ReverseDequeIterator{T}) where {T} = qi.q.len
Base.collect(q::Deque{T}) where {T} = T[x for x in q]
function show(io::IO, q::Deque)
    print(io, "Deque [$(collect(q))]")
end
function dump(io::IO, q::Deque)
    println(io, "Deque (length = $(q.len), nblocks = $(q.nblocks))")
    cb::DequeBlock = q.head
    i = 1
    while true
        print(io, "block $i [$(cb.front):$(cb.back)] ==> ")
        for j = cb.front : cb.back
            print(io, cb.data[j])
            print(io, ' ')
        end
        println(io)
        cb_next::DequeBlock = cb.next
        if cb !== cb_next
            cb = cb_next
            i += 1
        else
            break
        end
    end
end
function empty!(q::Deque{T}) where T
    # release all blocks except the head
    if q.nblocks > 1
        cb::DequeBlock{T} = q.rear
        while cb != q.head
            empty!(cb.data)
            cb = cb.prev
        end
    end
    # clean the head block (but retain the block itself)
    reset!(q.head, 1)
    # reset queue fields
    q.nblocks = 1
    q.len = 0
    q.rear = q.head
    q
end
function push!(q::Deque{T}, x) where T  # push back
    rear = q.rear
    if isempty(rear)
        rear.front = 1
        rear.back = 0
    end
    if rear.back < rear.capa
        @inbounds rear.data[rear.back += 1] = convert(T, x)
    else
        new_rear = rear_deque_block(T, q.blksize)
        new_rear.back = 1
        new_rear.data[1] = convert(T, x)
        new_rear.prev = rear
        q.rear = rear.next = new_rear
        q.nblocks += 1
    end
    q.len += 1
    q
end
function unshift!(q::Deque{T}, x) where T   # push front
    head = q.head
    if isempty(head)
        n = head.capa
        head.front = n + 1
        head.back = n
    end
    if head.front > 1
        @inbounds head.data[head.front -= 1] = convert(T, x)
    else
        n::Int = q.blksize
        new_head = head_deque_block(T, n)
        new_head.front = n
        new_head.data[n] = convert(T, x)
        new_head.next = head
        q.head = head.prev = new_head
        q.nblocks += 1
    end
    q.len += 1
    q
end
function pop!(q::Deque{T}) where T   # pop back
    isempty(q) && throw(ArgumentError("Deque must be non-empty"))
    rear = q.rear
    @assert rear.back >= rear.front
    @inbounds x = rear.data[rear.back]
    rear.back -= 1
    if rear.back < rear.front
        if q.nblocks > 1
            # release and detach the rear block
            empty!(rear.data)
            q.rear = rear.prev::DequeBlock{T}
            q.rear.next = q.rear
            q.nblocks -= 1
        end
    end
    q.len -= 1
    x
end
function shift!(q::Deque{T}) where T  # pop front
    isempty(q) && throw(ArgumentError("Deque must be non-empty"))
    head = q.head
    @assert head.back >= head.front
    @inbounds x = head.data[head.front]
    head.front += 1
    if head.back < head.front
        if q.nblocks > 1
            # release and detach the head block
            empty!(head.data)
            q.head = head.next::DequeBlock{T}
            q.head.prev = q.head
            q.nblocks -= 1
        end
    end
    q.len -= 1
    x
end
const _deque_hashseed = UInt === UInt64 ? 0x950aa17a3246be82 : 0x4f26f881
function hash(x::Deque, h::UInt)
    h += _deque_hashseed
    for (i, x) in enumerate(x)
        h += i * hash(x)
    end
    h
end
function ==(x::Deque, y::Deque)
    length(x) != length(y) && return false
    for (i, j) in zip(x, y)
        i == j || return false
    end
    true
end
"""
    CircularDeque{T}(n)
Create a double-ended queue of maximum capacity `n`, implemented as a circular buffer. The element type is `T`.
"""
mutable struct CircularDeque{T}
    buffer::Vector{T}
    capacity::Int
    n::Int
    first::Int
    last::Int
end
CircularDeque{T}(n::Int) where {T} = CircularDeque(Vector{T}(uninitialized, n), n, 0, 1, n)
Base.length(D::CircularDeque) = D.n
Base.eltype(::Type{CircularDeque{T}}) where {T} = T
capacity(D::CircularDeque) = D.capacity
function Base.empty!(D::CircularDeque)
    D.n = 0
    D.first = 1
    D.last = D.capacity
    D
end
Base.isempty(D::CircularDeque) = D.n == 0
@inline function front(D::CircularDeque)
    @boundscheck D.n > 0 || throw(BoundsError())
    D.buffer[D.first]
end
@inline function back(D::CircularDeque)
    @boundscheck D.n > 0 || throw(BoundsError())
    D.buffer[D.last]
end
@inline function Base.push!(D::CircularDeque, v)
    @boundscheck D.n < D.capacity || throw(BoundsError()) # prevent overflow
    D.n += 1
    tmp = D.last+1
    D.last = ifelse(tmp > D.capacity, 1, tmp)  # wraparound
    @inbounds D.buffer[D.last] = v
    D
end
@inline function Base.pop!(D::CircularDeque)
    v = back(D)
    D.n -= 1
    tmp = D.last - 1
    D.last = ifelse(tmp < 1, D.capacity, tmp)
    v
end
@inline function Base.unshift!(D::CircularDeque, v)
    @boundscheck D.n < D.capacity || throw(BoundsError())
    D.n += 1
    tmp = D.first - 1
    D.first = ifelse(tmp < 1, D.capacity, tmp)
    @inbounds D.buffer[D.first] = v
    D
end
@inline function Base.shift!(D::CircularDeque)
    v = front(D)
    D.n -= 1
    tmp = D.first + 1
    D.first = ifelse(tmp > D.capacity, 1, tmp)
    v
end
@inline function _unsafe_getindex(D::CircularDeque, i::Integer)
    j = D.first + i - 1
    if j > D.capacity
        j -= D.capacity
    end
    @inbounds ret = D.buffer[j]
    return ret
end
@inline function Base.getindex(D::CircularDeque, i::Integer)
    @boundscheck 1 <= i <= D.n || throw(BoundsError())
    return _unsafe_getindex(D, i)
end
@inline Base.start(d::CircularDeque) = 1
@inline Base.next(d::CircularDeque, i) = (_unsafe_getindex(d, i), i+1)
@inline Base.done(d::CircularDeque, i) = i == d.n + 1
function Base.show(io::IO, D::CircularDeque{T}) where T
    print(io, "CircularDeque{$T}([")
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
Stack(ty::Type{T}, blksize::Integer) where {T} = Stack(Deque{T}(blksize))
isempty(s::Stack) = isempty(s.store)
length(s::Stack) = length(s.store)
top(s::Stack) = back(s.store)
function push!(s::Stack, x)
    push!(s.store, x)
    s
end
pop!(s::Stack) = pop!(s.store)
start(st::Stack) = start(reverse_iter(st.store))
next(st::Stack, s) = next(reverse_iter(st.store), s)
done(st::Stack, s) = done(reverse_iter(st.store), s)
reverse_iter(s::Stack{T}) where {T} = DequeIterator{T}(s.store)
mutable struct Queue{T}
    store::Deque{T}
end
"""
    Queue(T[, blksize::Integer=1024])
Create a `Queue` object containing elements of type `T`.
"""
Queue(ty::Type{T}) where {T} = Queue(Deque{T}())
Queue(ty::Type{T}, blksize::Integer) where {T} = Queue(Deque{T}(blksize))
isempty(s::Queue) = isempty(s.store)
length(s::Queue) = length(s.store)
front(s::Queue) = front(s.store)
back(s::Queue) = back(s.store)
"""
    enqueue!(s::Queue, x)
Inserts the value `x` to the end of the queue `s`.
"""
function enqueue!(s::Queue, x)
    push!(s.store, x)
    s
end
"""
    dequeue!(s::Queue)
Removes an element from the front of the queue `s` and returns it.
"""
dequeue!(s::Queue) = shift!(s.store)
start(q::Queue) = start(q.store)
next(q::Queue, s) = next(q.store, s)
done(q::Queue, s) = done(q.store, s)
reverse_iter(q::Queue) = reverse_iter(q.store)
struct Accumulator{T, V<:Number} <: AbstractDict{T,V}
    map::Dict{T,V}
end
Accumulator(::Type{T}, ::Type{V}) where {T,V<:Number} = Accumulator{T,V}(Dict{T,V}())
counter(T::Type) = Accumulator(T,Int)
counter(dct::Dict{T,V}) where {T,V<:Integer} = Accumulator{T,V}(copy(dct))
"""
    counter(seq)
Returns an `Accumulator` object containing the elements from `seq`.
"""
function counter(seq)
    ct = counter(eltype_for_accumulator(seq))
    for x in seq
        inc!(ct, x)
    end
    return ct
end
eltype_for_accumulator(seq::T) where T = eltype(T)
function eltype_for_accumulator(seq::T) where {T<:Base.Generator}
    @static if VERSION < v"0.7.0-DEV.2104"
        Base._default_eltype(T)
    else
        Base.@default_eltype(T)
    end
end
copy(ct::Accumulator) = Accumulator(copy(ct.map))
length(a::Accumulator) = length(a.map)
get(ct::Accumulator, x, default) = get(ct.map, x, default)
getindex(ct::Accumulator{T,V}, x) where {T,V} = get(ct.map, x, zero(V))
setindex!(ct::Accumulator, x, v) = setindex!(ct.map, x, v)
haskey(ct::Accumulator, x) = haskey(ct.map, x)
keys(ct::Accumulator) = keys(ct.map)
values(ct::Accumulator) = values(ct.map)
sum(ct::Accumulator) = sum(values(ct.map))
start(ct::Accumulator) = start(ct.map)
next(ct::Accumulator, state) = next(ct.map, state)
done(ct::Accumulator, state) = done(ct.map, state)
"""
    inc!(ct, x, [v=1])
Increments the count for `x` by `v` (defaulting to one)
"""
inc!(ct::Accumulator, x, a::Number) = (ct[x] += a)
inc!(ct::Accumulator{T,V}, x) where {T,V} = inc!(ct, x, one(V))
push!(ct::Accumulator, x) = inc!(ct, x)
push!(ct::Accumulator, x, a::Number) = inc!(ct, x, a)
push!(ct::Accumulator, x::Pair)  = inc!(ct, x)
"""
    dec!(ct, x, [v=1])
Decrements the count for `x` by `v` (defaulting to one)
"""
dec!(ct::Accumulator, x, a::Number) = (ct[x] -= a)
dec!(ct::Accumulator{T,V}, x) where {T,V} = dec!(ct, x, one(V))
"""
    merge!(ct1, others...)
Merges the other counters into `ctl`,
summing the counts for all elements.
"""
function merge!(ct::Accumulator, other::Accumulator)
    for (x, v) in other
        inc!(ct, x, v)
    end
    ct
end
function merge!(ct1::Accumulator, others::Accumulator...)
    for ct in others
        merge!(ct1,ct)
    end
    return ct1
end
"""
     merge(counters...)
Creates a new counter with total counts equal to the sum of the counts in the counters given as arguments.
See also merge!
"""
function merge(ct1::Accumulator, others::Accumulator...)
    ct = copy(ct1)
    merge!(ct,others...)
end
"""
    reset!(ct::Accumulator, x)
Resets the count of `x` to zero.
Returns its former count.
"""
reset!(ct::Accumulator, x) = pop!(ct.map, x)
@deprecate pop!(ct::Accumulator, x) reset!(ct, x)
@deprecate push!(ct1::Accumulator, ct2::Accumulator) merge!(ct1,ct2)
mutable struct ClassifiedCollections{K, Collection}
    map::Dict{K, Collection}
end
ClassifiedCollections(K::Type, C::Type) = ClassifiedCollections{K, C}(Dict{K,C}())
classified_lists(K::Type, V::Type) = ClassifiedCollections(K, Vector{V})
classified_sets(K::Type, V::Type) = ClassifiedCollections(K, Set{V})
classified_counters(K::Type, T::Type) = ClassifiedCollections(K, Accumulator{T, Int})
_create_empty(::Type{Vector{T}}) where {T} = Vector{T}()
_create_empty(::Type{Set{T}}) where {T} = Set{T}()
_create_empty(::Type{Accumulator{T,V}}) where {T,V} = Accumulator(T, V)
copy(cc::ClassifiedCollections{K, C}) where {K, C} = ClassifiedCollections{K, C}(copy(cc.map))
length(cc::ClassifiedCollections) = length(cc.map)
getindex(cc::ClassifiedCollections{T,C}, x::T) where {T,C} = cc.map[x]
haskey(cc::ClassifiedCollections{T,C}, x::T) where {T,C} = haskey(cc.map, x)
keys(cc::ClassifiedCollections) = keys(cc.map)
start(cc::ClassifiedCollections) = start(cc.map)
next(cc::ClassifiedCollections, state) = next(cc.map, state)
done(cc::ClassifiedCollections, state) = done(cc.map, state)
function push!(cc::ClassifiedCollections{K, C}, key::K, e) where {K, C}
    c = get(cc.map, key, nothing)
    if c === nothing
        c = _create_empty(C)
        cc.map[key] = c
    end
    push!(c, e)
end
pop!(cc::ClassifiedCollections{K}, key::K) where {K} = pop!(cc.map, key)
mutable struct IntDisjointSets
    parents::Vector{Int}
    ranks::Vector{Int}
    ngroups::Int
    # creates a disjoint set comprised of n singletons
    IntDisjointSets(n::Integer) = new(collect(1:n), zeros(Int, n), n)
end
length(s::IntDisjointSets) = length(s.parents)
num_groups(s::IntDisjointSets) = s.ngroups
function find_root_impl!(parents::Array{Int}, x::Integer)
    p = parents[x]
    @inbounds if parents[p] != p
        parents[x] = p = _find_root_impl!(parents, p)
    end
    p
end
function _find_root_impl!(parents::Array{Int}, x::Integer)
    @inbounds p = parents[x]
    @inbounds if parents[p] != p
        parents[x] = p = _find_root_impl!(parents, p)
    end
    p
end
find_root(s::IntDisjointSets, x::Integer) = find_root_impl!(s.parents, x)
"""
    in_same_set(s::IntDisjointSets, x::Integer, y::Integer)
Returns `true` if `x` and `y` belong to the same subset in `s` and `false` otherwise.
"""
in_same_set(s::IntDisjointSets, x::Integer, y::Integer) = find_root(s, x) == find_root(s, y)
function union!(s::IntDisjointSets, x::Integer, y::Integer)
    parents = s.parents
    xroot = find_root_impl!(parents, x)
    yroot = find_root_impl!(parents, y)
    xroot != yroot ?  root_union!(s, xroot, yroot) : xroot
end
function root_union!(s::IntDisjointSets, x::Integer, y::Integer)
    parents = s.parents
    rks = s.ranks
    @inbounds xrank = rks[x]
    @inbounds yrank = rks[y]
    if xrank < yrank
        x, y = y, x
    elseif xrank == yrank
        rks[x] += 1
    end
    @inbounds parents[y] = x
    @inbounds s.ngroups -= 1
    x
end
function push!(s::IntDisjointSets)
    x = length(s) + 1
    push!(s.parents, x)
    push!(s.ranks, 0)
    s.ngroups += 1
    return x
end
mutable struct DisjointSets{T}
    intmap::Dict{T,Int}
    revmap::Vector{T}
    internal::IntDisjointSets
    function DisjointSets{T}(xs) where T    # xs must be iterable
        imap = Dict{T,Int}()
        rmap = Vector{T}()
        n = length(xs)
        sizehint!(imap, n)
        sizehint!(rmap, n)
        id = 0
        for x in xs
            imap[x] = (id += 1)
            push!(rmap,x)
        end
        new{T}(imap, rmap, IntDisjointSets(n))
    end
end
length(s::DisjointSets) = length(s.internal)
num_groups(s::DisjointSets) = num_groups(s.internal)
"""
    find_root{T}(s::DisjointSets{T}, x::T)
Finds the root element of the subset in `s` which has the element `x` as a member.
"""
find_root(s::DisjointSets{T}, x::T) where {T} = s.revmap[find_root(s.internal, s.intmap[x])]
in_same_set(s::DisjointSets{T}, x::T, y::T) where {T} = in_same_set(s.internal, s.intmap[x], s.intmap[y])
union!(s::DisjointSets{T}, x::T, y::T) where {T} = s.revmap[union!(s.internal, s.intmap[x], s.intmap[y])]
root_union!(s::DisjointSets{T}, x::T, y::T) where {T} = s.revmap[root_union!(s.internal, s.intmap[x], s.intmap[y])]
function push!(s::DisjointSets{T}, x::T) where T
    id = push!(s.internal)
    s.intmap[x] = id
    push!(s.revmap,x) # Note, this assumes invariant: length(s.revmap) == id
    x
end
abstract type AbstractHeap{VT} end
abstract type AbstractMutableHeap{VT,HT} <: AbstractHeap{VT} end
struct LessThan
end
struct GreaterThan
end
compare(c::LessThan, x, y) = x < y
compare(c::GreaterThan, x, y) = x > y
function _heap_bubble_up!(comp::Comp, valtree::Array{T}, i::Int) where {Comp,T}
    i0::Int = i
    @inbounds v = valtree[i]
    while i > 1  # nd is not root
        p = i >> 1
        @inbounds vp = valtree[p]
        if compare(comp, v, vp)
            # move parent downward
            @inbounds valtree[i] = vp
            i = p
        else
            break
        end
    end
    if i != i0
        @inbounds valtree[i] = v
    end
end
function _heap_bubble_down!(comp::Comp, valtree::Array{T}, i::Int) where {Comp,T}
    @inbounds v::T = valtree[i]
    swapped = true
    n = length(valtree)
    last_parent = n >> 1
    while swapped && i <= last_parent
        lc = i << 1
        if lc < n   # contains both left and right children
            rc = lc + 1
            @inbounds lv = valtree[lc]
            @inbounds rv = valtree[rc]
            if compare(comp, rv, lv)
                if compare(comp, rv, v)
                    @inbounds valtree[i] = rv
                    i = rc
                else
                    swapped = false
                end
            else
                if compare(comp, lv, v)
                    @inbounds valtree[i] = lv
                    i = lc
                else
                    swapped = false
                end
            end
        else        # contains only left child
            @inbounds lv = valtree[lc]
            if compare(comp, lv, v)
                @inbounds valtree[i] = lv
                i = lc
            else
                swapped = false
            end
        end
    end
    valtree[i] = v
end
function _binary_heap_pop!(comp::Comp, valtree::Array{T}) where {Comp,T}
    # extract root
    v = valtree[1]
    if length(valtree) == 1
        empty!(valtree)
    else
        valtree[1] = pop!(valtree)
        if length(valtree) > 1
            _heap_bubble_down!(comp, valtree, 1)
        end
    end
    v
end
function _make_binary_heap(comp::Comp, ty::Type{T}, xs) where {Comp,T}
    n = length(xs)
    valtree = copy(xs)
    for i = 2 : n
        _heap_bubble_up!(comp, valtree, i)
    end
    valtree
end
mutable struct BinaryHeap{T,Comp} <: AbstractHeap{T}
    comparer::Comp
    valtree::Vector{T}
    function BinaryHeap{T,Comp}(comp::Comp) where {T,Comp}
        new{T,Comp}(comp, Vector{T}())
    end
    function BinaryHeap{T,Comp}(comp::Comp, xs) where {T,Comp}  # xs is an iterable collection of values
        valtree = _make_binary_heap(comp, T, xs)
        new{T,Comp}(comp, valtree)
    end
end
function binary_minheap(ty::Type{T}) where T
    BinaryHeap{T,LessThan}(LessThan())
end
binary_maxheap(ty::Type{T}) where {T} = BinaryHeap{T,GreaterThan}(GreaterThan())
binary_minheap(xs::AbstractVector{T}) where {T} = BinaryHeap{T,LessThan}(LessThan(), xs)
binary_maxheap(xs::AbstractVector{T}) where {T} = BinaryHeap{T,GreaterThan}(GreaterThan(), xs)
length(h::BinaryHeap) = length(h.valtree)
isempty(h::BinaryHeap) = isempty(h.valtree)
function push!(h::BinaryHeap{T}, v::T) where T
    valtree = h.valtree
    push!(valtree, v)
    _heap_bubble_up!(h.comparer, valtree, length(valtree))
    h
end
"""
    top(h::BinaryHeap)
Returns the element at the top of the heap `h`.
"""
@inline top(h::BinaryHeap) = h.valtree[1]
pop!(h::BinaryHeap{T}) where {T} = _binary_heap_pop!(h.comparer, h.valtree)
struct MutableBinaryHeapNode{T}
    value::T
    handle::Int
end
function _heap_bubble_up!(comp::Comp,
    nodes::Vector{MutableBinaryHeapNode{T}}, nodemap::Vector{Int}, nd_id::Int) where {Comp, T}
    @inbounds nd = nodes[nd_id]
    v::T = nd.value
    swapped = true  # whether swap happens at last step
    i = nd_id
    while swapped && i > 1  # nd is not root
        p = i >> 1
        @inbounds nd_p = nodes[p]
        if compare(comp, v, nd_p.value)
            # move parent downward
            @inbounds nodes[i] = nd_p
            @inbounds nodemap[nd_p.handle] = i
            i = p
        else
            swapped = false
        end
    end
    if i != nd_id
        nodes[i] = nd
        nodemap[nd.handle] = i
    end
end
function _heap_bubble_down!(comp::Comp,
    nodes::Vector{MutableBinaryHeapNode{T}}, nodemap::Vector{Int}, nd_id::Int) where {Comp, T}
    @inbounds nd = nodes[nd_id]
    v::T = nd.value
    n = length(nodes)
    last_parent = n >> 1
    swapped = true
    i = nd_id
    while swapped && i <= last_parent
        il = i << 1
        if il < n   # contains both left and right children
            ir = il + 1
            # determine the better child
            @inbounds nd_l = nodes[il]
            @inbounds nd_r = nodes[ir]
            if compare(comp, nd_r.value, nd_l.value)
                # consider right child
                if compare(comp, nd_r.value, v)
                    @inbounds nodes[i] = nd_r
                    @inbounds nodemap[nd_r.handle] = i
                    i = ir
                else
                    swapped = false
                end
            else
                # consider left child
                if compare(comp, nd_l.value, v)
                    @inbounds nodes[i] = nd_l
                    @inbounds nodemap[nd_l.handle] = i
                    i = il
                else
                    swapped = false
                end
            end
        else  # contains only left child
            nd_l = nodes[il]
            if compare(comp, nd_l.value, v)
                @inbounds nodes[i] = nd_l
                @inbounds nodemap[nd_l.handle] = i
                i = il
            else
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
    rt = nodes[1]
    v = rt.value
    @inbounds nodemap[rt.handle] = 0
    if length(nodes) == 1
        # clear
        empty!(nodes)
    else
        # place last node to root
        @inbounds nodes[1] = new_rt = pop!(nodes)
        @inbounds nodemap[new_rt.handle] = 1
        if length(nodes) > 1
            _heap_bubble_down!(comp, nodes, nodemap, 1)
        end
    end
    v
end
function _make_mutable_binary_heap(comp::Comp, ty::Type{T}, values) where {Comp,T}
    # make a static binary index tree from a list of values
    n = length(values)
    nodes = Vector{MutableBinaryHeapNode{T}}(uninitialized, n)
    nodemap = Vector{Int}(uninitialized, n)
    i::Int = 0
    for v in values
        i += 1
        @inbounds nodes[i] = MutableBinaryHeapNode{T}(v, i)
        @inbounds nodemap[i] = i
    end
    for i = 1 : n
        _heap_bubble_up!(comp, nodes, nodemap, i)
    end
    return nodes, nodemap
end
mutable struct MutableBinaryHeap{VT, Comp} <: AbstractMutableHeap{VT,Int}
    comparer::Comp
    nodes::Vector{MutableBinaryHeapNode{VT}}
    node_map::Vector{Int}
    function MutableBinaryHeap{VT, Comp}(comp::Comp) where {VT, Comp}
        nodes = Vector{MutableBinaryHeapNode{VT}}()
        node_map = Vector{Int}()
        new{VT, Comp}(comp, nodes, node_map)
    end
    function MutableBinaryHeap{VT, Comp}(comp::Comp, xs) where {VT, Comp}  # xs is an iterable collection of values
        nodes, node_map = _make_mutable_binary_heap(comp, VT, xs)
        new{VT, Comp}(comp, nodes, node_map)
    end
end
mutable_binary_minheap(ty::Type{T}) where {T} = MutableBinaryHeap{T,LessThan}(LessThan())
mutable_binary_maxheap(ty::Type{T}) where {T} = MutableBinaryHeap{T,GreaterThan}(GreaterThan())
mutable_binary_minheap(xs::AbstractVector{T}) where {T} = MutableBinaryHeap{T,LessThan}(LessThan(), xs)
mutable_binary_maxheap(xs::AbstractVector{T}) where {T} = MutableBinaryHeap{T,GreaterThan}(GreaterThan(), xs)
function show(io::IO, h::MutableBinaryHeap)
    print(io, "MutableBinaryHeap(")
    nodes = h.nodes
    n = length(nodes)
    if n > 0
        print(io, string(nodes[1].value))
        for i = 2 : n
            print(io, ", $(nodes[i].value)")
        end
    end
    print(io, ")")
end
length(h::MutableBinaryHeap) = length(h.nodes)
isempty(h::MutableBinaryHeap) = isempty(h.nodes)
function push!(h::MutableBinaryHeap{T}, v::T) where T
    nodes = h.nodes
    nodemap = h.node_map
    i = length(nodemap) + 1
    nd_id = length(nodes) + 1
    push!(nodes, MutableBinaryHeapNode(v, i))
    push!(nodemap, nd_id)
    _heap_bubble_up!(h.comparer, nodes, nodemap, nd_id)
    i
end
@inline top(h::MutableBinaryHeap) = h.nodes[1].value
"""
    top_with_handle(h::MutableBinaryHeap)
Returns the element at the top of the heap `h` and its handle.
"""
function top_with_handle(h::MutableBinaryHeap)
    el = h.nodes[1]
    return el.value, el.handle
end
pop!(h::MutableBinaryHeap{T}) where {T} = _binary_heap_pop!(h.comparer, h.nodes, h.node_map)
"""
    update!{T}(h::MutableBinaryHeap{T}, i::Int, v::T)
Replace the element at index `i` in heap `h` with `v`.
This is equivalent to `h[i]=v`.
"""
function update!(h::MutableBinaryHeap{T}, i::Int, v::T) where T
    nodes = h.nodes
    nodemap = h.node_map
    comp = h.comparer
    nd_id = nodemap[i]
    v0 = nodes[nd_id].value
    nodes[nd_id] = MutableBinaryHeapNode(v, i)
    if compare(comp, v, v0)
        _heap_bubble_up!(comp, nodes, nodemap, nd_id)
    else
        _heap_bubble_down!(comp, nodes, nodemap, nd_id)
    end
end
setindex!(h::MutableBinaryHeap, v, i::Int) = update!(h, i, v)
getindex(h::MutableBinaryHeap, i::Int) = h.nodes[h.node_map[i]].value
import Base.Order: Forward, Ordering, lt
heapleft(i::Integer) = 2i
heapright(i::Integer) = 2i + 1
heapparent(i::Integer) = div(i, 2)
function percolate_down!(xs::AbstractArray, i::Integer, x=xs[i], o::Ordering=Forward, len::Integer=length(xs))
    @inbounds while (l = heapleft(i)) <= len
        r = heapright(i)
        j = r > len || lt(o, xs[l], xs[r]) ? l : r
        if lt(o, xs[j], x)
            xs[i] = xs[j]
            i = j
        else
            break
        end
    end
    xs[i] = x
end
percolate_down!(xs::AbstractArray, i::Integer, o::Ordering, len::Integer=length(xs)) = percolate_down!(xs, i, xs[i], o, len)
function percolate_up!(xs::AbstractArray, i::Integer, x=xs[i], o::Ordering=Forward)
    @inbounds while (j = heapparent(i)) >= 1
        if lt(o, x, xs[j])
            xs[i] = xs[j]
            i = j
        else
            break
        end
    end
    xs[i] = x
end
percolate_up!(xs::AbstractArray{T}, i::Integer, o::Ordering) where {T} = percolate_up!(xs, i, xs[i], o)
"""
    heappop!(v, [ord])
Given a binary heap-ordered array, remove and return the lowest ordered element.
For efficiency, this function does not check that the array is indeed heap-ordered.
"""
function heappop!(xs::AbstractArray, o::Ordering=Forward)
    x = xs[1]
    y = pop!(xs)
    if !isempty(xs)
        percolate_down!(xs, 1, y, o)
    end
    x
end
"""
    heappush!(v, x, [ord])
Given a binary heap-ordered array, push a new element `x`, preserving the heap property.
For efficiency, this function does not check that the array is indeed heap-ordered.
"""
function heappush!(xs::AbstractArray, x, o::Ordering=Forward)
    push!(xs, x)
    percolate_up!(xs, length(xs), x, o)
    xs
end
"""
    heapify!(v, ord::Ordering=Forward)
In-place [`heapify`](@ref).
"""
function heapify!(xs::AbstractArray, o::Ordering=Forward)
    for i in heapparent(length(xs)):-1:1
        percolate_down!(xs, i, o)
    end
    xs
end
"""
    heapify(v, ord::Ordering=Forward)
Returns a new vector in binary heap order, optionally using the given ordering.
```jldoctest
julia> a = [1,3,4,5,2];
julia> heapify(a)
5-element Array{Int64,1}:
 1
 2
 4
 5
 3
julia> heapify(a, Base.Order.Reverse)
5-element Array{Int64,1}:
 5
 3
 4
 1
 2
```
"""
heapify(xs::AbstractArray, o::Ordering=Forward) = heapify!(copy!(similar(xs), xs), o)
"""
    isheap(v, ord::Ordering=Forward)
Return `true` if an array is heap-ordered according to the given order.
```jldoctest
julia> a = [1,2,3]
3-element Array{Int64,1}:
 1
 2
 3
julia> isheap(a,Base.Order.Forward)
true
julia> isheap(a,Base.Order.Reverse)
false
```
"""
function isheap(xs::AbstractArray, o::Ordering=Forward)
    for i in 1:div(length(xs), 2)
        if lt(o, xs[heapleft(i)], xs[i]) ||
           (heapright(i) <= length(xs) && lt(o, xs[heapright(i)], xs[i]))
            return false
        end
    end
    true
end
function extract_all!(h::AbstractHeap{VT}) where VT
    n = length(h)
    r = Vector{VT}(uninitialized, n)
    for i = 1 : n
        r[i] = pop!(h)
    end
    r
end
function extract_all_rev!(h::AbstractHeap{VT}) where VT
    n = length(h)
    r = Vector{VT}(uninitialized, n)
    for i = 1 : n
        r[n + 1 - i] = pop!(h)
    end
    r
end
function nextreme(comp::Comp, n::Int, arr::AbstractVector{T}) where {T, Comp}
    if n <= 0
        return T[] # sort(arr)[1:n] returns [] for n <= 0
    elseif n >= length(arr)
        return sort(arr, lt = (x, y) -> compare(comp, y, x))
    end
    buffer = BinaryHeap{T,Comp}(comp)
    for i = 1 : n
        @inbounds xi = arr[i]
        push!(buffer, xi)
    end
    for i = n + 1 : length(arr)
        @inbounds xi = arr[i]
        if compare(comp, top(buffer), xi)
            # This could use a pushpop method
            pop!(buffer)
            push!(buffer, xi)
        end
    end
    return extract_all_rev!(buffer)
end
@doc """
Returns the `n` largest elements of `arr`.
Equivalent to `sort(arr, lt = >)[1:min(n, end)]`
""" ->
function nlargest(n::Int, arr::AbstractVector{T}) where T
    return nextreme(LessThan(), n, arr)
end
@doc """
Returns the `n` smallest elements of `arr`.
Equivalent to `sort(arr, lt = <)[1:min(n, end)]`
""" ->
function nsmallest(n::Int, arr::AbstractVector{T}) where T
    return nextreme(GreaterThan(), n, arr)
end
_tablesz(x::Integer) = x < 16 ? 16 : one(x)<<((sizeof(x)<<3)-leading_zeros(x-1))
hashindex(key, sz) = (reinterpret(Int,(hash(key))) & (sz-1)) + 1
function not_iterator_of_pairs(kv)
    return any(x->isempty(methodswith(typeof(kv), x, true)),
               [start, next, done]) ||
           any(x->!isa(x, Union{Tuple,Pair}), kv)
end
import Base: haskey, get, get!, getkey, delete!, push!, pop!, empty!,
             setindex!, getindex, length, isempty, start,
             next, done, keys, values, setdiff, setdiff!,
             union, union!, intersect, filter, filter!,
             hash, eltype, ValueIterator, convert, copy,
             merge
"""
    OrderedDict
`OrderedDict`s are  simply dictionaries  whose entries  have a  particular order.  The order
refers to insertion order, which allows deterministic iteration over the dictionary or set.
"""
mutable struct OrderedDict{K,V} <: AbstractDict{K,V}
    slots::Array{Int32,1}
    keys::Array{K,1}
    vals::Array{V,1}
    ndel::Int
    dirty::Bool
    function OrderedDict{K,V}() where {K,V}
        new{K,V}(zeros(Int32,16), Vector{K}(), Vector{V}(), 0, false)
    end
    function OrderedDict{K,V}(kv) where {K,V}
        h = OrderedDict{K,V}()
        for (k,v) in kv
            h[k] = v
        end
        return h
    end
    OrderedDict{K,V}(p::Pair) where {K,V} = setindex!(OrderedDict{K,V}(), p.second, p.first)
    function OrderedDict{K,V}(ps::Pair...) where {K,V}
        h = OrderedDict{K,V}()
        sizehint!(h, length(ps))
        for p in ps
            h[p.first] = p.second
        end
        return h
    end
    function OrderedDict{K,V}(d::OrderedDict{K,V}) where {K,V}
        if d.ndel > 0
            rehash!(d)
        end
        @assert d.ndel == 0
        new{K,V}(copy(d.slots), copy(d.keys), copy(d.vals), 0)
    end
end
OrderedDict() = OrderedDict{Any,Any}()
OrderedDict(kv::Tuple{}) = OrderedDict()
copy(d::OrderedDict) = OrderedDict(d)
OrderedDict(kv::Tuple{Vararg{Pair{K,V}}}) where {K,V}       = OrderedDict{K,V}(kv)
OrderedDict(kv::Tuple{Vararg{Pair{K}}}) where {K}           = OrderedDict{K,Any}(kv)
OrderedDict(kv::Tuple{Vararg{Pair{K,V} where K}}) where {V} = OrderedDict{Any,V}(kv)
OrderedDict(kv::Tuple{Vararg{Pair}})                        = OrderedDict{Any,Any}(kv)
OrderedDict(kv::AbstractArray{Tuple{K,V}}) where {K,V} = OrderedDict{K,V}(kv)
OrderedDict(kv::AbstractArray{Pair{K,V}}) where {K,V}  = OrderedDict{K,V}(kv)
OrderedDict(kv::AbstractDict{K,V}) where {K,V}          = OrderedDict{K,V}(kv)
OrderedDict(ps::Pair{K,V}...) where {K,V}          = OrderedDict{K,V}(ps)
OrderedDict(ps::Pair{K}...,) where {K}             = OrderedDict{K,Any}(ps)
OrderedDict(ps::(Pair{K,V} where K)...,) where {V} = OrderedDict{Any,V}(ps)
OrderedDict(ps::Pair...)                           = OrderedDict{Any,Any}(ps)
function OrderedDict(kv)
    try
        dict_with_eltype(kv, eltype(kv))
    catch e
        if any(x->isempty(methods(x, (typeof(kv),))), [start, next, done]) ||
            !all(x->isa(x,Union{Tuple,Pair}),kv)
            throw(ArgumentError("Dict(kv): kv needs to be an iterator of tuples or pairs"))
        else
            rethrow(e)
        end
    end
end
dict_with_eltype(kv, ::Type{Tuple{K,V}}) where {K,V} = OrderedDict{K,V}(kv)
dict_with_eltype(kv, ::Type{Pair{K,V}}) where {K,V} = OrderedDict{K,V}(kv)
dict_with_eltype(kv, t) = OrderedDict{Any,Any}(kv)
similar(d::OrderedDict{K,V}) where {K,V} = OrderedDict{K,V}()
length(d::OrderedDict) = length(d.keys) - d.ndel
isempty(d::OrderedDict) = (length(d)==0)
"""
    isordered(::Type)
Property of associative containers, that is `true` if the container type has a
defined order (such as `OrderedDict` and `SortedDict`), and `false` otherwise.
"""
isordered(::Type{T}) where {T<:AbstractDict} = false
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
        else
            error("key collision during dictionary conversion")
        end
    end
    return h
end
convert(::Type{OrderedDict{K,V}},d::OrderedDict{K,V}) where {K,V} = d
function rehash!(h::OrderedDict{K,V}, newsz = length(h.slots)) where {K,V}
    olds = h.slots
    keys = h.keys
    vals = h.vals
    sz = length(olds)
    newsz = _tablesz(newsz)
    h.dirty = true
    count0 = length(h)
    if count0 == 0
        resize!(h.slots, newsz)
        fill!(h.slots, 0)
        resize!(h.keys, 0)
        resize!(h.vals, 0)
        h.ndel = 0
        return h
    end
    slots = zeros(Int32,newsz)
    if h.ndel > 0
        ndel0 = h.ndel
        ptrs = !isbits(K)
        to = 1
        # TODO: to get the best performance we need to avoid reallocating these.
        # This algorithm actually works in place, unless the dict is modified
        # due to GC during this process.
        newkeys = similar(keys, count0)
        newvals = similar(vals, count0)
        @inbounds for from = 1:length(keys)
            if !ptrs || isassigned(keys, from)
                k = keys[from]
                hashk = hash(k)%Int
                isdeleted = false
                if !ptrs
                    iter = 0
                    maxprobe = max(16, sz>>6)
                    index = (hashk & (sz-1)) + 1
                    while iter <= maxprobe
                        si = olds[index]
                        #si == 0 && break  # shouldn't happen
                        si == from && break
                        si == -from && (isdeleted=true; break)
                        index = (index & (sz-1)) + 1
                        iter += 1
                    end
                end
                if !isdeleted
                    index = (hashk & (newsz-1)) + 1
                    while slots[index] != 0
                        index = (index & (newsz-1)) + 1
                    end
                    slots[index] = to
                    newkeys[to] = k
                    newvals[to] = vals[from]
                    to += 1
                end
                if h.ndel != ndel0
                    # if items are removed by finalizers, retry
                    return rehash!(h, newsz)
                end
            end
        end
        h.keys = newkeys
        h.vals = newvals
        h.ndel = 0
    else
        @inbounds for i = 1:count0
            k = keys[i]
            index = hashindex(k, newsz)
            while slots[index] != 0
                index = (index & (newsz-1)) + 1
            end
            slots[index] = i
            if h.ndel > 0
                # if items are removed by finalizers, retry
                return rehash!(h, newsz)
            end
        end
    end
    h.slots = slots
    return h
end
function sizehint!(d::OrderedDict, newsz)
    slotsz = (newsz*3)>>1
    oldsz = length(d.slots)
    if slotsz <= oldsz
        # todo: shrink
        # be careful: rehash!() assumes everything fits. it was only designed
        # for growing.
        return d
    end
    # grow at least 25%
    slotsz = max(slotsz, (oldsz*5)>>2)
    rehash!(d, slotsz)
end
function empty!(h::OrderedDict{K,V}) where {K,V}
    fill!(h.slots, 0)
    empty!(h.keys)
    empty!(h.vals)
    h.ndel = 0
    h.dirty = true
    return h
end
function ht_keyindex(h::OrderedDict{K,V}, key, direct) where {K,V}
    slots = h.slots
    sz = length(slots)
    iter = 0
    maxprobe = max(16, sz>>6)
    index = hashindex(key, sz)
    keys = h.keys
    @inbounds while iter <= maxprobe
        si = slots[index]
        si == 0 && break
        if si > 0 && isequal(key, keys[si])
            return ifelse(direct, oftype(index, si), index)
        end
        index = (index & (sz-1)) + 1
        iter+=1
    end
    return -1
end
function ht_keyindex2(h::OrderedDict{K,V}, key) where {K,V}
    slots = h.slots
    sz = length(slots)
    iter = 0
    maxprobe = max(16, sz>>6)
    index = hashindex(key, sz)
    keys = h.keys
    @inbounds while iter <= maxprobe
        si = slots[index]
        if si == 0
            return -index
        elseif si > 0 && isequal(key, keys[si])
            return oftype(index, si)
        end
        index = (index & (sz-1)) + 1
        iter+=1
    end
    rehash!(h, length(h) > 64000 ? sz*2 : sz*4)
    return ht_keyindex2(h, key)
end
function _setindex!(h::OrderedDict, v, key, index)
    hk, hv = h.keys, h.vals
    #push!(h.keys, key)
    ccall(:jl_array_grow_end, Cvoid, (Any, UInt), hk, 1)
    nk = length(hk)
    @inbounds hk[nk] = key
    #push!(h.vals, v)
    ccall(:jl_array_grow_end, Cvoid, (Any, UInt), hv, 1)
    @inbounds hv[nk] = v
    @inbounds h.slots[index] = nk
    h.dirty = true
    sz = length(h.slots)
    cnt = nk - h.ndel
    # Rehash now if necessary
    if h.ndel >= ((3*nk)>>2) || cnt*3 > sz*2
        # > 3/4 deleted or > 2/3 full
        rehash!(h, cnt > 64000 ? cnt*2 : cnt*4)
    end
end
function setindex!(h::OrderedDict{K,V}, v0, key0) where {K,V}
    key = convert(K,key0)
    if !isequal(key,key0)
        throw(ArgumentError("$key0 is not a valid key for type $K"))
    end
    v = convert(V,  v0)
    index = ht_keyindex2(h, key)
    if index > 0
        @inbounds h.keys[index] = key
        @inbounds h.vals[index] = v
    else
        _setindex!(h, v, key, -index)
    end
    return h
end
function get!(h::OrderedDict{K,V}, key0, default) where {K,V}
    key = convert(K,key0)
    if !isequal(key,key0)
        throw(ArgumentError("$key0 is not a valid key for type $K"))
    end
    index = ht_keyindex2(h, key)
    index > 0 && return h.vals[index]
    v = convert(V,  default)
    _setindex!(h, v, key, -index)
    return v
end
function get!(default::Base.Callable, h::OrderedDict{K,V}, key0) where {K,V}
    key = convert(K,key0)
    if !isequal(key,key0)
        throw(ArgumentError("$key0 is not a valid key for type $K"))
    end
    index = ht_keyindex2(h, key)
    index > 0 && return h.vals[index]
    h.dirty = false
    v = convert(V,  default())
    if h.dirty
        index = ht_keyindex2(h, key)
    end
    if index > 0
        h.keys[index] = key
        h.vals[index] = v
    else
        _setindex!(h, v, key, -index)
    end
    return v
end
function getindex(h::OrderedDict{K,V}, key) where {K,V}
    index = ht_keyindex(h, key, true)
    return (index<0) ? throw(KeyError(key)) : h.vals[index]::V
end
function get(h::OrderedDict{K,V}, key, default) where {K,V}
    index = ht_keyindex(h, key, true)
    return (index<0) ? default : h.vals[index]::V
end
function get(default::Base.Callable, h::OrderedDict{K,V}, key) where {K,V}
    index = ht_keyindex(h, key, true)
    return (index<0) ? default() : h.vals[index]::V
end
haskey(h::OrderedDict, key) = (ht_keyindex(h, key, true) >= 0)
if isdefined(Base, :KeySet) # 0.7.0-DEV.2722
    in(key, v::Base.KeySet{K,T}) where {K,T<:OrderedDict{K}} = (ht_keyindex(v.dict, key, true) >= 0)
else
    in(key, v::Base.KeyIterator{T}) where {T<:OrderedDict} = (ht_keyindex(v.dict, key, true) >= 0)
end
function getkey(h::OrderedDict{K,V}, key, default) where {K,V}
    index = ht_keyindex(h, key, true)
    return (index<0) ? default : h.keys[index]::K
end
function _pop!(h::OrderedDict, index)
    @inbounds val = h.vals[h.slots[index]]
    _delete!(h, index)
    return val
end
function pop!(h::OrderedDict)
    h.ndel > 0 && rehash!(h)
    key = h.keys[end]
    index = ht_keyindex(h, key, false)
    key => _pop!(h, index)
end
function pop!(h::OrderedDict, key)
    index = ht_keyindex(h, key, false)
    index > 0 ? _pop!(h, index) : throw(KeyError(key))
end
function pop!(h::OrderedDict, key, default)
    index = ht_keyindex(h, key, false)
    index > 0 ? _pop!(h, index) : default
end
function _delete!(h::OrderedDict, index)
    @inbounds ki = h.slots[index]
    @inbounds h.slots[index] = -ki
    ccall(:jl_arrayunset, Cvoid, (Any, UInt), h.keys, ki-1)
    ccall(:jl_arrayunset, Cvoid, (Any, UInt), h.vals, ki-1)
    h.ndel += 1
    h.dirty = true
    h
end
function delete!(h::OrderedDict, key)
    index = ht_keyindex(h, key, false)
    if index > 0; _delete!(h, index); end
    h
end
function start(t::OrderedDict)
    t.ndel > 0 && rehash!(t)
    1
end
done(t::OrderedDict, i) = done(t.keys, i)
next(t::OrderedDict, i) = (Pair(t.keys[i],t.vals[i]), i+1)
if isdefined(Base, :KeySet) # 0.7.0-DEV.2722
    next(v::Base.KeySet{K,T}, i) where {K,T<:OrderedDict{K}} = (v.dict.keys[i], i+1)
else
    next(v::Base.KeyIterator{T}, i) where {T<:OrderedDict} = (v.dict.keys[i], i+1)
end
next(v::ValueIterator{T}, i) where {T<:OrderedDict} = (v.dict.vals[i], i+1)
function merge(d::OrderedDict, others::AbstractDict...)
    K, V = keytype(d), valtype(d)
    for other in others
        K = promote_type(K, keytype(other))
        V = promote_type(V, valtype(other))
    end
    merge!(OrderedDict{K,V}(), d, others...)
end
struct OrderedSet{T}
    dict::OrderedDict{T,Nothing}
    OrderedSet{T}() where {T} = new{T}(OrderedDict{T,Nothing}())
    OrderedSet{T}(xs) where {T} = union!(new{T}(OrderedDict{T,Nothing}()), xs)
end
OrderedSet() = OrderedSet{Any}()
OrderedSet(xs) = OrderedSet{eltype(xs)}(xs)
show(io::IO, s::OrderedSet) = (show(io, typeof(s)); print(io, "("); !isempty(s) && Base.show_comma_array(io, s,'[',']'); print(io, ")"))
@delegate OrderedSet.dict [isempty, length]
sizehint!(s::OrderedSet, sz::Integer) = (sizehint!(s.dict, sz); s)
eltype(s::OrderedSet{T}) where {T} = T
in(x, s::OrderedSet) = haskey(s.dict, x)
push!(s::OrderedSet, x) = (s.dict[x] = nothing; s)
pop!(s::OrderedSet, x) = (pop!(s.dict, x); x)
pop!(s::OrderedSet, x, deflt) = pop!(s.dict, x, deflt) == deflt ? deflt : x
delete!(s::OrderedSet, x) = (delete!(s.dict, x); s)
getindex(x::OrderedSet,i::Int) = x.dict.keys[i]
endof(x::OrderedSet) = endof(x.dict.keys)
Base.nextind(::OrderedSet, i::Int) = i + 1
union!(s::OrderedSet, xs) = (for x in xs; push!(s,x); end; s)
setdiff!(s::OrderedSet, xs) = (for x in xs; delete!(s,x); end; s)
setdiff!(s::Set, xs::OrderedSet) = (for x in xs; delete!(s,x); end; s)
similar(s::OrderedSet{T}) where {T} = OrderedSet{T}()
copy(s::OrderedSet) = union!(similar(s), s)
empty!(s::OrderedSet{T}) where {T} = (empty!(s.dict); s)
start(s::OrderedSet)       = start(s.dict)
done(s::OrderedSet, state) = done(s.dict, state)
next(s::OrderedSet, i)     = (s.dict.keys[i], i+1)
pop!(s::OrderedSet) = pop!(s.dict)[1]
union(s::OrderedSet) = copy(s)
function union(s::OrderedSet, sets...)
    u = OrderedSet{Base.promote_eltype(s, sets...)}()
    union!(u,s)
    for t in sets
        union!(u,t)
    end
    return u
end
intersect(s::OrderedSet) = copy(s)
function intersect(s::OrderedSet, sets...)
    i = copy(s)
    for x in s
        for t in sets
            if !in(x,t)
                delete!(i,x)
                break
            end
        end
    end
    return i
end
function setdiff(a::OrderedSet, b)
    d = similar(a)
    for x in a
        if !(x in b)
            push!(d, x)
        end
    end
    d
end
==(l::OrderedSet, r::OrderedSet) = (length(l) == length(r)) && (l <= r)
<(l::OrderedSet, r::OrderedSet) = (length(l) < length(r)) && (l <= r)
<=(l::OrderedSet, r::OrderedSet) = issubset(l, r)
function filter!(f::Function, s::OrderedSet)
    for x in s
        if !f(x)
            delete!(s, x)
        end
    end
    return s
end
filter(f::Function, s::OrderedSet) = filter!(f, copy(s))
const orderedset_seed = UInt === UInt64 ? 0x2114638a942a91a5 : 0xd86bdbf1
function hash(s::OrderedSet, h::UInt)
    h = hash(orderedset_seed, h)
    s.dict.ndel > 0 && rehash!(s.dict)
    hash(s.dict.keys, h)
end
struct DefaultDictBase{K,V,F,D} <: AbstractDict{K,V}
    default::F
    d::D
    check_D(D,K,V) = (D <: AbstractDict{K,V}) ||
        throw(ArgumentError("Default dict must be <: AbstractDict{$K,$V}"))
    DefaultDictBase{K,V,F,D}(x::F, kv::AbstractArray{Tuple{K,V}}) where {K,V,F,D} =
        (check_D(D,K,V); new{K,V,F,D}(x, D(kv)))
    DefaultDictBase{K,V,F,D}(x::F, ps::Pair{K,V}...) where {K,V,F,D} =
        (check_D(D,K,V); new{K,V,F,D}(x, D(ps...)))
    DefaultDictBase{K,V,F,D}(x::F, d::D) where {K,V,F,D<:DefaultDictBase} =
        (check_D(D,K,V); DefaultDictBase(x, d.d))
    DefaultDictBase{K,V,F,D}(x::F, d::D = D()) where {K,V,F,D} =
        (check_D(D,K,V); new{K,V,F,D}(x, d))
end
DefaultDictBase() = throw(ArgumentError("no default specified"))
DefaultDictBase(k,v) = throw(ArgumentError("no default specified"))
DefaultDictBase(default::F) where {F} = DefaultDictBase{Any,Any,F,Dict{Any,Any}}(default)
DefaultDictBase(default::F, kv::AbstractArray{Tuple{K,V}}) where {K,V,F} = DefaultDictBase{K,V,F,Dict{K,V}}(default, kv)
DefaultDictBase(default::F, ps::Pair{K,V}...) where {K,V,F} = DefaultDictBase{K,V,F,Dict{K,V}}(default, ps...)
DefaultDictBase(default::F, d::D) where {F,D<:AbstractDict} = (K=keytype(d); V=valtype(d); DefaultDictBase{K,V,F,D}(default, d))
DefaultDictBase{K,V}(default::F) where {K,V,F} = DefaultDictBase{K,V,F,Dict{K,V}}(default)
@delegate DefaultDictBase.d [ get, haskey, getkey, pop!,
                              start, done, next, isempty, length ]
@delegate_return_parent DefaultDictBase.d [ delete!, empty!, setindex!, sizehint! ]
similar(d::DefaultDictBase{K,V,F}) where {K,V,F} = DefaultDictBase{K,V,F}(d.default)
if isdefined(Base, :KeySet) # 0.7.0-DEV.2722
    in(key, v::Base.KeySet{K,T}) where {K,T<:DefaultDictBase{K}} = key in keys(v.dict.d)
    next(v::Base.KeySet{K,T}, i) where {K,T<:DefaultDictBase{K}} = (v.dict.d.keys[i], Base.skip_deleted(v.dict.d,i+1))
else
    in(key, v::Base.KeyIterator{T}) where {T<:DefaultDictBase} = key in keys(v.dict.d)
    next(v::Base.KeyIterator{T}, i) where {T<:DefaultDictBase} = (v.dict.d.keys[i], Base.skip_deleted(v.dict.d,i+1))
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
                new{K,V,F}(DefaultDictBase{K,V,F,$_Dict{K,V}}(x, ps...))
            $DefaultDict{K,V,F}(x, kv::AbstractArray{Tuple{K,V}}) where {K,V,F} =
                new{K,V,F}(DefaultDictBase{K,V,F,$_Dict{K,V}}(x, kv))
            $DefaultDict{K,V,F}(x, d::$DefaultDict) where {K,V,F} = $DefaultDict(x, d.d)
            $DefaultDict{K,V,F}(x, d::$_Dict) where {K,V,F} =
                new{K,V,F}(DefaultDictBase{K,V,F,$_Dict{K,V}}(x, d))
            $DefaultDict{K,V,F}(x) where {K,V,F} =
                new{K,V,F}(DefaultDictBase{K,V,F,$_Dict{K,V}}(x))
        end
        ## Constructors
        $DefaultDict() = throw(ArgumentError("$DefaultDict: no default specified"))
        $DefaultDict(k,v) = throw(ArgumentError("$DefaultDict: no default specified"))
        # syntax entry points
        $DefaultDict(default::F) where {F} = $DefaultDict{Any,Any,F}(default)
        $DefaultDict(default::F, kv::AbstractArray{Tuple{K,V}}) where {K,V,F} = $DefaultDict{K,V,F}(default, kv)
        $DefaultDict(default::F, ps::Pair{K,V}...) where {K,V,F} = $DefaultDict{K,V,F}(default, ps...)
        $DefaultDict(default::F, d::AbstractDict) where {F} = ((K,V)= (Base.keytype(d), Base.valtype(d)); $DefaultDict{K,V,F}(default, $_Dict(d)))
        # Constructor syntax: DefaultDictBase{Int,Float64}(default)
        $DefaultDict{K,V}() where {K,V} = throw(ArgumentError("$DefaultDict: no default specified"))
        $DefaultDict{K,V}(default::F) where {K,V,F} = $DefaultDict{K,V,F}(default)
        ## Functions
        # Most functions are simply delegated to the wrapped DefaultDictBase object
        @delegate $DefaultDict.d [ getindex, get, get!, haskey,
                                   getkey, pop!, start, next,
                                   done, isempty, length ]
        # Some functions are delegated, but then need to return the main dictionary
        # NOTE: push! is not included below, because the fallback version just
        #       calls setindex!
        @delegate_return_parent $DefaultDict.d [ delete!, empty!, setindex!, sizehint! ]
        # NOTE: The second and third definition of push! below are only
        # necessary for disambiguating with the fourth, fifth, and sixth
        # definitions of push! below.
        # If these are removed, the second and third definitions can be
        # removed as well.
        push!(d::$DefaultDict, p::Pair) = (setindex!(d.d, p.second, p.first); d)
        push!(d::$DefaultDict, p::Pair, q::Pair) = push!(push!(d, p), q)
        push!(d::$DefaultDict, p::Pair, q::Pair, r::Pair...) = push!(push!(push!(d, p), q), r...)
        push!(d::$DefaultDict, p) = (setindex!(d.d, p[2], p[1]); d)
        push!(d::$DefaultDict, p, q) = push!(push!(d, p), q)
        push!(d::$DefaultDict, p, q, r...) = push!(push!(push!(d, p), q), r...)
        similar(d::$DefaultDict{K,V,F}) where {K,V,F} = $DefaultDict{K,V,F}(d.d.default)
        if isdefined(Base, :KeySet) # 0.7.0-DEV.2722
            in(key, v::Base.KeySet{K,T}) where {K,T<:$DefaultDict{K}} = key in keys(v.dict.d.d)
        else
            in(key, v::Base.KeyIterator{T}) where {T<:$DefaultDict} = key in keys(v.dict.d.d)
        end
    end
end
isordered(::Type{T}) where {T<:DefaultOrderedDict} = true
mutable struct Trie{T}
    value::T
    children::Dict{Char,Trie{T}}
    is_key::Bool
    function Trie{T}() where T
        self = new{T}()
        self.children = Dict{Char,Trie{T}}()
        self.is_key = false
        self
    end
    function Trie{T}(ks, vs) where T
        t = Trie{T}()
        for (k, v) in zip(ks, vs)
            t[k] = v
        end
        return t
    end
    function Trie{T}(kv) where T
        t = Trie{T}()
        for (k,v) in kv
            t[k] = v
        end
        return t
    end
end
Trie() = Trie{Any}()
Trie(ks::AbstractVector{K}, vs::AbstractVector{V}) where {K<:AbstractString,V} = Trie{V}(ks, vs)
Trie(kv::AbstractVector{Tuple{K,V}}) where {K<:AbstractString,V} = Trie{V}(kv)
Trie(kv::AbstractDict{K,V}) where {K<:AbstractString,V} = Trie{V}(kv)
Trie(ks::AbstractVector{K}) where {K<:AbstractString} = Trie{Nothing}(ks, similar(ks, Nothing))
function setindex!(t::Trie{T}, val::T, key::AbstractString) where T
    node = t
    for char in key
        if !haskey(node.children, char)
            node.children[char] = Trie{T}()
        end
        node = node.children[char]
    end
    node.is_key = true
    node.value = val
end
function getindex(t::Trie, key::AbstractString)
    node = subtrie(t, key)
    if node != nothing && node.is_key
        return node.value
    end
    throw(KeyError("key not found: $key"))
end
function subtrie(t::Trie, prefix::AbstractString)
    node = t
    for char in prefix
        if !haskey(node.children, char)
            return nothing
        else
            node = node.children[char]
        end
    end
    node
end
function haskey(t::Trie, key::AbstractString)
    node = subtrie(t, key)
    node != nothing && node.is_key
end
function get(t::Trie, key::AbstractString, notfound)
    node = subtrie(t, key)
    if node != nothing && node.is_key
        return node.value
    end
    notfound
end
function keys(t::Trie, prefix::AbstractString="", found=AbstractString[])
    if t.is_key
        push!(found, prefix)
    end
    for (char,child) in t.children
        keys(child, string(prefix,char), found)
    end
    found
end
function keys_with_prefix(t::Trie, prefix::AbstractString)
    st = subtrie(t, prefix)
    st != nothing ? keys(st,prefix) : []
end
struct TrieIterator
    t::Trie
    str::AbstractString
end
start(it::TrieIterator) = (it.t, 0)
function next(it::TrieIterator, state)
    t, i = state
    i == 0 && return it.t, (it.t, 1)
    t = t.children[it.str[i]]
    return (t, (t, i + 1))
end
function done(it::TrieIterator, state)
    t, i = state
    i == 0 && return false
    i == length(it.str) + 1 && return true
    return !(it.str[i] in keys(t.children))
end
path(t::Trie, str::AbstractString) = TrieIterator(t, str)
Base.iteratorsize(::Type{TrieIterator}) = Base.SizeUnknown()
import Base: similar, copy, copy!, eltype, push!, pop!, delete!, shift!,
             empty!, isempty, union, union!, intersect, intersect!,
             setdiff, setdiff!, symdiff, symdiff!, in, start, next, done,
             last, length, show, hash, issubset, ==, <=, <, unsafe_getindex,
             unsafe_setindex!, findnextnot, first
if !isdefined(Base, :complement)
    export complement, complement!
else
    import Base: complement, complement!
end
mutable struct IntSet
    bits::BitVector
    inverse::Bool
    IntSet() = new(falses(256), false)
end
IntSet(itr) = union!(IntSet(), itr)
similar(s::IntSet) = IntSet()
copy(s1::IntSet) = copy!(IntSet(), s1)
function copy!(to::IntSet, from::IntSet)
    resize!(to.bits, length(from.bits))
    copy!(to.bits, from.bits)
    to.inverse = from.inverse
    to
end
eltype(s::IntSet) = Int
sizehint!(s::IntSet, n::Integer) = (_resize0!(s.bits, n+1); s)
function first(itr::IntSet)
    state = start(itr)
    done(itr, state) && throw(ArgumentError("collection must be non-empty"))
    next(itr, state)[1]
end
@inline function _setint!(s::IntSet, n::Integer, b::Bool)
    idx = n+1
    if idx > length(s.bits)
        !b && return s # setting a bit to zero outside the set's bits is a no-op
        newlen = idx + idx>>1 # This operation may overflow; we want saturation
        _resize0!(s.bits, ifelse(newlen<0, typemax(Int), newlen))
    end
    unsafe_setindex!(s.bits, b, idx) # Use @inbounds once available
    s
end
@inline function _resize0!(b::BitVector, newlen::Integer)
    len = length(b)
    resize!(b, newlen)
    len < newlen && unsafe_setindex!(b, false, len+1:newlen) # resize! gives dirty memory
    b
end
function _matchlength!(b::BitArray, newlen::Integer)
    len = length(b)
    len > newlen && return splice!(b, newlen+1:len)
    len < newlen && _resize0!(b, newlen)
    return BitVector()
end
const _intset_bounds_err_msg = "elements of IntSet must be between 0 and typemax(Int)-1"
function push!(s::IntSet, n::Integer)
    0 <= n < typemax(Int) || throw(ArgumentError(_intset_bounds_err_msg))
    _setint!(s, n, !s.inverse)
end
push!(s::IntSet, ns::Integer...) = (for n in ns; push!(s, n); end; s)
function pop!(s::IntSet)
    s.inverse && throw(ArgumentError("cannot pop the last element of complement IntSet"))
    pop!(s, last(s))
end
function pop!(s::IntSet, n::Integer)
    0 <= n < typemax(Int) || throw(ArgumentError(_intset_bounds_err_msg))
    n in s ? (_delete!(s, n); n) : throw(KeyError(n))
end
function pop!(s::IntSet, n::Integer, default)
    0 <= n < typemax(Int) || throw(ArgumentError(_intset_bounds_err_msg))
    n in s ? (_delete!(s, n); n) : default
end
function pop!(f::Function, s::IntSet, n::Integer)
    0 <= n < typemax(Int) || throw(ArgumentError(_intset_bounds_err_msg))
    n in s ? (_delete!(s, n); n) : f()
end
_delete!(s::IntSet, n::Integer) = _setint!(s, n, s.inverse)
delete!(s::IntSet, n::Integer) = n < 0 ? s : _delete!(s, n)
shift!(s::IntSet) = pop!(s, first(s))
empty!(s::IntSet) = (fill!(s.bits, false); s.inverse = false; s)
isempty(s::IntSet) = s.inverse ? length(s.bits) == typemax(Int) && all(s.bits) : !any(s.bits)
union(s::IntSet, ns) = union!(copy(s), ns)
union!(s::IntSet, ns) = (for n in ns; push!(s, n); end; s)
function union!(s1::IntSet, s2::IntSet)
    l = length(s2.bits)
    if     !s1.inverse & !s2.inverse;  e = _matchlength!(s1.bits, l); map!(|, s1.bits, s1.bits, s2.bits); append!(s1.bits, e)
    elseif  s1.inverse & !s2.inverse;  e = _matchlength!(s1.bits, l); map!(>, s1.bits, s1.bits, s2.bits); append!(s1.bits, e)
    elseif !s1.inverse &  s2.inverse;  _resize0!(s1.bits, l);         map!(<, s1.bits, s1.bits, s2.bits); s1.inverse = true
    else #= s1.inverse &  s2.inverse=# _resize0!(s1.bits, l);         map!(&, s1.bits, s1.bits, s2.bits)
    end
    s1
end
intersect(s1::IntSet) = copy(s1)
intersect(s1::IntSet, ss...) = intersect(s1, intersect(ss...))
function intersect(s1::IntSet, ns)
    s = IntSet()
    for n in ns
        n in s1 && push!(s, n)
    end
    s
end
intersect(s1::IntSet, s2::IntSet) = intersect!(copy(s1), s2)
function intersect!(s1::IntSet, s2::IntSet)
    l = length(s2.bits)
    if     !s1.inverse & !s2.inverse;  _resize0!(s1.bits, l);         map!(&, s1.bits, s1.bits, s2.bits)
    elseif  s1.inverse & !s2.inverse;  _resize0!(s1.bits, l);         map!(<, s1.bits, s1.bits, s2.bits); s1.inverse = false
    elseif !s1.inverse &  s2.inverse;  e = _matchlength!(s1.bits, l); map!(>, s1.bits, s1.bits, s2.bits); append!(s1.bits, e)
    else #= s1.inverse &  s2.inverse=# e = _matchlength!(s1.bits, l); map!(|, s1.bits, s1.bits, s2.bits); append!(s1.bits, e)
    end
    s1
end
setdiff(s::IntSet, ns) = setdiff!(copy(s), ns)
setdiff!(s::IntSet, ns) = (for n in ns; _delete!(s, n); end; s)
function setdiff!(s1::IntSet, s2::IntSet)
    l = length(s2.bits)
    if     !s1.inverse & !s2.inverse;  e = _matchlength!(s1.bits, l); map!(>, s1.bits, s1.bits, s2.bits); append!(s1.bits, e)
    elseif  s1.inverse & !s2.inverse;  e = _matchlength!(s1.bits, l); map!(|, s1.bits, s1.bits, s2.bits); append!(s1.bits, e)
    elseif !s1.inverse &  s2.inverse;  _resize0!(s1.bits, l);         map!(&, s1.bits, s1.bits, s2.bits)
    else #= s1.inverse &  s2.inverse=# _resize0!(s1.bits, l);         map!(<, s1.bits, s1.bits, s2.bits); s1.inverse = false
    end
    s1
end
symdiff(s::IntSet, ns) = symdiff!(copy(s), ns)
symdiff!(s::IntSet, ns) = (for n in ns; symdiff!(s, n); end; s)
function symdiff!(s::IntSet, n::Integer)
    0 <= n < typemax(Int) || throw(ArgumentError(_intset_bounds_err_msg))
    val = (n in s) ⊻ !s.inverse
    _setint!(s, n, val)
    s
end
function symdiff!(s1::IntSet, s2::IntSet)
    e = _matchlength!(s1.bits, length(s2.bits))
    map!(⊻, s1.bits, s1.bits, s2.bits)
    s2.inverse && (s1.inverse = !s1.inverse)
    append!(s1.bits, e)
    s1
end
function in(n::Integer, s::IntSet)
    idx = n+1
    if 1 <= idx <= length(s.bits)
        unsafe_getindex(s.bits, idx) != s.inverse
    else
        ifelse((idx <= 0) | (idx > typemax(Int)), false, s.inverse)
    end
end
start(s::IntSet) = next(s, 0)[2]
function next(s::IntSet, i, invert=false)
    if s.inverse ⊻ invert
        # i+1 could rollover causing a BoundsError in findnext/findnextnot
        nextidx = i == typemax(Int) ? 0 : findnextnot(s.bits, i+1)
        # Extend indices beyond the length of the bits since it is inverted
        nextidx = nextidx == 0 ? max(i, length(s.bits))+1 : nextidx
    else
        nextidx = i == typemax(Int) ? 0 : findnext(s.bits, i+1)
    end
    (i-1, nextidx)
end
done(s::IntSet, i) = i <= 0
nextnot(s::IntSet, i) = next(s, i, true)
function last(s::IntSet)
    l = length(s.bits)
    if s.inverse
        idx = l < typemax(Int) ? typemax(Int) : findprevnot(s.bits, l)
    else
        idx = findprev(s.bits, l)
    end
    idx == 0 ? throw(ArgumentError("collection must be non-empty")) : idx - 1
end
length(s::IntSet) = (n = sum(s.bits); ifelse(s.inverse, typemax(Int) - n, n))
complement(s::IntSet) = complement!(copy(s))
complement!(s::IntSet) = (s.inverse = !s.inverse; s)
function show(io::IO, s::IntSet)
    print(io, "IntSet([")
    first = true
    for n in s
        if s.inverse && n > 2 && done(s, nextnot(s, n-3)[2])
             print(io, ", ..., ", typemax(Int)-1)
             break
         end
        !first && print(io, ", ")
        print(io, n)
        first = false
    end
    print(io, "])")
end
function ==(s1::IntSet, s2::IntSet)
    l1 = length(s1.bits)
    l2 = length(s2.bits)
    l1 < l2 && return ==(s2, s1) # Swap so s1 is always equal-length or longer
    # Try to do this without allocating memory or checking bit-by-bit
    if s1.inverse == s2.inverse
        # If the lengths are the same, simply punt to bitarray comparison
        l1 == l2 && return s1.bits == s2.bits
        # Otherwise check the last bit. If equal, we only need to check up to l2
        return findprev(s1.bits, l1) == findprev(s2.bits, l2) &&
               unsafe_getindex(s1.bits, 1:l2) == s2.bits
    else
        # one complement, one not. Could feasibly be true on 32 bit machines
        # Only if all non-overlapping bits are set and overlaps are inverted
        return l1 == typemax(Int) &&
               map!(!, unsafe_getindex(s1.bits, 1:l2)) == s2.bits &&
               (l1 == l2 || all(unsafe_getindex(s1.bits, l2+1:l1)))
    end
end
const hashis_seed = UInt === UInt64 ? 0x88989f1fc7dea67d : 0xc7dea67d
function hash(s::IntSet, h::UInt)
    # Only hash the bits array up to the last-set bit to prevent extra empty
    # bits from changing the hash result
    l = findprev(s.bits, length(s.bits))
    hash(unsafe_getindex(s.bits, 1:l), h) ⊻ hash(s.inverse) ⊻ hashis_seed
end
issubset(a::IntSet, b::IntSet) = isequal(a, intersect(a,b))
<(a::IntSet, b::IntSet) = (a<=b) && !isequal(a,b)
<=(a::IntSet, b::IntSet) = issubset(a, b)
abstract type LinkedList{T} end
mutable struct Nil{T} <: LinkedList{T}
end
mutable struct Cons{T} <: LinkedList{T}
    head::T
    tail::LinkedList{T}
end
cons(h, t::LinkedList{T}) where {T} = Cons{T}(h, t)
nil(T) = Nil{T}()
nil() = nil(Any)
head(x::Cons) = x.head
tail(x::Cons) = x.tail
==(x::Nil, y::Nil) = true
==(x::Cons, y::Cons) = (x.head == y.head) && (x.tail == y.tail)
function show(io::IO, l::LinkedList{T}) where T
    if isa(l,Nil)
        if T === Any
            print(io, "nil()")
        else
            print(io, "nil(", T, ")")
        end
    else
        print(io, "list(")
        show(io, head(l))
        for t in tail(l)
            print(io, ", ")
            show(io, t)
        end
        print(io, ")")
    end
end
list() = nil()
function list(elts...)
    l = nil()
    for i=length(elts):-1:1
        l = cons(elts[i],l)
    end
    return l
end
function list(elts::T...) where T
    l = nil(T)
    for i=length(elts):-1:1
        l = cons(elts[i],l)
    end
    return l
end
length(l::Nil) = 0
function length(l::Cons)
    n = 0
    for i in l
        n += 1
    end
    n
end
map(f::Base.Callable, l::Nil) = l
function map(f::Base.Callable, l::Cons)
    first = f(l.head)
    l2 = cons(first, nil(typeof(first)))
    for h in l.tail
        l2 = cons(f(h), l2)
    end
    reverse(l2)
end
function filter(f::Function, l::LinkedList{T}) where T
    l2 = nil(T)
    for h in l
        if f(h)
            l2 = cons(h, l2)
        end
    end
    reverse(l2)
end
function reverse(l::LinkedList{T}) where T
    l2 = nil(T)
    for h in l
        l2 = cons(h, l2)
    end
    l2
end
copy(l::Nil) = l
function copy(l::Cons)
    l2 = reverse(reverse(l))
end
cat(lst::LinkedList) = lst
function cat(lst::LinkedList, lsts::LinkedList...)
    T = typeof(lst).parameters[1]
    n = length(lsts)
    for i = 1:n
        T2 = typeof(lsts[i]).parameters[1]
        T = typejoin(T, T2)
    end
    l2 = nil(T)
    for h in lst
        l2 = cons(h, l2)
    end
    for i = 1:n
        for h in lsts[i]
            l2 = cons(h, l2)
        end
    end
    reverse(l2)
end
start(l::Nil{T}) where {T} = l
start(l::Cons{T}) where {T} = l
done(l::Cons{T}, state::Cons{T}) where {T} = false
done(l::LinkedList, state::Nil{T}) where {T} = true
next(l::Cons{T}, state::Cons{T}) where {T} = (state.head, state.tail)
struct KDRec{K,D}
    parent::Int
    k::K
    d::D
    KDRec{K,D}(p::Int, k1::K, d1::D) where {K,D} = new{K,D}(p,k1,d1)
    KDRec{K,D}(p::Int) where {K,D} = new{K,D}(p)
end
struct TreeNode{K}
    child1::Int
    child2::Int
    child3::Int
    parent::Int
    splitkey1::K
    splitkey2::K
    TreeNode{K}(::Type{K}, c1::Int, c2::Int, c3::Int, p::Int) where {K} = new{K}(c1, c2, c3, p)
    TreeNode{K}(c1::Int, c2::Int, c3::Int, p::Int, sk1::K, sk2::K) where {K} =
        new{K}(c1, c2, c3, p, sk1, sk2)
end
function initializeTree!(tree::Array{TreeNode{K},1}) where K
    resize!(tree,1)
    tree[1] = TreeNode{K}(K, 1, 2, 0, 0)
    nothing
end
function initializeData!(data::Array{KDRec{K,D},1}) where {K,D}
    resize!(data, 2)
    data[1] = KDRec{K,D}(1)
    data[2] = KDRec{K,D}(1)
    nothing
end
mutable struct BalancedTree23{K, D, Ord <: Ordering}
    ord::Ord
    data::Array{KDRec{K,D}, 1}
    tree::Array{TreeNode{K}, 1}
    rootloc::Int
    depth::Int
    freetreeinds::Array{Int,1}
    freedatainds::Array{Int,1}
    useddatacells::IntSet
    # The next two arrays are used as a workspace by the delete!
    # function.
    deletionchild::Array{Int,1}
    deletionleftkey::Array{K,1}
    function BalancedTree23{K,D,Ord}(ord1::Ord) where {K,D,Ord<:Ordering}
        tree1 = Vector{TreeNode{K}}(uninitialized, 1)
        initializeTree!(tree1)
        data1 = Vector{KDRec{K,D}}(uninitialized, 2)
        initializeData!(data1)
        u1 = IntSet()
        push!(u1, 1, 2)
        new{K,D,Ord}(ord1, data1, tree1, 1, 1, Vector{Int}(), Vector{Int}(),
                     u1,
                     Vector{Int}(uninitialized, 3), Vector{K}(uninitialized, 3))
    end
end
@inline function cmp2_nonleaf(o::Ordering,
                              treenode::TreeNode,
                              k)
    lt(o, k, treenode.splitkey1) ? 1 : 2
end
@inline function cmp2_leaf(o::Ordering,
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
    lt(o, k, treenode.splitkey1) ?                           1 :
    (treenode.child3 == 2 || lt(o, k, treenode.splitkey2)) ? 2 : 3
end
@inline function cmp2le_nonleaf(o::Ordering,
                                treenode::TreeNode,
                                k)
    !lt(o,treenode.splitkey1,k) ? 1 : 2
end
@inline function cmp2le_leaf(o::Ordering,
                             treenode::TreeNode,
                             k)
    treenode.child2 == 2 || !lt(o,treenode.splitkey1,k) ? 1 : 2
end
@inline function cmp3le_nonleaf(o::Ordering,
                                treenode::TreeNode,
                                k)
    !lt(o,treenode.splitkey1, k) ? 1 :
    !lt(o,treenode.splitkey2, k) ? 2 : 3
end
@inline function cmp3le_leaf(o::Ordering,
                             treenode::TreeNode,
                             k)
    !lt(o,treenode.splitkey1,k) ?                            1 :
    (treenode.child3 == 2 || !lt(o,treenode.splitkey2, k)) ? 2 : 3
end
function empty!(t::BalancedTree23)
    resize!(t.data,2)
    initializeData!(t.data)
    resize!(t.tree,1)
    initializeTree!(t.tree)
    t.depth = 1
    t.rootloc = 1
    t.freetreeinds = Vector{Int}()
    t.freedatainds = Vector{Int}()
    empty!(t.useddatacells)
    push!(t.useddatacells, 1, 2)
    nothing
end
eq(::ForwardOrdering, a, b) = isequal(a,b)
eq(::ReverseOrdering{ForwardOrdering}, a, b) = isequal(a,b)
eq(o::Ordering, a, b) = !lt(o, a, b) && !lt(o, b, a)
function findkey(t::BalancedTree23, k)
    curnode = t.rootloc
    for depthcount = 1 : t.depth - 1
        @inbounds thisnode = t.tree[curnode]
        cmp = thisnode.child3 == 0 ?
                         cmp2_nonleaf(t.ord, thisnode, k) :
                         cmp3_nonleaf(t.ord, thisnode, k)
        curnode = cmp == 1 ? thisnode.child1 :
                  cmp == 2 ? thisnode.child2 : thisnode.child3
    end
    @inbounds thisnode = t.tree[curnode]
    cmp = thisnode.child3 == 0 ?
                cmp2_leaf(t.ord, thisnode, k) :
                cmp3_leaf(t.ord, thisnode, k)
    curnode = cmp == 1 ? thisnode.child1 :
              cmp == 2 ? thisnode.child2 : thisnode.child3
    @inbounds return curnode, (curnode > 2 && eq(t.ord, t.data[curnode].k, k))
end
function findkeyless(t::BalancedTree23, k)
    curnode = t.rootloc
    for depthcount = 1 : t.depth - 1
        @inbounds thisnode = t.tree[curnode]
        cmp = thisnode.child3 == 0 ?
               cmp2le_nonleaf(t.ord, thisnode, k) :
               cmp3le_nonleaf(t.ord, thisnode, k)
        curnode = cmp == 1 ? thisnode.child1 :
                  cmp == 2 ? thisnode.child2 : thisnode.child3
    end
    @inbounds thisnode = t.tree[curnode]
    cmp = thisnode.child3 == 0 ?
            cmp2le_leaf(t.ord, thisnode, k) :
            cmp3le_leaf(t.ord, thisnode, k)
    curnode = cmp == 1 ? thisnode.child1 :
              cmp == 2 ? thisnode.child2 : thisnode.child3
    curnode
end
function replaceparent!(data::Array{KDRec{K,D},1}, whichind::Int, newparent::Int) where {K,D}
    data[whichind] = KDRec{K,D}(newparent, data[whichind].k, data[whichind].d)
    nothing
end
function replaceparent!(tree::Array{TreeNode{K},1}, whichind::Int, newparent::Int) where K
    tree[whichind] = TreeNode{K}(tree[whichind].child1, tree[whichind].child2,
                                 tree[whichind].child3, newparent,
                                 tree[whichind].splitkey1,
                                 tree[whichind].splitkey2)
    nothing
end
function push_or_reuse!(a::Vector, freelocs::Array{Int,1}, item)
    if isempty(freelocs)
        push!(a, item)
        return length(a)
    end
    loc = pop!(freelocs)
    a[loc] = item
    return loc
end
function insert!(t::BalancedTree23{K,D,Ord}, k, d, allowdups::Bool) where {K,D,Ord <: Ordering}
    ## First we find the greatest data node that is <= k.
    leafind, exactfound = findkey(t, k)
    parent = t.data[leafind].parent
    ## The following code is necessary because in the case of a
    ## brand new tree, the initial tree and data entries were incompletely
    ## initialized by the constructor.  In this case, the call to insert!
    ## underway carries
    ## valid K and D values, so these valid values may now be
    ## stored in the dummy placeholder nodes so that they no
    ## longer hold undefined references.
    if size(t.data,1) == 2
        # @assert(t.rootloc == 1 && t.depth == 1)
        t.tree[1] = TreeNode{K}(t.tree[1].child1, t.tree[1].child2,
                                t.tree[1].child3, t.tree[1].parent,
                                k, k)
        t.data[1] = KDRec{K,D}(t.data[1].parent, k, d)
        t.data[2] = KDRec{K,D}(t.data[2].parent, k, d)
    end
    ## If we have found exactly k in the tree, then we
    ## replace the data associated with k and return.
    if exactfound && !allowdups
        t.data[leafind] = KDRec{K,D}(parent, k,d)
        return false, leafind
    end
    # We get here if k was not already found in the tree or
    # if duplicates are allowed.
    # In this case we insert a new node.
    depth = t.depth
    ord = t.ord
    ## Store the new data item in the tree's data array.  Later
    ## go back and fix the parent.
    newind = push_or_reuse!(t.data, t.freedatainds, KDRec{K,D}(0,k,d))
    p1 = parent
    oldchild = leafind
    newchild = newind
    minkeynewchild = k
    splitroot = false
    curdepth = depth
    ## This loop ascends the tree (i.e., follows the path from a leaf to the root)
    ## starting from the parent p1 of
    ## where the new key k would go.  For each 3-node we encounter
    ## during the ascent, we add a new child, which requires splitting
    ## the 3-node into two 2-nodes.  Then we keep going until we hit the root.
    ## If we encounter a 2-node, then the ascent can stop; we can
    ## change the 2-node to a 3-node with the new child. Invariants
    ## during this loop are:
    ##     p1: the parent node (a tree node index) where the insertion must occur
    ##     oldchild,newchild: the two children of the parent node; oldchild
    ##          was already in the tree; newchild was just added to it.
    ##     minkeynewchild:  This is the key that is the minimum value in
    ##         the subtree rooted at newchild.
    while t.tree[p1].child3 > 0
        isleaf = (curdepth == depth)
        oldtreenode = t.tree[p1]
        ## Node p1 index a 3-node. There are three cases for how to
        ## insert new child.  All three cases involve splitting the
        ## existing node (oldtreenode, numbered p1) into
        ## two new nodes.  One keeps the index p1; the other has
        ## has a new index called newparentnum.
        cmp = isleaf ? cmp3_leaf(ord, oldtreenode, minkeynewchild) :
                      cmp3_nonleaf(ord, oldtreenode, minkeynewchild)
        if cmp == 1
            lefttreenodenew = TreeNode{K}(oldtreenode.child1, newchild, 0,
                                          oldtreenode.parent,
                                          minkeynewchild, minkeynewchild)
            righttreenodenew = TreeNode{K}(oldtreenode.child2, oldtreenode.child3, 0,
                                           oldtreenode.parent, oldtreenode.splitkey2,
                                           oldtreenode.splitkey2)
            minkeynewchild = oldtreenode.splitkey1
            whichp = 1
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
        # Replace p1 with a new 2-node and insert another 2-node at
        # index newparentnum.
        t.tree[p1] = lefttreenodenew
        newparentnum = push_or_reuse!(t.tree, t.freetreeinds, righttreenodenew)
        if isleaf
            par = (whichp == 1) ? p1 : newparentnum
            # fix the parent of the new datanode.
            replaceparent!(t.data, newind, par)
            push!(t.useddatacells, newind)
            replaceparent!(t.data, righttreenodenew.child1, newparentnum)
            replaceparent!(t.data, righttreenodenew.child2, newparentnum)
        else
            # If this is not a leaf, we still have to fix the
            # parent fields of the two nodes that are now children
            ## of the newparent.
            replaceparent!(t.tree, righttreenodenew.child1, newparentnum)
            replaceparent!(t.tree, righttreenodenew.child2, newparentnum)
        end
        oldchild = p1
        newchild = newparentnum
        ## If p1 is the root (i.e., we have encountered only 3-nodes during
        ## our ascent of the tree), then the root must be split.
        if p1 == t.rootloc
            # @assert(curdepth == 1)
            splitroot = true
            break
        end
        p1 = t.tree[oldchild].parent
        curdepth -= 1
    end
    ## big loop terminated either because a 2-node was reached
    ## (splitroot == false) or we went up the whole tree seeing
    ## only 3-nodes (splitroot == true).
    if !splitroot
        ## If our ascent reached a 2-node, then we convert it to
        ## a 3-node by giving it a child3 field that is >0.
        ## Encountering a 2-node halts the ascent up the tree.
        isleaf = curdepth == depth
        oldtreenode = t.tree[p1]
        cmpres = isleaf ? cmp2_leaf(ord, oldtreenode, minkeynewchild) :
                         cmp2_nonleaf(ord, oldtreenode, minkeynewchild)
        t.tree[p1] = cmpres == 1 ?
                         TreeNode{K}(oldtreenode.child1, newchild, oldtreenode.child2,
                                     oldtreenode.parent,
                                     minkeynewchild, oldtreenode.splitkey1) :
                         TreeNode{K}(oldtreenode.child1, oldtreenode.child2, newchild,
                                     oldtreenode.parent,
                                     oldtreenode.splitkey1, minkeynewchild)
        if isleaf
            replaceparent!(t.data, newind, p1)
            push!(t.useddatacells, newind)
        end
    else
        ## Splitroot is set if the ascent of the tree encountered only 3-nodes.
        ## In this case, the root itself was replaced by two nodes, so we need
        ## a new root above those two.
        newroot = TreeNode{K}(oldchild, newchild, 0, 0,
                              minkeynewchild, minkeynewchild)
        newrootloc = push_or_reuse!(t.tree, t.freetreeinds, newroot)
        replaceparent!(t.tree, oldchild, newrootloc)
        replaceparent!(t.tree, newchild, newrootloc)
        t.rootloc = newrootloc
        t.depth += 1
    end
    true, newind
end
function nextloc0(t, i::Int)
    ii = i
    # @assert(i != 2 && i in t.useddatacells)
    @inbounds p = t.data[i].parent
    nextchild = 0
    depthp = t.depth
    @inbounds while true
        if depthp < t.depth
            p = t.tree[ii].parent
        end
        if t.tree[p].child1 == ii
            nextchild = t.tree[p].child2
            break
        end
        if t.tree[p].child2 == ii && t.tree[p].child3 > 0
            nextchild = t.tree[p].child3
            break
        end
        ii = p
        depthp -= 1
    end
    @inbounds while true
        if depthp == t.depth
            return nextchild
        end
        p = nextchild
        nextchild = t.tree[p].child1
        depthp += 1
    end
end
function prevloc0(t::BalancedTree23, i::Int)
    # @assert(i != 1 && i in t.useddatacells)
    ii = i
    @inbounds p = t.data[i].parent
    prevchild = 0
    depthp = t.depth
    @inbounds while true
        if depthp < t.depth
            p = t.tree[ii].parent
        end
        if t.tree[p].child3 == ii
            prevchild = t.tree[p].child2
            break
        end
        if t.tree[p].child2 == ii
            prevchild = t.tree[p].child1
            break
        end
        ii = p
        depthp -= 1
    end
    @inbounds while true
        if depthp == t.depth
            return prevchild
        end
        p = prevchild
        c3 = t.tree[p].child3
        prevchild = c3 > 0 ? c3 : t.tree[p].child2
        depthp += 1
    end
end
function compareInd(t::BalancedTree23, i1::Int, i2::Int)
    @assert(i1 in t.useddatacells && i2 in t.useddatacells)
    if i1 == i2
        return 0
    end
    i1a = i1
    i2a = i2
    p1 = t.data[i1].parent
    p2 = t.data[i2].parent
    curdepth = t.depth
    while true
        @assert(curdepth > 0)
        if p1 == p2
            if i1a == t.tree[p1].child1
                @assert(t.tree[p1].child2 == i2a || t.tree[p1].child3 == i2a)
                return -1
            end
            if i1a == t.tree[p1].child2
                if (t.tree[p1].child1 == i2a)
                    return 1
                end
                @assert(t.tree[p1].child3 == i2a)
                return -1
            end
            @assert(i1a == t.tree[p1].child3)
            @assert(t.tree[p1].child1 == i2a || t.tree[p1].child2 == i2a)
            return 1
        end
        i1a = p1
        i2a = p2
        p1 = t.tree[i1a].parent
        p2 = t.tree[i2a].parent
        curdepth -= 1
    end
end
beginloc(t::BalancedTree23) = nextloc0(t,1)
endloc(t::BalancedTree23) = prevloc0(t,2)
function delete!(t::BalancedTree23{K,D,Ord}, it::Int) where {K,D,Ord<:Ordering}
    ## Put the cell indexed by 'it' into the deletion list.
    ##
    ## Create the following data items maintained in the
    ## upcoming loop.
    ##
    ## p is a tree-node ancestor of the deleted node
    ## The children of p are stored in
    ## t.deletionchild[..]
    ## The number of these children is newchildcount, which is 1, 2 or 3.
    ## The keys that lower bound the children
    ## are stored in t.deletionleftkey[..]
    ## There is a special case for t.deletionleftkey[1]; the
    ## flag deletionleftkey1_valid indicates that the left key
    ## for the immediate right neighbor of the
    ## deleted node has not yet been been stored in the tree.
    ## Once it is stored, t.deletionleftkey[1] is no longer needed
    ## or used.
    ## The flag mustdeleteroot means that the tree has contracted
    ## enough that it loses a level.
    p = t.data[it].parent
    newchildcount = 0
    c1 = t.tree[p].child1
    deletionleftkey1_valid = true
    if c1 != it
        deletionleftkey1_valid = false
        newchildcount += 1
        t.deletionchild[newchildcount] = c1
        t.deletionleftkey[newchildcount] = t.data[c1].k
    end
    c2 = t.tree[p].child2
    if c2 != it
        newchildcount += 1
        t.deletionchild[newchildcount] = c2
        t.deletionleftkey[newchildcount] = t.data[c2].k
    end
    c3 = t.tree[p].child3
    if c3 != it && c3 > 0
        newchildcount += 1
        t.deletionchild[newchildcount] = c3
        t.deletionleftkey[newchildcount] = t.data[c3].k
    end
    # @assert(newchildcount == 1 || newchildcount == 2)
    push!(t.freedatainds, it)
    pop!(t.useddatacells,it)
    defaultKey = t.tree[1].splitkey1
    curdepth = t.depth
    mustdeleteroot = false
    pparent = -1
    ## The following loop ascends the tree and contracts nodes (reduces their
    ## number of children) as
    ## needed.  If newchildcount == 2 or 3, then the ascent is terminated
    ## and a node is created with 2 or 3 children.
    ## If newchildcount == 1, then the ascent must continue since a tree
    ## node cannot have one child.
    while true
        pparent = t.tree[p].parent
        ## Simple cases when the new child count is 2 or 3
        if newchildcount == 2
            t.tree[p] = TreeNode{K}(t.deletionchild[1],
                                    t.deletionchild[2], 0, pparent,
                                    t.deletionleftkey[2], defaultKey)
            break
        end
        if newchildcount == 3
            t.tree[p] = TreeNode{K}(t.deletionchild[1], t.deletionchild[2],
                                    t.deletionchild[3], pparent,
                                    t.deletionleftkey[2], t.deletionleftkey[3])
            break
        end
        # @assert(newchildcount == 1)
        ## For the rest of this loop, we cover the case
        ## that p has one child.
        ## If newchildcount == 1 and curdepth==1, this means that
        ## the root of the tree has only one child.  In this case, we can
        ## delete the root and make its one child the new root (see below).
        if curdepth == 1
            mustdeleteroot = true
            break
        end
        ## We now branch on three cases depending on whether p is child1,
        ## child2 or child3 of its parent.
        if t.tree[pparent].child1 == p
            rightsib = t.tree[pparent].child2
            ## Here p is child1 and rightsib is child2.
            ## If rightsib has 2 children, then p and
            ## rightsib are merged into a single node
            ## that has three children.
            ## If rightsib has 3 children, then p and
            ## rightsib are reformed so that each has
            ## two children.
            if t.tree[rightsib].child3 == 0
                rc1 = t.tree[rightsib].child1
                rc2 = t.tree[rightsib].child2
                t.tree[p] = TreeNode{K}(t.deletionchild[1],
                                        rc1, rc2,
                                        pparent,
                                        t.tree[pparent].splitkey1,
                                        t.tree[rightsib].splitkey1)
                if curdepth == t.depth
                    replaceparent!(t.data, rc1, p)
                    replaceparent!(t.data, rc2, p)
                else
                    replaceparent!(t.tree, rc1, p)
                    replaceparent!(t.tree, rc2, p)
                end
                push!(t.freetreeinds, rightsib)
                newchildcount = 1
                t.deletionchild[1] = p
            else
                rc1 = t.tree[rightsib].child1
                t.tree[p] = TreeNode{K}(t.deletionchild[1], rc1, 0,
                                        pparent,
                                        t.tree[pparent].splitkey1,
                                        defaultKey)
                sk1 = t.tree[rightsib].splitkey1
                t.tree[rightsib] = TreeNode{K}(t.tree[rightsib].child2,
                                               t.tree[rightsib].child3,
                                               0,
                                               pparent,
                                               t.tree[rightsib].splitkey2,
                                               defaultKey)
                if curdepth == t.depth
                    replaceparent!(t.data, rc1, p)
                else
                    replaceparent!(t.tree, rc1, p)
                end
                newchildcount = 2
                t.deletionchild[1] = p
                t.deletionchild[2] = rightsib
                t.deletionleftkey[2] = sk1
            end
            ## If pparent had a third child (besides p and rightsib)
            ## then we add this to t.deletionchild
            c3 = t.tree[pparent].child3
            if c3 > 0
                newchildcount += 1
                t.deletionchild[newchildcount] = c3
                t.deletionleftkey[newchildcount] = t.tree[pparent].splitkey2
            end
            p = pparent
        elseif t.tree[pparent].child2 == p
            ## Here p is child2 and leftsib is child1.
            ## If leftsib has 2 children, then p and
            ## leftsib are merged into a single node
            ## that has three children.
            ## If leftsib has 3 children, then p and
            ## leftsib are reformed so that each has
            ## two children.
            leftsib = t.tree[pparent].child1
            lk = deletionleftkey1_valid ?
                      t.deletionleftkey[1] :
                      t.tree[pparent].splitkey1
            if t.tree[leftsib].child3 == 0
                lc1 = t.tree[leftsib].child1
                lc2 = t.tree[leftsib].child2
                t.tree[p] = TreeNode{K}(lc1, lc2,
                                        t.deletionchild[1],
                                        pparent,
                                        t.tree[leftsib].splitkey1,
                                        lk)
                if curdepth == t.depth
                    replaceparent!(t.data, lc1, p)
                    replaceparent!(t.data, lc2, p)
                else
                    replaceparent!(t.tree, lc1, p)
                    replaceparent!(t.tree, lc2, p)
                end
                push!(t.freetreeinds, leftsib)
                newchildcount = 1
                t.deletionchild[1] = p
            else
                lc3 = t.tree[leftsib].child3
                t.tree[p] = TreeNode{K}(lc3, t.deletionchild[1], 0,
                                        pparent, lk, defaultKey)
                sk2 = t.tree[leftsib].splitkey2
                t.tree[leftsib] = TreeNode{K}(t.tree[leftsib].child1,
                                              t.tree[leftsib].child2,
                                              0, pparent,
                                              t.tree[leftsib].splitkey1,
                                              defaultKey)
                if curdepth == t.depth
                    replaceparent!(t.data, lc3, p)
                else
                    replaceparent!(t.tree, lc3, p)
                end
                newchildcount = 2
                t.deletionchild[1] = leftsib
                t.deletionchild[2] = p
                t.deletionleftkey[2] = sk2
            end
            ## If pparent had a third child (besides p and leftsib)
            ## then we add this to t.deletionchild
            c3 = t.tree[pparent].child3
            if c3 > 0
                newchildcount += 1
                t.deletionchild[newchildcount] = c3
                t.deletionleftkey[newchildcount] = t.tree[pparent].splitkey2
            end
            p = pparent
            deletionleftkey1_valid = false
        else
            ## Here p is child3 and leftsib is child2.
            ## If leftsib has 2 children, then p and
            ## leftsib are merged into a single node
            ## that has three children.
            ## If leftsib has 3 children, then p and
            ## leftsib are reformed so that each has
            ## two children.
            # @assert(t.tree[pparent].child3 == p)
            leftsib = t.tree[pparent].child2
            lk = deletionleftkey1_valid ?
                       t.deletionleftkey[1] :
                       t.tree[pparent].splitkey2
            if t.tree[leftsib].child3 == 0
                lc1 = t.tree[leftsib].child1
                lc2 = t.tree[leftsib].child2
                t.tree[p] = TreeNode{K}(lc1, lc2,
                                        t.deletionchild[1],
                                        pparent,
                                        t.tree[leftsib].splitkey1,
                                        lk)
                if curdepth == t.depth
                    replaceparent!(t.data, lc1, p)
                    replaceparent!(t.data, lc2, p)
                else
                    replaceparent!(t.tree, lc1, p)
                    replaceparent!(t.tree, lc2, p)
                end
                push!(t.freetreeinds, leftsib)
                newchildcount = 2
                t.deletionchild[1] = t.tree[pparent].child1
                t.deletionleftkey[2] = t.tree[pparent].splitkey1
                t.deletionchild[2] = p
            else
                lc3 = t.tree[leftsib].child3
                t.tree[p] = TreeNode{K}(lc3, t.deletionchild[1], 0,
                                        pparent, lk, defaultKey)
                sk2 = t.tree[leftsib].splitkey2
                t.tree[leftsib] = TreeNode{K}(t.tree[leftsib].child1,
                                              t.tree[leftsib].child2,
                                              0, pparent,
                                              t.tree[leftsib].splitkey1,
                                              defaultKey)
                if curdepth == t.depth
                    replaceparent!(t.data, lc3, p)
                else
                    replaceparent!(t.tree, lc3, p)
                end
                newchildcount = 3
                t.deletionchild[1] = t.tree[pparent].child1
                t.deletionchild[2] = leftsib
                t.deletionchild[3] = p
                t.deletionleftkey[2] = t.tree[pparent].splitkey1
                t.deletionleftkey[3] = sk2
            end
            p = pparent
            deletionleftkey1_valid = false
        end
        curdepth -= 1
    end
    if mustdeleteroot
        # @assert(!deletionleftkey1_valid)
        # @assert(p == t.rootloc)
        t.rootloc = t.deletionchild[1]
        t.depth -= 1
        push!(t.freetreeinds, p)
    end
    ## If deletionleftkey1_valid, this means that the new
    ## min key of the deleted node and its right neighbors
    ## has never been stored in the tree.  It must be stored
    ## as splitkey1 or splitkey2 of some ancestor of the
    ## deleted node, so we continue ascending the tree
    ## until we find a node which has p (and therefore the
    ## deleted node) as its descendent through its second
    ## or third child.
    ## It cannot be the case that the deleted node is
    ## is a descendent of the root always through
    ## first children, since this would mean the deleted
    ## node is the leftmost placeholder, which
    ## cannot be deleted.
    if deletionleftkey1_valid
        while true
            pparentnode = t.tree[pparent]
            if pparentnode.child2 == p
                t.tree[pparent] = TreeNode{K}(pparentnode.child1,
                                              pparentnode.child2,
                                              pparentnode.child3,
                                              pparentnode.parent,
                                              t.deletionleftkey[1],
                                              pparentnode.splitkey2)
                break
            elseif pparentnode.child3 == p
                t.tree[pparent] = TreeNode{K}(pparentnode.child1,
                                              pparentnode.child2,
                                              pparentnode.child3,
                                              pparentnode.parent,
                                              pparentnode.splitkey1,
                                              t.deletionleftkey[1])
                break
            else
                p = pparent
                pparent = pparentnode.parent
                curdepth -= 1
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
             insert!, getindex, length, isempty, start,
             next, done, keys, values, copy, similar,  push!,
             count, size, eltype
struct MultiDict{K,V}
    d::Dict{K,Vector{V}}
    MultiDict{K,V}() where {K,V} = new{K,V}(Dict{K,Vector{V}}())
    MultiDict{K,V}(kvs) where {K,V} = new{K,V}(Dict{K,Vector{V}}(kvs))
    MultiDict{K,V}(ps::Pair{K,Vector{V}}...) where {K,V} = new{K,V}(Dict{K,Vector{V}}(ps...))
end
MultiDict() = MultiDict{Any,Any}()
MultiDict(kv::Tuple{}) = MultiDict()
MultiDict(kvs) = multi_dict_with_eltype(kvs, eltype(kvs))
multi_dict_with_eltype(kvs, ::Type{Tuple{K,Vector{V}}}) where {K,V} = MultiDict{K,V}(kvs)
function multi_dict_with_eltype(kvs, ::Type{Tuple{K,V}}) where {K,V}
    md = MultiDict{K,V}()
    for (k,v) in kvs
        insert!(md, k, v)
    end
    return md
end
multi_dict_with_eltype(kvs, t) = MultiDict{Any,Any}(kvs)
MultiDict(ps::Pair{K,V}...) where {K,V<:AbstractArray} = MultiDict{K, eltype(V)}(ps)
MultiDict(kv::AbstractArray{Pair{K,V}}) where {K,V}  = MultiDict(kv...)
function MultiDict(ps::Pair{K,V}...) where {K,V}
    md = MultiDict{K,V}()
    for (k,v) in ps
        insert!(md, k, v)
    end
    return md
end
@delegate MultiDict.d [ haskey, get, get!, getkey,
                        getindex, length, isempty, eltype,
                        start, next, done, keys, values]
sizehint!(d::MultiDict, sz::Integer) = (sizehint!(d.d, sz); d)
copy(d::MultiDict) = MultiDict(d)
similar(d::MultiDict{K,V}) where {K,V} = MultiDict{K,V}()
==(d1::MultiDict, d2::MultiDict) = d1.d == d2.d
delete!(d::MultiDict, key) = (delete!(d.d, key); d)
empty!(d::MultiDict) = (empty!(d.d); d)
function insert!(d::MultiDict{K,V}, k, v) where {K,V}
    if !haskey(d.d, k)
        d.d[k] = isa(v, AbstractArray) ? eltype(v)[] : V[]
    end
    if isa(v, AbstractArray)
        append!(d.d[k], v)
    else
        push!(d.d[k], v)
    end
    return d
end
function in(pr::(Tuple{Any,Any}), d::MultiDict{K,V}) where {K,V}
    k = convert(K, pr[1])
    v = get(d,k,Base.secret_table_token)
    (v !== Base.secret_table_token) && (isa(pr[2], AbstractArray) ? v == pr[2] : pr[2] in v)
end
function pop!(d::MultiDict, key, default)
    vs = get(d, key, Base.secret_table_token)
    if vs === Base.secret_table_token
        if default !== Base.secret_table_token
            return default
        else
            throw(KeyError(key))
        end
    end
    v = pop!(vs)
    (length(vs) == 0) && delete!(d, key)
    return v
end
pop!(d::MultiDict, key) = pop!(d, key, Base.secret_table_token)
push!(d::MultiDict, kv::Pair) = insert!(d, kv[1], kv[2])
push!(d::MultiDict, kv) = insert!(d, kv[1], kv[2])
count(d::MultiDict) = length(keys(d)) == 0 ? 0 : mapreduce(k -> length(d[k]), +, keys(d))
size(d::MultiDict) = (length(keys(d)), count(d::MultiDict))
struct EnumerateAll
    d::MultiDict
end
enumerateall(d::MultiDict) = EnumerateAll(d)
length(e::EnumerateAll) = count(e.d)
function start(e::EnumerateAll)
    V = eltype(eltype(values(e.d)))
    vs = V[]
    (start(e.d.d), nothing, vs, start(vs))
end
function done(e::EnumerateAll, s)
    dst, k, vs, vst = s
    done(vs, vst) && done(e.d.d, dst)
end
function next(e::EnumerateAll, s)
    dst, k, vs, vst = s
    while done(vs, vst)
        ((k, vs), dst) = next(e.d.d, dst)
        vst = start(vs)
    end
    v, vst = next(vs, vst)
    ((k, v), (dst, k, vs, vst))
end
mutable struct SortedDict{K, D, Ord <: Ordering} <: AbstractDict{K,D}
    bt::BalancedTree23{K,D,Ord}
    ## Base constructors
    """
        SortedDict{K,V}(o=Forward)
    Construct an empty `SortedDict` with key type `K` and value type
    `V` with `o` ordering (default to forward ordering).
    """
    SortedDict{K,D,Ord}(o::Ord) where {K, D, Ord <: Ordering} =
        new{K,D,Ord}(BalancedTree23{K,D,Ord}(o))
    function SortedDict{K,D,Ord}(o::Ord, kv) where {K, D, Ord <: Ordering}
        s = new{K,D,Ord}(BalancedTree23{K,D,Ord}(o))
        if eltype(kv) <: Pair
            # It's (possibly?) more efficient to access the first and second
            # elements of Pairs directly, rather than destructure
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
"""
    SortedDict()
Construct an empty `SortedDict` with key type `Any` and value type
`Any`. Ordering defaults to `Forward` ordering.
**Note that a key type of `Any` or any other abstract type will lead
to slow performance, as the values are stored boxed (i.e., as
pointers), and insertion will require a run-time lookup of the
appropriate comparison function. It is recommended to always specify
a concrete key type, or to use one of the constructors below in
which the key type is inferred.**
"""
SortedDict() = SortedDict{Any,Any,ForwardOrdering}(Forward)
"""
    SortedDict(o=Forward)
Construct an empty `SortedDict` with key type `K` and value type
`V`. If `K` and `V` are not specified, the dictionary defaults to a
`SortedDict{Any,Any}`. Keys and values are converted to the given
type upon insertion. Ordering `o` defaults to `Forward` ordering.
**Note that a key type of `Any` or any other abstract type will lead
to slow performance, as the values are stored boxed (i.e., as
pointers), and insertion will require a run-time lookup of the
appropriate comparison function. It is recommended to always specify
a concrete key type, or to use one of the constructors below in
which the key type is inferred.**
"""
SortedDict(o::Ord) where {Ord <: Ordering} = SortedDict{Any,Any,Ord}(o)
"""
    SortedDict(k1=>v1, k2=>v2, ...)
and `SortedDict{K,V}(k1=>v1, k2=>v2, ...)`
Construct a `SortedDict` from the given key-value pairs. If `K` and
`V` are not specified, key type and value type are inferred from the
given key-value pairs, and ordering is assumed to be `Forward`
ordering.
"""
SortedDict(ps::Pair...) = SortedDict(Forward, ps)
SortedDict{K,D}(ps::Pair...) where {K,D} = SortedDict{K,D,ForwardOrdering}(Forward, ps)
"""
    SortedDict(o, k1=>v1, k2=>v2, ...)
Construct a `SortedDict` from the given pairs with the specified
ordering `o`. The key type and value type are inferred from the 
given pairs.
"""
SortedDict(o::Ordering, ps::Pair...) = SortedDict(o, ps)
"""
    SortedDict{K,V}(o, k1=>v1, k2=>v2, ...)
Construct a `SortedDict` from the given pairs with the specified
ordering `o`. If `K` and `V` are not specified, the key type and
value type are inferred from the given pairs. See below for more
information about ordering.
"""
SortedDict{K,D}(o::Ord, ps::Pair...) where {K,D,Ord<:Ordering} = SortedDict{K,D,Ord}(o, ps)
SortedDict(o::Ord, d::AbstractDict{K,D}) where {K,D,Ord<:Ordering} = SortedDict{K,D,Ord}(o, d)
"""
    SortedDict(iter, o=Forward)
and `SortedDict{K,V}(iter, o=Forward)`
Construct a `SortedDict` from an arbitrary iterable object of
`key=>value` pairs. If `K` and `V` are not specified, the key type
and value type are inferred from the given iterable. The ordering
object `o` defaults to `Forward`.
"""
SortedDict{K,D}(kv) where {K,D} = SortedDict{K,D}(Forward, kv)
function SortedDict{K,D}(o::Ord, kv) where {K,D,Ord<:Ordering}
    try
        SortedDict{K,D,Ord}(o, kv)
    catch e
        if not_iterator_of_pairs(kv)
            throw(ArgumentError("SortedDict(kv): kv needs to be an iterator of tuples or pairs"))
        else
            rethrow(e)
        end
    end
end
SortedDict(o1::Ordering, o2::Ordering) = throw(ArgumentError("SortedDict with two parameters must be called with an Ordering and an interable of pairs"))
"""
    SortedDict(d, o=Forward)
and `SortedDict{K,V}(d, o=Forward)`
Construct a `SortedDict` from an ordinary Julia dict `d` (or any
associative type), e.g.:
```julia
d = Dict("New York" => 1788, "Illinois" => 1818)
c = SortedDict(d)
```
In this example the key-type is deduced to be `String`, while the
value-type is `Int`.
If `K` and `V` are not specified, the key type and value type are
inferred from the given dictionary. The ordering object `o` defaults
to `Forward`.
"""
SortedDict(kv, o::Ordering=Forward) = SortedDict(o, kv)
function SortedDict(o::Ordering, kv)
    try
        _sorted_dict_with_eltype(o, kv, eltype(kv))
    catch e
        if not_iterator_of_pairs(kv)
            throw(ArgumentError("SortedDict(kv): kv needs to be an iterator of tuples or pairs"))
        else
            rethrow(e)
        end
    end
end
_sorted_dict_with_eltype(o::Ord, ps, ::Type{Pair{K,D}}) where {K,D,Ord} = SortedDict{  K,  D,Ord}(o, ps)
_sorted_dict_with_eltype(o::Ord, kv, ::Type{Tuple{K,D}}) where {K,D,Ord} = SortedDict{  K,  D,Ord}(o, kv)
_sorted_dict_with_eltype(o::Ord, ps, ::Type{Pair{K}}  ) where {K,  Ord} = SortedDict{  K,Any,Ord}(o, ps)
_sorted_dict_with_eltype(o::Ord, kv, ::Type            ) where {    Ord} = SortedDict{Any,Any,Ord}(o, kv)
const SDSemiToken = IntSemiToken
const SDToken = Tuple{SortedDict,IntSemiToken}
"""
    v = sd[k]
Argument `sd` is a SortedDict and `k` is a key. In an expression,
this retrieves the value (`v`) associated with the key (or `KeyError` if
none). On the left-hand side of an assignment, this assigns or
reassigns the value associated with the key. (For assigning and
reassigning, see also `insert!` below.) Time: O(*c* log *n*)
"""
@inline function getindex(m::SortedDict, k_)
    i, exactfound = findkey(m.bt, convert(keytype(m),k_))
    !exactfound && throw(KeyError(k_))
    return m.bt.data[i].d
end
"""
    sc[st] = v
If `st` is a semitoken and `sc` is a SortedDict or SortedMultiDict,
then `sc[st]` refers to the value field of the (key,value) pair that
the full token `(sc,st)` refers to. This expression may occur on
either side of an assignment statement. Time: O(1)
"""
@inline function setindex!(m::SortedDict{K,D,Ord}, d_, k_) where {K, D, Ord <: Ordering}
    insert!(m.bt, convert(K,k_), convert(D,d_), false)
    m
end
"""
    push!(sc, k=>v)
Argument `sc` is a SortedDict or SortedMultiDict and `k=>v` is a
key-value pair. This inserts the key-value pair into the container.
If the key is already present, this overwrites the old value. The
return value is `sc`. Time: O(*c* log *n*)
"""
@inline function push!(m::SortedDict{K,D}, pr::Pair) where {K,D}
    insert!(m.bt, convert(K, pr[1]), convert(D, pr[2]), false)
    m
end
"""
    find(sd, k)
Argument `sd` is a SortedDict and argument `k` is a key. This
function returns the semitoken that refers to the item whose key is
`k`, or past-end semitoken if `k` is absent. Time: O(*c* log *n*)
"""
@inline function find(m::SortedDict, k_)
    ll, exactfound = findkey(m.bt, convert(keytype(m),k_))
    IntSemiToken(exactfound ? ll : 2)
end
"""
    insert!(sc, k)
Argument `sc` is a SortedDict or SortedMultiDict, `k` is a key and
`v` is the corresponding value. This inserts the `(k,v)` pair into
the container. If the key is already present in a SortedDict, this
overwrites the old value. In the case of SortedMultiDict, no
overwriting takes place (since SortedMultiDict allows the same key
to associate with multiple values). In the case of SortedDict, the
return value is a pair whose first entry is boolean and indicates
whether the insertion was new (i.e., the key was not previously
present) and the second entry is the semitoken of the new entry. In
the case of SortedMultiDict, a semitoken is returned (but no
boolean). Time: O(*c* log *n*)
"""
@inline function insert!(m::SortedDict{K,D,Ord}, k_, d_) where {K,D, Ord <: Ordering}
    b, i = insert!(m.bt, convert(K,k_), convert(D,d_), false)
    b, IntSemiToken(i)
end
"""
    eltype(sc)
Returns the (key,value) type (a 2-entry pair, i.e., `Pair{K,V}`) for
SortedDict and SortedMultiDict. Returns the key type for SortedSet.
This function may also be applied to the type itself. Time: O(1)
"""
@inline eltype(m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering} =  Pair{K,D}
@inline eltype(::Type{SortedDict{K,D,Ord}}) where {K,D,Ord <: Ordering} =  Pair{K,D}
"""
    in(p, sc)
Returns true if `p` is in `sc`. In the case that `sc` is a
SortedDict or SortedMultiDict, `p` is a key=>value pair. In the
case that `sc` is a SortedSet, `p` should be a key. Time: O(*c* log
*n* + *d*) for SortedDict and SortedSet, where *d* stands for the
time to compare two values. In the case of SortedMultiDict, the time
is O(*c* log *n* + *dl*), and *l* stands for the number of entries
that have the key of the given pair. (So therefore this call is
inefficient if the same key addresses a large number of values, and
an alternative should be considered.)
"""
@inline function in(pr::Pair, m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering}
    i, exactfound = findkey(m.bt,convert(K,pr[1]))
    return exactfound && isequal(m.bt.data[i].d,convert(D,pr[2]))
end
@inline in(::Tuple{Any,Any}, ::SortedDict) =
    throw(ArgumentError("'(k,v) in sorteddict' not supported in Julia 0.4 or 0.5.  See documentation"))
"""
    keytype(sc)
Returns the key type for SortedDict, SortedMultiDict and SortedSet.
This function may also be applied to the type itself. Time: O(1)
"""
@inline keytype(m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering} = K
@inline keytype(::Type{SortedDict{K,D,Ord}}) where {K,D,Ord <: Ordering} = K
"""
    valtype(sc)
Returns the value type for SortedDict and SortedMultiDict. This
function may also be applied to the type itself. Time: O(1)
"""
@inline valtype(m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering} = D
@inline valtype(::Type{SortedDict{K,D,Ord}}) where {K,D,Ord <: Ordering} = D
"""
    ordtype(sc)
Returns the order type for SortedDict, SortedMultiDict and
SortedSet. This function may also be applied to the type itself.
Time: O(1)
"""
@inline ordtype(m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering} = Ord
@inline ordtype(::Type{SortedDict{K,D,Ord}}) where {K,D,Ord <: Ordering} = Ord
"""
    orderobject(sc)
Returns the order object used to construct the container. Time: O(1)
"""
@inline orderobject(m::SortedDict) = m.bt.ord
"""
    first(sc)
Argument `sc` is a SortedDict, SortedMultiDict or SortedSet. This
function returns the first item (a `k=>v` pair for SortedDict and
SortedMultiDict or a key for SortedSet) according to the sorted
order in the container. Thus, `first(sc)` is equivalent to
`deref((sc,startof(sc)))`. It is an error to call this function on
an empty container. Time: O(log *n*)
"""
@inline function first(m::SortedDict)
    i = beginloc(m.bt)
    i == 2 && throw(BoundsError())
    return Pair(m.bt.data[i].k, m.bt.data[i].d)
end
"""
    last(sc)
Argument `sc` is a SortedDict, SortedMultiDict or SortedSet. This
function returns the last item (a `k=>v` pair for SortedDict and
SortedMultiDict or a key for SortedSet) according to the sorted
order in the container. Thus, `last(sc)` is equivalent to
`deref((sc,endof(sc)))`. It is an error to call this function on an
empty container. Time: O(log *n*)
"""
@inline function last(m::SortedDict)
    i = endloc(m.bt)
    i == 1 && throw(BoundsError())
    return Pair(m.bt.data[i].k, m.bt.data[i].d)
end
"""
    haskey(sc,k)
Returns true if key `k` is present for SortedDict, SortedMultiDict
or SortedSet `sc`. For SortedSet, `haskey(sc,k)` is a synonym for
`in(k,sc)`. For SortedDict and SortedMultiDict, `haskey(sc,k)` is
equivalent to `in(k,keys(sc))`. Time: O(*c* log *n*)
"""
@inline function haskey(m::SortedDict, k_)
    i, exactfound = findkey(m.bt, convert(keytype(m), k_))
    exactfound
end
"""
    get(sd,k,v)
Returns the value associated with key `k` where `sd` is a
SortedDict, or else returns `v` if `k` is not in `sd`. Time: O(*c*
log *n*)
"""
function get(default_::Union{Function,Type}, m::SortedDict{K,D}, k_) where {K,D}
    i, exactfound = findkey(m.bt, convert(K, k_))
    return exactfound ? m.bt.data[i].d : convert(D, default_())
end
get(m::SortedDict, k_, default_) = get(()->default_, m, k_)
"""
    get!(sd,k,v)
Returns the value associated with key `k` where `sd` is a
SortedDict, or else returns `v` if `k` is not in `sd`, and in the
latter case, inserts `(k,v)` into `sd`. Time: O(*c* log *n*)
"""
function get!(default_::Union{Function,Type}, m::SortedDict{K,D}, k_) where {K,D}
    k = convert(K,k_)
    i, exactfound = findkey(m.bt, k)
    if exactfound
        return m.bt.data[i].d
    else
        default = convert(D, default_())
        insert!(m.bt,k, default, false)
        return default
    end
end
get!(m::SortedDict, k_, default_) = get!(()->default_, m, k_)
"""
    getkey(sd,k,defaultk)
Returns key `k` where `sd` is a SortedDict, if `k` is in `sd` else
it returns `defaultk`. If the container uses in its ordering an `eq`
method different from isequal (e.g., case-insensitive ASCII strings
illustrated below), then the return value is the actual key stored
in the SortedDict that is equivalent to `k` according to the `eq`
method, which might not be equal to `k`. Similarly, if the user
performs an implicit conversion as part of the call (e.g., the
container has keys that are floats, but the `k` argument to `getkey`
is an Int), then the returned key is the actual stored key rather
than `k`. Time: O(*c* log *n*)
"""
function getkey(m::SortedDict{K,D,Ord}, k_, default_) where {K,D,Ord <: Ordering}
    i, exactfound = findkey(m.bt, convert(K, k_))
    exactfound ? m.bt.data[i].k : convert(K, default_)
end
"""
    delete!(sc, k)
Argument `sc` is a SortedDict or SortedSet and `k` is a key. This
operation deletes the item whose key is `k`. It is a `KeyError` if
`k` is not a key of an item in the container. After this operation
is complete, any token addressing the deleted item is invalid.
Returns `sc`. Time: O(*c* log *n*)
"""
@inline function delete!(m::SortedDict, k_)
    i, exactfound = findkey(m.bt, convert(keytype(m), k_))
    !exactfound && throw(KeyError(k_))
    delete!(m.bt, i)
    m
end
"""
    pop!(sc, k)
Deletes the item with key `k` in SortedDict or SortedSet `sc` and
returns the value that was associated with `k` in the case of
SortedDict or `k` itself in the case of SortedSet. A `KeyError`
results if `k` is not in `sc`. Time: O(*c* log *n*)
"""
@inline function pop!(m::SortedDict, k_)
    i, exactfound = findkey(m.bt, convert(keytype(m), k_))
    !exactfound && throw(KeyError(k_))
    d = m.bt.data[i].d
    delete!(m.bt, i)
    d
end
"""
    isequal(sc1,sc2)
Checks if two containers are equal in the sense that they contain
the same items; the keys are compared using the `eq` method, while
the values are compared with the `isequal` function. In the case of
SortedMultiDict, equality requires that the values associated with a
particular key have same order (that is, the same insertion order).
Note that `isequal` in this sense does not imply any correspondence
between semitokens for items in `sc1` with those for `sc2`. If the
equality-testing method associated with the keys and values implies
hash-equivalence in the case of SortedDict, then `isequal` of the
entire containers implies hash-equivalence of the containers. Time:
O(*cn* + *n* log *n*)
"""
function isequal(m1::SortedDict, m2::SortedDict)
    ord = orderobject(m1)
    if !isequal(ord, orderobject(m2)) || !isequal(eltype(m1), eltype(m2))
        throw(ArgumentError("Cannot use isequal for two SortedDicts unless their element types and ordering objects are equal"))
    end
    p1 = startof(m1)
    p2 = startof(m2)
    while true
        if p1 == pastendsemitoken(m1)
            return p2 == pastendsemitoken(m2)
        end
        if p2 == pastendsemitoken(m2)
            return false
        end
        k1,d1 = deref((m1,p1))
        k2,d2 = deref((m2,p2))
        if !eq(ord,k1,k2) || !isequal(d1,d2)
            return false
        end
        p1 = advance((m1,p1))
        p2 = advance((m2,p2))
    end
end
function mergetwo!(m::SortedDict{K,D,Ord},
                   m2::AbstractDict{K,D}) where {K,D,Ord <: Ordering}
    for (k,v) in m2
        m[convert(K,k)] = convert(D,v)
    end
end
"""
    packcopy(sc)
This returns a copy of `sc` in which the data is packed. When
deletions take place, the previously allocated memory is not
returned. This function can be used to reclaim memory after many
deletions. Time: O(*cn* log *n*)
"""
function packcopy(m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering}
    w = SortedDict(Dict{K,D}(), orderobject(m))
    mergetwo!(w,m)
    w
end
"""
    packdeepcopy(sc)
This returns a packed copy of `sc` in which the keys and values are
deep-copied. This function can be used to reclaim memory after many
deletions. Time: O(*cn* log *n*)
"""
function packdeepcopy(m::SortedDict{K,D,Ord}) where {K,D,Ord <: Ordering}
    w = SortedDict(Dict{K,D}(),orderobject(m))
    for (k,v) in m
        newk = deepcopy(k)
        newv = deepcopy(v)
        w[newk] = newv
    end
    w
end
"""
    merge!(sc, sc1...)
This updates `sc` by merging SortedDicts or SortedMultiDicts `sc1`,
etc. into `sc`. These must all must have the same key-value types.
In the case of keys duplicated among the arguments, the rightmost
argument that owns the key gets its value stored for SortedDict. In
the case of SortedMultiDict all the key-value pairs are stored, and
for overlapping keys the ordering is left-to-right. This function is
not available for SortedSet, but the `union!` function (see below)
provides equivalent functionality. Time: O(*cN* log *N*), where *N*
is the total size of all the arguments.
"""
function merge!(m::SortedDict{K,D,Ord},
                others::AbstractDict{K,D}...) where {K,D,Ord <: Ordering}
    for o in others
        mergetwo!(m,o)
    end
end
"""
    merge(sc1, sc2...)
This returns a SortedDict or SortedMultiDict that results from
merging SortedDicts or SortedMultiDicts `sc1`, `sc2`, etc., which
all must have the same key-value-ordering types. In the case of keys
duplicated among the arguments, the rightmost argument that owns the
key gets its value stored for SortedDict. In the case of
SortedMultiDict all the key-value pairs are stored, and for keys
shared between `sc1` and `sc2` the ordering is left-to-right. This
function is not available for SortedSet, but the `union` function
(see below) provides equivalent functionality. Time: O(*cN* log
*N*), where *N* is the total size of all the arguments.
"""
function merge(m::SortedDict{K,D,Ord},
               others::AbstractDict{K,D}...) where {K,D,Ord <: Ordering}
    result = packcopy(m)
    merge!(result, others...)
    result
end
"""
    similar(sc)
Returns a new SortedDict, SortedMultiDict, or SortedSet of the same
type and with the same ordering as `sc` but with no entries (i.e.,
empty). Time: O(1)
"""
similar(m::SortedDict{K,D,Ord}) where {K,D,Ord<:Ordering} =
    SortedDict{K,D,Ord}(orderobject(m))
isordered(::Type{T}) where {T<:SortedDict} = true
mutable struct SortedMultiDict{K, D, Ord <: Ordering}
    bt::BalancedTree23{K,D,Ord}
    ## Base constructors
    """
        SortedMultiDict{K,V,Ord}(o)
    Construct an empty sorted multidict in which type parameters are
    explicitly listed; ordering object is explicitly specified. (See
    below for discussion of ordering.) An empty SortedMultiDict may also
    be constructed via `SortedMultiDict(K[], V[], o)` where the `o`
    argument is optional.
    """
    SortedMultiDict{K,D,Ord}(o::Ord) where {K,D,Ord} = new{K,D,Ord}(BalancedTree23{K,D,Ord}(o))
    function SortedMultiDict{K,D,Ord}(o::Ord, kv) where {K,D,Ord}
        smd = new{K,D,Ord}(BalancedTree23{K,D,Ord}(o))
        if eltype(kv) <: Pair
            # It's (possibly?) more efficient to access the first and second
            # elements of Pairs directly, rather than destructure
            for p in kv
                insert!(smd, p.first, p.second)
            end
        else
            for (k,v) in kv
                insert!(smd, k, v)
            end
        end
        return smd
    end
end
"""
    SortedMultiDict()
Construct an empty `SortedMultiDict` with key type `Any` and value type
`Any`. Ordering defaults to `Forward` ordering.
**Note that a key type of `Any` or any other abstract type will lead
to slow performance.**
"""
SortedMultiDict() = SortedMultiDict{Any,Any,ForwardOrdering}(Forward)
"""
    SortedMultiDict(o)
Construct an empty `SortedMultiDict` with key type `Any` and value type
`Any`, ordered using `o`.
**Note that a key type of `Any` or any other abstract type will lead
to slow performance.**
"""
SortedMultiDict(o::O) where {O<:Ordering} = SortedMultiDict{Any,Any,O}(o)
"""
    SortedMultiDict(k1=>v1, k2=>v2, ...)
Arguments are key-value pairs for insertion into the multidict. The
keys must be of the same type as one another; the values must also
be of one type.
"""
SortedMultiDict(ps::Pair...) = SortedMultiDict(Forward, ps)
"""
    SortedMultiDict(o, k1=>v1, k2=>v2, ...)
The first argument `o` is an ordering object. The remaining
arguments are key-value pairs for insertion into the multidict. The
keys must be of the same type as one another; the values must also
be of one type.
"""
SortedMultiDict(o::Ordering, ps::Pair...) = SortedMultiDict(o, ps)
SortedMultiDict{K,D}(ps::Pair...) where {K,D} = SortedMultiDict{K,D,ForwardOrdering}(Forward, ps)
SortedMultiDict{K,D}(o::Ord, ps::Pair...) where {K,D,Ord<:Ordering} = SortedMultiDict{K,D,Ord}(o, ps)
SortedMultiDict(o::Ord, d::AbstractDict{K,D}) where {K,D,Ord<:Ordering} = SortedMultiDict{K,D,Ord}(o, d)
"""
    SortedMultiDict{K,D}(iter)
Takes an arbitrary iterable object of key=>value pairs with
key type `K` and value type `D`. The default Forward ordering is used.
"""
SortedMultiDict{K,D}(kv) where {K,D} = SortedMultiDict{K,D}(Forward, kv)
"""
    SortedMultiDict{K,D}(o, iter)
Takes an arbitrary iterable object of key=>value pairs with
key type `K` and value type `D`. The ordering object `o` is explicitly given.
"""
function SortedMultiDict{K,D}(o::Ord, kv) where {K,D,Ord<:Ordering}
    try
        SortedMultiDict{K,D,Ord}(o, kv)
    catch e
        if not_iterator_of_pairs(kv)
            throw(ArgumentError("SortedMultiDict(kv): kv needs to be an iterator of tuples or pairs"))
        else
            rethrow(e)
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
            throw(ArgumentError("SortedMultiDict(kv): kv needs to be an iterator of tuples or pairs"))
        else
            rethrow(e)
        end
    end
end
_sorted_multidict_with_eltype(o::Ord, ps, ::Type{Pair{K,D}}) where {K,D,Ord} = SortedMultiDict{  K,  D,Ord}(o, ps)
_sorted_multidict_with_eltype(o::Ord, kv, ::Type{Tuple{K,D}}) where {K,D,Ord} = SortedMultiDict{  K,  D,Ord}(o, kv)
_sorted_multidict_with_eltype(o::Ord, ps, ::Type{Pair{K}}  ) where {K,  Ord} = SortedMultiDict{  K,Any,Ord}(o, ps)
_sorted_multidict_with_eltype(o::Ord, kv, ::Type            ) where {    Ord} = SortedMultiDict{Any,Any,Ord}(o, kv)
const SMDSemiToken = IntSemiToken
const SMDToken = Tuple{SortedMultiDict, IntSemiToken}
"""
    insert!(sc, k)
Argument `sc` is a SortedDict or SortedMultiDict, `k` is a key and
`v` is the corresponding value. This inserts the `(k,v)` pair into
the container. If the key is already present in a SortedDict, this
overwrites the old value. In the case of SortedMultiDict, no
overwriting takes place (since SortedMultiDict allows the same key
to associate with multiple values). In the case of SortedDict, the
return value is a pair whose first entry is boolean and indicates
whether the insertion was new (i.e., the key was not previously
present) and the second entry is the semitoken of the new entry. In
the case of SortedMultiDict, a semitoken is returned (but no
boolean). Time: O(*c* log *n*)
"""
@inline function insert!(m::SortedMultiDict{K,D,Ord}, k_, d_) where {K, D, Ord <: Ordering}
    b, i = insert!(m.bt, convert(K,k_), convert(D,d_), true)
    IntSemiToken(i)
end
"""
    push!(sc, k=>v)
Argument `sc` is a SortedDict or SortedMultiDict and `k=>v` is a
key-value pair. This inserts the key-value pair into the container.
If the key is already present, this overwrites the old value. The
return value is `sc`. Time: O(*c* log *n*)
"""
@inline function push!(m::SortedMultiDict{K,D}, pr::Pair) where {K,D}
    insert!(m.bt, convert(K,pr[1]), convert(D,pr[2]), true)
    m
end
"""
    first(sc)
Argument `sc` is a SortedDict, SortedMultiDict or SortedSet. This
function returns the first item (a `k=>v` pair for SortedDict and
SortedMultiDict or a key for SortedSet) according to the sorted
order in the container. Thus, `first(sc)` is equivalent to
`deref((sc,startof(sc)))`. It is an error to call this function on
an empty container. Time: O(log *n*)
"""
@inline function first(m::SortedMultiDict)
    i = beginloc(m.bt)
    i == 2 && throw(BoundsError())
    return Pair(m.bt.data[i].k, m.bt.data[i].d)
end
"""
    last(sc)
Argument `sc` is a SortedDict, SortedMultiDict or SortedSet. This
function returns the last item (a `k=>v` pair for SortedDict and
SortedMultiDict or a key for SortedSet) according to the sorted
order in the container. Thus, `last(sc)` is equivalent to
`deref((sc,endof(sc)))`. It is an error to call this function on an
empty container. Time: O(log *n*)
"""
@inline function last(m::SortedMultiDict)
    i = endloc(m.bt)
    i == 1 && throw(BoundsError())
    return Pair(m.bt.data[i].k, m.bt.data[i].d)
end
function searchequalrange(m::SortedMultiDict, k_)
    k = convert(keytype(m),k_)
    i1 = findkeyless(m.bt, k)
    i2, exactfound = findkey(m.bt, k)
    if exactfound
        i1a = nextloc0(m.bt, i1)
        return IntSemiToken(i1a), IntSemiToken(i2)
    else
        return IntSemiToken(2), IntSemiToken(1)
    end
end
function in_(k_, d_, m::SortedMultiDict)
    k = convert(keytype(m), k_)
    d = convert(valtype(m), d_)
    i1 = findkeyless(m.bt, k)
    i2,exactfound = findkey(m.bt,k)
    !exactfound && return false
    ord = m.bt.ord
    while true
        i1 = nextloc0(m.bt, i1)
        @assert(eq(ord, m.bt.data[i1].k, k))
        m.bt.data[i1].d == d && return true
        i1 == i2 && return false
    end
end
"""
    eltype(sc)
Returns the (key,value) type (a 2-entry pair, i.e., `Pair{K,V}`) for
SortedDict and SortedMultiDict. Returns the key type for SortedSet.
This function may also be applied to the type itself. Time: O(1)
"""
@inline eltype(m::SortedMultiDict{K,D,Ord}) where {K,D,Ord <: Ordering} =  Pair{K,D}
@inline eltype(::Type{SortedMultiDict{K,D,Ord}}) where {K,D,Ord <: Ordering} =  Pair{K,D}
"""
    in(p, sc)
Returns true if `p` is in `sc`. In the case that `sc` is a
SortedDict or SortedMultiDict, `p` is a key=>value pair. In the
case that `sc` is a SortedSet, `p` should be a key. Time: O(*c* log
*n* + *d*) for SortedDict and SortedSet, where *d* stands for the
time to compare two values. In the case of SortedMultiDict, the time
is O(*c* log *n* + *dl*), and *l* stands for the number of entries
that have the key of the given pair. (So therefore this call is
inefficient if the same key addresses a large number of values, and
an alternative should be considered.)
"""
@inline in(pr::Pair, m::SortedMultiDict) =
    in_(pr[1], pr[2], m)
@inline in(::Tuple{Any,Any}, ::SortedMultiDict) =
    throw(ArgumentError("'(k,v) in sortedmultidict' not supported in Julia 0.4 or 0.5.  See documentation"))
"""
    keytype(sc)
Returns the key type for SortedDict, SortedMultiDict and SortedSet.
This function may also be applied to the type itself. Time: O(1)
"""
@inline keytype(m::SortedMultiDict{K,D,Ord}) where {K,D,Ord <: Ordering} = K
@inline keytype(::Type{SortedMultiDict{K,D,Ord}}) where {K,D,Ord <: Ordering} = K
"""
    valtype(sc)
Returns the value type for SortedDict and SortedMultiDict. This
function may also be applied to the type itself. Time: O(1)
"""
@inline valtype(m::SortedMultiDict{K,D,Ord}) where {K,D,Ord <: Ordering} = D
@inline valtype(::Type{SortedMultiDict{K,D,Ord}}) where {K,D,Ord <: Ordering} = D
"""
    ordtype(sc)
Returns the order type for SortedDict, SortedMultiDict and
SortedSet. This function may also be applied to the type itself.
Time: O(1)
"""
@inline ordtype(m::SortedMultiDict{K,D,Ord}) where {K,D,Ord <: Ordering} = Ord
@inline ordtype(::Type{SortedMultiDict{K,D,Ord}}) where {K,D,Ord <: Ordering} = Ord
"""
    orderobject(sc)
Returns the order object used to construct the container. Time: O(1)
"""
@inline orderobject(m::SortedMultiDict) = m.bt.ord
"""
    haskey(sc,k)
Returns true if key `k` is present for SortedDict, SortedMultiDict
or SortedSet `sc`. For SortedSet, `haskey(sc,k)` is a synonym for
`in(k,sc)`. For SortedDict and SortedMultiDict, `haskey(sc,k)` is
equivalent to `in(k,keys(sc))`. Time: O(*c* log *n*)
"""
@inline function haskey(m::SortedMultiDict, k_)
    i, exactfound = findkey(m.bt,convert(keytype(m),k_))
    exactfound
end
"""
    isequal(sc1,sc2)
Checks if two containers are equal in the sense that they contain
the same items; the keys are compared using the `eq` method, while
the values are compared with the `isequal` function. In the case of
SortedMultiDict, equality requires that the values associated with a
particular key have same order (that is, the same insertion order).
Note that `isequal` in this sense does not imply any correspondence
between semitokens for items in `sc1` with those for `sc2`. If the
equality-testing method associated with the keys and values implies
hash-equivalence in the case of SortedDict, then `isequal` of the
entire containers implies hash-equivalence of the containers. Time:
O(*cn* + *n* log *n*)
"""
function isequal(m1::SortedMultiDict, m2::SortedMultiDict)
    ord = orderobject(m1)
    if !isequal(ord, orderobject(m2)) || !isequal(eltype(m1), eltype(m2))
        throw(ArgumentError("Cannot use isequal for two SortedMultiDicts unless their element types and ordering objects are equal"))
    end
    p1 = startof(m1)
    p2 = startof(m2)
    while true
        if p1 == pastendsemitoken(m1)
            return p2 == pastendsemitoken(m2)
        end
        if p2 == pastendsemitoken(m2)
            return false
        end
        k1,d1 = deref((m1,p1))
        k2,d2 = deref((m2,p2))
        if !eq(ord,k1,k2) || !isequal(d1,d2)
            return false
        end
        p1 = advance((m1,p1))
        p2 = advance((m2,p2))
    end
end
const SDorAbstractDict = Union{AbstractDict,SortedMultiDict}
function mergetwo!(m::SortedMultiDict{K,D,Ord},
                   m2::SDorAbstractDict) where {K,D,Ord <: Ordering}
    for (k,v) in m2
        insert!(m.bt, convert(K,k), convert(D,v), true)
    end
end
"""
    packcopy(sc)
This returns a copy of `sc` in which the data is packed. When
deletions take place, the previously allocated memory is not
returned. This function can be used to reclaim memory after many
deletions. Time: O(*cn* log *n*)
"""
function packcopy(m::SortedMultiDict{K,D,Ord}) where {K,D,Ord <: Ordering}
    w = SortedMultiDict{K,D}(orderobject(m))
    mergetwo!(w,m)
    w
end
"""
    packdeepcopy(sc)
This returns a packed copy of `sc` in which the keys and values are
deep-copied. This function can be used to reclaim memory after many
deletions. Time: O(*cn* log *n*)
"""
function packdeepcopy(m::SortedMultiDict{K,D,Ord}) where {K,D,Ord <: Ordering}
    w = SortedMultiDict{K,D}(orderobject(m))
    for (k,v) in m
        insert!(w.bt, deepcopy(k), deepcopy(v), true)
    end
    w
end
"""
    merge!(sc, sc1...)
This updates `sc` by merging SortedDicts or SortedMultiDicts `sc1`,
etc. into `sc`. These must all must have the same key-value types.
In the case of keys duplicated among the arguments, the rightmost
argument that owns the key gets its value stored for SortedDict. In
the case of SortedMultiDict all the key-value pairs are stored, and
for overlapping keys the ordering is left-to-right. This function is
not available for SortedSet, but the `union!` function (see below)
provides equivalent functionality. Time: O(*cN* log *N*), where *N*
is the total size of all the arguments.
"""
function merge!(m::SortedMultiDict{K,D,Ord},
                others::SDorAbstractDict...) where {K,D,Ord <: Ordering}
    for o in others
        mergetwo!(m,o)
    end
end
"""
    merge(sc1, sc2...)
This returns a SortedDict or SortedMultiDict that results from
merging SortedDicts or SortedMultiDicts `sc1`, `sc2`, etc., which
all must have the same key-value-ordering types. In the case of keys
duplicated among the arguments, the rightmost argument that owns the
key gets its value stored for SortedDict. In the case of
SortedMultiDict all the key-value pairs are stored, and for keys
shared between `sc1` and `sc2` the ordering is left-to-right. This
function is not available for SortedSet, but the `union` function
(see below) provides equivalent functionality. Time: O(*cN* log
*N*), where *N* is the total size of all the arguments.
"""
function merge(m::SortedMultiDict{K,D,Ord},
               others::SDorAbstractDict...) where {K,D,Ord <: Ordering}
    result = packcopy(m)
    merge!(result, others...)
    result
end
function Base.show(io::IO, m::SortedMultiDict{K,D,Ord}) where {K,D,Ord <: Ordering}
    print(io, "SortedMultiDict(")
    print(io, orderobject(m), ",")
    l = length(m)
    for (count,(k,v)) in enumerate(m)
        print(io, k, " => ", v)
        if count < l
            print(io, ", ")
        end
    end
    print(io, ")")
end
"""
    similar(sc)
Returns a new SortedDict, SortedMultiDict, or SortedSet of the same
type and with the same ordering as `sc` but with no entries (i.e.,
empty). Time: O(1)
"""
similar(m::SortedMultiDict{K,D,Ord}) where {K,D,Ord<:Ordering} =
   SortedMultiDict{K,D}(orderobject(m))
isordered(::Type{T}) where {T<:SortedMultiDict} = true
"""
    SortedSet(iter, o=Forward)
and
    `SortedSet{K}(iter, o=Forward)`
and
    `SortedSet(o, iter)`
and
    `SortedSet{K}(o, iter)`
Construct a SortedSet using keys given by iterable `iter` (e.g., an
array) and ordering object `o`. The ordering object defaults to
`Forward` if not specified.
"""
mutable struct SortedSet{K, Ord <: Ordering}
    bt::BalancedTree23{K,Nothing,Ord}
    function SortedSet{K,Ord}(o::Ord=Forward, iter=[]) where {K,Ord<:Ordering}
        sorted_set = new{K,Ord}(BalancedTree23{K,Nothing,Ord}(o))
        for item in iter
            push!(sorted_set, item)
        end
        sorted_set
    end
end
"""
    SortedSet()
Construct a `SortedSet{Any}` with `Forward` ordering.
**Note that a key type of `Any` or any other abstract type will lead
to slow performance.**
"""
SortedSet() = SortedSet{Any,ForwardOrdering}(Forward)
"""
    SortedSet(o)
Construct a `SortedSet{Any}` with `o` ordering.
**Note that a key type of `Any` or any other abstract type will lead
to slow performance.**
"""
SortedSet(o::O) where {O<:Ordering} = SortedSet{Any,O}(o)
SortedSet(o1::Ordering,o2::Ordering) =
    throw(ArgumentError("SortedSet with two parameters must be called with an Ordering and an interable"))
SortedSet(o::Ordering, iter) = sortedset_with_eltype(o, iter, eltype(iter))
SortedSet(iter, o::Ordering=Forward) = sortedset_with_eltype(o, iter, eltype(iter))
"""
    SortedSet{K}()
Construct a `SortedSet` of keys of type `K` with `Forward` ordering.
"""
SortedSet{K}() where {K} = SortedSet{K,ForwardOrdering}(Forward)
"""
    SortedSet{K}(o)
Construct a `SortedSet` of keys of type `K` with ordering given according 
`o` parameter.
"""
SortedSet{K}(o::O) where {K,O<:Ordering} = SortedSet{K,O}(o)
SortedSet{K}(o1::Ordering,o2::Ordering) where {K} =
    throw(ArgumentError("SortedSet with two parameters must be called with an Ordering and an interable"))
SortedSet{K}(o::Ordering, iter) where {K} = sortedset_with_eltype(o, iter, K)
SortedSet{K}(iter, o::Ordering=Forward) where {K} = sortedset_with_eltype(o, iter, K)
sortedset_with_eltype(o::Ord, iter, ::Type{K}) where {K,Ord} = SortedSet{K,Ord}(o, iter)
const SetSemiToken = IntSemiToken
@inline function find(m::SortedSet, k_)
    ll, exactfound = findkey(m.bt, convert(keytype(m),k_))
    IntSemiToken(exactfound ? ll : 2)
end
"""
    insert!(sc, k)
Argument `sc` is a SortedSet and `k` is a key. This inserts the key
into the container. If the key is already present, this overwrites
the old value. (This is not necessarily a no-op; see below for
remarks about the customizing the sort order.) The return value is a
pair whose first entry is boolean and indicates whether the
insertion was new (i.e., the key was not previously present) and the
second entry is the semitoken of the new entry. Time: O(*c* log *n*)
"""
@inline function insert!(m::SortedSet, k_)
    b, i = insert!(m.bt, convert(keytype(m),k_), nothing, false)
    b, IntSemiToken(i)
end
"""
    push!(sc, k)
Argument `sc` is a SortedSet and `k` is a key. This inserts the key
into the container. If the key is already present, this overwrites
the old value. (This is not necessarily a no-op; see below for
remarks about the customizing the sort order.) The return value is
`sc`. Time: O(*c* log *n*)
"""
@inline function push!(m::SortedSet, k_)
    b, i = insert!(m.bt, convert(keytype(m),k_), nothing, false)
    m
end
"""
    first(sc)
Argument `sc` is a SortedDict, SortedMultiDict or SortedSet. This
function returns the first item (a `k=>v` pair for SortedDict and
SortedMultiDict or a key for SortedSet) according to the sorted
order in the container. Thus, `first(sc)` is equivalent to
`deref((sc,startof(sc)))`. It is an error to call this function on
an empty container. Time: O(log *n*)
"""
@inline function first(m::SortedSet)
    i = beginloc(m.bt)
    i == 2 && throw(BoundsError())
    return m.bt.data[i].k
end
"""
    last(sc)
Argument `sc` is a SortedDict, SortedMultiDict or SortedSet. This
function returns the last item (a `k=>v` pair for SortedDict and
SortedMultiDict or a key for SortedSet) according to the sorted
order in the container. Thus, `last(sc)` is equivalent to
`deref((sc,endof(sc)))`. It is an error to call this function on an
empty container. Time: O(log *n*)
"""
@inline function last(m::SortedSet)
    i = endloc(m.bt)
    i == 1 && throw(BoundsError())
    return m.bt.data[i].k
end
@inline function in(k_, m::SortedSet)
    i, exactfound = findkey(m.bt, convert(keytype(m),k_))
    return exactfound
end
"""
    eltype(sc)
Returns the key type for SortedSet.
This function may also be applied to the type itself. Time: O(1)
"""
@inline eltype(m::SortedSet{K,Ord}) where {K,Ord <: Ordering} = K
@inline eltype(::Type{SortedSet{K,Ord}}) where {K,Ord <: Ordering} = K
"""
    keytype(sc)
Returns the key type for SortedDict, SortedMultiDict and SortedSet.
This function may also be applied to the type itself. Time: O(1)
"""
@inline keytype(m::SortedSet{K,Ord}) where {K,Ord <: Ordering} = K
@inline keytype(::Type{SortedSet{K,Ord}}) where {K,Ord <: Ordering} = K
"""
    ordtype(sc)
Returns the order type for SortedDict, SortedMultiDict and
SortedSet. This function may also be applied to the type itself.
Time: O(1)
"""
@inline ordtype(m::SortedSet{K,Ord}) where {K,Ord <: Ordering} = Ord
@inline ordtype(::Type{SortedSet{K,Ord}}) where {K,Ord <: Ordering} = Ord
"""
    orderobject(sc)
Returns the order object used to construct the container. Time: O(1)
"""
@inline orderobject(m::SortedSet) = m.bt.ord
"""
    haskey(sc,k)
Returns true if key `k` is present for SortedDict, SortedMultiDict
or SortedSet `sc`. For SortedSet, `haskey(sc,k)` is a synonym for
`in(k,sc)`. For SortedDict and SortedMultiDict, `haskey(sc,k)` is
equivalent to `in(k,keys(sc))`. Time: O(*c* log *n*)
"""
haskey(m::SortedSet, k_) = in(k_, m)
"""
    delete!(sc, k)
Argument `sc` is a SortedDict or SortedSet and `k` is a key. This
operation deletes the item whose key is `k`. It is a `KeyError` if
`k` is not a key of an item in the container. After this operation
is complete, any token addressing the deleted item is invalid.
Returns `sc`. Time: O(*c* log *n*)
"""
@inline function delete!(m::SortedSet, k_)
    i, exactfound = findkey(m.bt,convert(keytype(m),k_))
    !exactfound && throw(KeyError(k_))
    delete!(m.bt, i)
    m
end
"""
    pop!(sc, k)
Deletes the item with key `k` in SortedDict or SortedSet `sc` and
returns the value that was associated with `k` in the case of
SortedDict or `k` itself in the case of SortedSet. A `KeyError`
results if `k` is not in `sc`. Time: O(*c* log *n*)
"""
@inline function pop!(m::SortedSet, k_)
    k = convert(keytype(m),k_)
    i, exactfound = findkey(m.bt, k)
    !exactfound && throw(KeyError(k_))
    d = m.bt.data[i].d
    delete!(m.bt, i)
    k
end
"""
    pop!(ss)
Deletes the item with first key in SortedSet `ss` and returns the
key. A `BoundsError` results if `ss` is empty. Time: O(*c* log *n*)
"""
@inline function pop!(m::SortedSet)
    i = beginloc(m.bt)
    i == 2 && throw(BoundsError())
    k = m.bt.data[i].k
    delete!(m.bt, i)
    k
end
"""
    isequal(sc1,sc2)
Checks if two containers are equal in the sense that they contain
the same items; the keys are compared using the `eq` method, while
the values are compared with the `isequal` function. In the case of
SortedMultiDict, equality requires that the values associated with a
particular key have same order (that is, the same insertion order).
Note that `isequal` in this sense does not imply any correspondence
between semitokens for items in `sc1` with those for `sc2`. If the
equality-testing method associated with the keys and values implies
hash-equivalence in the case of SortedDict, then `isequal` of the
entire containers implies hash-equivalence of the containers. Time:
O(*cn* + *n* log *n*)
"""
function isequal(m1::SortedSet, m2::SortedSet)
    ord = orderobject(m1)
    if !isequal(ord, orderobject(m2)) || !isequal(eltype(m1), eltype(m2))
        throw(ArgumentError("Cannot use isequal for two SortedSets unless their element types and ordering objects are equal"))
    end
    p1 = startof(m1)
    p2 = startof(m2)
    while true
        if p1 == pastendsemitoken(m1)
            return p2 == pastendsemitoken(m2)
        end
        if p2 == pastendsemitoken(m2)
            return false
        end
        k1 = deref((m1,p1))
        k2 = deref((m2,p2))
        if !eq(ord,k1,k2)
            return false
        end
        p1 = advance((m1,p1))
        p2 = advance((m2,p2))
    end
end
"""
    union!(ss, iterable)
This function inserts each item from the second argument (which must
iterable) into the SortedSet `ss`. The items must be convertible to
the key-type of `ss`. Time: O(*ci* log *n*) where *i* is the number
of items in the iterable argument.
"""
function union!(m1::SortedSet{K,Ord}, iterable_item) where {K, Ord <: Ordering}
    for k in iterable_item
        push!(m1,convert(K,k))
    end
    m1
end
"""
    union(ss, iterable...)
This function creates a new SortedSet (the return argument) and
inserts each item from `ss` and each item from each iterable
argument into the returned SortedSet. Time: O(*cn* log *n*) where
*n* is the total number of items in all the arguments.
"""
function union(m1::SortedSet, others...)
    mr = packcopy(m1)
    for m2 in others
        union!(mr, m2)
    end
    mr
end
function intersect2(m1::SortedSet{K, Ord}, m2::SortedSet{K, Ord}) where {K, Ord <: Ordering}
    ord = orderobject(m1)
    mi = SortedSet(K[], ord)
    p1 = startof(m1)
    p2 = startof(m2)
    while true
        if p1 == pastendsemitoken(m1) || p2 == pastendsemitoken(m2)
            return mi
        end
        k1 = deref((m1,p1))
        k2 = deref((m2,p2))
        if lt(ord,k1,k2)
            p1 = advance((m1,p1))
        elseif lt(ord,k2,k1)
            p2 = advance((m2,p2))
        else
            push!(mi,k1)
            p1 = advance((m1,p1))
            p2 = advance((m2,p2))
        end
    end
end
"""
    intersect(ss, others...)
Each argument is a SortedSet with the same key and order type. The
return variable is a new SortedSet that is the intersection of all
the sets that are input. Time: O(*cn* log *n*), where *n* is the
total number of items in all the arguments.
"""
function intersect(m1::SortedSet{K,Ord}, others::SortedSet{K,Ord}...) where {K, Ord <: Ordering}
    ord = orderobject(m1)
    for s2 in others
        if !isequal(ord, orderobject(s2))
            throw(ArgumentError("Cannot intersect two SortedSets unless their ordering objects are equal"))
        end
    end
    if length(others) == 0
        return m1
    else
        mi = intersect2(m1, others[1])
        for s2 = others[2:end]
            mi = intersect2(mi, s2)
        end
        return mi
    end
end
"""
    symdiff(ss1, ss2)
The two argument are sorted sets with the same key and order type.
This operation computes the symmetric difference, i.e., a sorted set
containing entries that are in one of `ss1`, `ss2` but not both.
Time: O(*cn* log *n*), where *n* is the total size of the two
containers.
"""
function symdiff(m1::SortedSet{K,Ord}, m2::SortedSet{K,Ord}) where {K, Ord <: Ordering}
    ord = orderobject(m1)
    if !isequal(ord, orderobject(m2))
        throw(ArgumentError("Cannot apply symdiff to two SortedSets unless their ordering objects are equal"))
    end
    mi = SortedSet(K[], ord)
    p1 = startof(m1)
    p2 = startof(m2)
    while true
        m1end = p1 == pastendsemitoken(m1)
        m2end = p2 == pastendsemitoken(m2)
        if m1end && m2end
            return mi
        elseif m1end
            push!(mi, deref((m2,p2)))
            p2 = advance((m2,p2))
        elseif m2end
            push!(mi, deref((m1,p1)))
            p1 = advance((m1,p1))
        else
            k1 = deref((m1,p1))
            k2 = deref((m2,p2))
            if lt(ord,k1,k2)
                push!(mi, k1)
                p1 = advance((m1,p1))
            elseif lt(ord,k2,k1)
                push!(mi, k2)
                p2 = advance((m2,p2))
            else
                p1 = advance((m1,p1))
                p2 = advance((m2,p2))
            end
        end
    end
end
"""
    setdiff(ss1, ss2)
The two arguments are sorted sets with the same key and order type.
This operation computes the difference, i.e., a sorted set
containing entries that in are in `ss1` but not `ss2`. Time: O(*cn*
log *n*), where *n* is the total size of the two containers.
"""
function setdiff(m1::SortedSet{K,Ord}, m2::SortedSet{K,Ord}) where {K, Ord <: Ordering}
    ord = orderobject(m1)
    if !isequal(ord, orderobject(m2))
        throw(ArgumentError("Cannot apply setdiff to two SortedSets unless their ordering objects are equal"))
    end
    mi = SortedSet(K[], ord)
    p1 = startof(m1)
    p2 = startof(m2)
    while true
        if p1 == pastendsemitoken(m1)
            return mi
        elseif p2 == pastendsemitoken(m2)
            push!(mi, deref((m1,p1)))
            p1 = advance((m1,p1))
        else
            k1 = deref((m1,p1))
            k2 = deref((m2,p2))
            if lt(ord,k1,k2)
                push!(mi, deref((m1,p1)))
                p1 = advance((m1,p1))
            elseif lt(ord,k2,k1)
                p2 = advance((m2,p2))
            else
                p1 = advance((m1,p1))
                p2 = advance((m2,p2))
            end
        end
    end
end
"""
    setdiff!(ss, iterable)
This function deletes items in `ss` that appear in the second
argument. The second argument must be iterable and its entries must
be convertible to the key type of m1. Time: O(*cm* log *n*), where
*n* is the size of `ss` and *m* is the number of items in
`iterable`.
"""
function setdiff!(m1::SortedSet, iterable)
    for p in iterable
        i = find(m1, p)
        if i != pastendsemitoken(m1)
            delete!((m1,i))
        end
    end
end
"""
    issubset(iterable, ss)
This function checks whether each item of the first argument is an
element of the SortedSet `ss`. The entries must be convertible to
the key-type of `ss`. Time: O(*cm* log *n*), where *n* is the sizes
of `ss` and *m* is the number of items in `iterable`.
"""
function issubset(iterable, m2::SortedSet)
    for k in iterable
        if !in(k, m2)
            return false
        end
    end
    return true
end
"""
    packcopy(sc)
This returns a copy of `sc` in which the data is packed. When
deletions take place, the previously allocated memory is not
returned. This function can be used to reclaim memory after many
deletions. Time: O(*cn* log *n*)
"""
function packcopy(m::SortedSet{K,Ord}) where {K,Ord <: Ordering}
    w = SortedSet(K[], orderobject(m))
    for k in m
        push!(w, k)
    end
    w
end
"""
    packdeepcopy(sc)
This returns a packed copy of `sc` in which the keys and values are
deep-copied. This function can be used to reclaim memory after many
deletions. Time: O(*cn* log *n*)
"""
function packdeepcopy(m::SortedSet{K,Ord}) where {K, Ord <: Ordering}
    w = SortedSet(K[], orderobject(m))
    for k in m
        newk = deepcopy(k)
        push!(w, newk)
    end
    w
end
function Base.show(io::IO, m::SortedSet{K,Ord}) where {K,Ord <: Ordering}
    print(io, "SortedSet(")
    keys = K[]
    for k in m
        push!(keys, k)
    end
    print(io, keys)
    println(io, ",")
    print(io, orderobject(m))
    print(io, ")")
end
"""
    similar(sc)
Returns a new SortedDict, SortedMultiDict, or SortedSet of the same
type and with the same ordering as `sc` but with no entries (i.e.,
empty). Time: O(1)
"""
similar(m::SortedSet{K,Ord}) where {K,Ord<:Ordering} =
SortedSet{K,Ord}(orderobject(m))
import Base.isless
import Base.isequal
import Base.colon
import Base.endof
const SDMContainer = Union{SortedDict, SortedMultiDict}
const SAContainer = Union{SDMContainer, SortedSet}
const Token = Tuple{SAContainer, IntSemiToken}
const SDMToken = Tuple{SDMContainer, IntSemiToken}
const SetToken = Tuple{SortedSet, IntSemiToken}
@inline startof(m::SAContainer) = IntSemiToken(beginloc(m.bt))
@inline endof(m::SAContainer) = IntSemiToken(endloc(m.bt))
@inline pastendsemitoken(::SAContainer) = IntSemiToken(2)
@inline beforestartsemitoken(::SAContainer) = IntSemiToken(1)
@inline function delete!(ii::Token)
    has_data(ii)
    delete!(ii[1].bt, ii[2].address)
end
@inline function advance(ii::Token)
    not_pastend(ii)
    IntSemiToken(nextloc0(ii[1].bt, ii[2].address))
end
@inline function regress(ii::Token)
    not_beforestart(ii)
    IntSemiToken(prevloc0(ii[1].bt, ii[2].address))
end
@inline status(ii::Token) =
       !(ii[2].address in ii[1].bt.useddatacells) ? 0 :
         ii[2].address == 1 ?                       2 :
         ii[2].address == 2 ?                       3 : 1
"""
    compare(m::SAContainer, s::IntSemiToken, t::IntSemiToken)
Determines the  relative positions of the  data items indexed
by `(m,s)` and  `(m,t)` in the sorted order. The  return value is `-1`
if `(m,s)` precedes `(m,t)`, `0` if they are equal, and `1` if `(m,s)`
succeeds `(m,t)`. `s`  and `t`  are semitokens  for the  same container `m`.
"""
@inline compare(m::SAContainer, s::IntSemiToken, t::IntSemiToken) =
      compareInd(m.bt, s.address, t.address)
@inline function deref(ii::SDMToken)
    has_data(ii)
    return Pair(ii[1].bt.data[ii[2].address].k, ii[1].bt.data[ii[2].address].d)
end
@inline function deref(ii::SetToken)
    has_data(ii)
    return ii[1].bt.data[ii[2].address].k
end
@inline function deref_key(ii::SDMToken)
    has_data(ii)
    return ii[1].bt.data[ii[2].address].k
end
@inline function deref_value(ii::SDMToken)
    has_data(ii)
    return ii[1].bt.data[ii[2].address].d
end
@inline function getindex(m::SortedDict,
                          i::IntSemiToken)
    has_data((m,i))
    return m.bt.data[i.address].d
end
@inline function getindex(m::SortedMultiDict,
                          i::IntSemiToken)
    has_data((m,i))
    return m.bt.data[i.address].d
end
@inline function setindex!(m::SortedDict,
                           d_,
                           i::IntSemiToken)
    has_data((m,i))
    m.bt.data[i.address] = KDRec{keytype(m),valtype(m)}(m.bt.data[i.address].parent,
                                                         m.bt.data[i.address].k,
                                                         convert(valtype(m),d_))
    m
end
@inline function setindex!(m::SortedMultiDict,
                           d_,
                           i::IntSemiToken)
    has_data((m,i))
    m.bt.data[i.address] = KDRec{keytype(m),valtype(m)}(m.bt.data[i.address].parent,
                                                         m.bt.data[i.address].k,
                                                         convert(valtype(m),d_))
    m
end
@inline function searchsortedfirst(m::SAContainer, k_)
    i = findkeyless(m.bt, convert(keytype(m), k_))
    IntSemiToken(nextloc0(m.bt, i))
end
@inline function searchsortedafter(m::SAContainer, k_)
    i, exactfound = findkey(m.bt, convert(keytype(m), k_))
    IntSemiToken(nextloc0(m.bt, i))
end
@inline function searchsortedlast(m::SAContainer, k_)
    i, exactfound = findkey(m.bt, convert(keytype(m),k_))
    IntSemiToken(i)
end
@inline not_beforestart(i::Token) =
    (!(i[2].address in i[1].bt.useddatacells) ||
     i[2].address == 1) && throw(BoundsError())
@inline not_pastend(i::Token) =
    (!(i[2].address in i[1].bt.useddatacells) ||
     i[2].address == 2) && throw(BoundsError())
@inline has_data(i::Token) =
    (!(i[2].address in i[1].bt.useddatacells) ||
     i[2].address < 3) && throw(BoundsError())
import Base.keys
import Base.values
@inline extractcontainer(s::SAContainer) = s
abstract type AbstractExcludeLast{ContainerType <: SAContainer} end
struct SDMExcludeLast{ContainerType <: SDMContainer} <:
                              AbstractExcludeLast{ContainerType}
    m::ContainerType
    first::Int
    pastlast::Int
end
struct SSExcludeLast{ContainerType <: SortedSet} <:
                              AbstractExcludeLast{ContainerType}
    m::ContainerType
    first::Int
    pastlast::Int
end
@inline extractcontainer(s::AbstractExcludeLast) = s.m
eltype(s::AbstractExcludeLast) = eltype(s.m)
abstract type AbstractIncludeLast{ContainerType <: SAContainer} end
struct SDMIncludeLast{ContainerType <: SDMContainer} <:
                               AbstractIncludeLast{ContainerType}
    m::ContainerType
    first::Int
    last::Int
end
struct SSIncludeLast{ContainerType <: SortedSet} <:
                               AbstractIncludeLast{ContainerType}
    m::ContainerType
    first::Int
    last::Int
end
@inline extractcontainer(s::AbstractIncludeLast) = s.m
eltype(s::AbstractIncludeLast) = eltype(s.m)
const SDMIterableTypesBase = Union{SDMContainer,
                                   SDMExcludeLast,
                                   SDMIncludeLast}
const SSIterableTypesBase = Union{SortedSet,
                                  SSExcludeLast,
                                  SSIncludeLast}
const SAIterableTypesBase = Union{SAContainer,
                                  AbstractExcludeLast,
                                  AbstractIncludeLast}
struct SDMKeyIteration{T <: SDMIterableTypesBase}
    base::T
end
eltype(s::SDMKeyIteration) = keytype(extractcontainer(s.base))
length(s::SDMKeyIteration) = length(extractcontainer(s.base))
struct SDMValIteration{T <: SDMIterableTypesBase}
    base::T
end
eltype(s::SDMValIteration) = valtype(extractcontainer(s.base))
length(s::SDMValIteration) = length(extractcontainer(s.base))
struct SDMSemiTokenIteration{T <: SDMIterableTypesBase}
    base::T
end
eltype(s::SDMSemiTokenIteration) = Tuple{IntSemiToken,
                                         keytype(extractcontainer(s.base)),
                                         valtype(extractcontainer(s.base))}
struct SSSemiTokenIteration{T <: SSIterableTypesBase}
    base::T
end
eltype(s::SSSemiTokenIteration) = Tuple{IntSemiToken,
                                        eltype(extractcontainer(s.base))}
struct SDMSemiTokenKeyIteration{T <: SDMIterableTypesBase}
    base::T
end
eltype(s::SDMSemiTokenKeyIteration) = Tuple{IntSemiToken,
                                            keytype(extractcontainer(s.base))}
struct SAOnlySemiTokensIteration{T <: SAIterableTypesBase}
    base::T
end
eltype(::SAOnlySemiTokensIteration) = IntSemiToken
struct SDMSemiTokenValIteration{T <: SDMIterableTypesBase}
    base::T
end
eltype(s::SDMSemiTokenValIteration) = Tuple{IntSemiToken,
                                            valtype(extractcontainer(s.base))}
const SACompoundIterable = Union{SDMKeyIteration,
                                 SDMValIteration,
                                 SDMSemiTokenIteration,
                                 SSSemiTokenIteration,
                                 SDMSemiTokenKeyIteration,
                                 SDMSemiTokenValIteration,
                                 SAOnlySemiTokensIteration}
@inline extractcontainer(s::SACompoundIterable) = extractcontainer(s.base)
const SAIterable = Union{SAIterableTypesBase, SACompoundIterable}
struct SAIterationState
    next::Int
    final::Int
end
@inline done(::SAIterable, state::SAIterationState) = state.next == state.final
@inline exclusive(m::T, ii::(Tuple{IntSemiToken,IntSemiToken})) where {T <: SDMContainer} =
    SDMExcludeLast(m, ii[1].address, ii[2].address)
@inline exclusive(m::T, ii::(Tuple{IntSemiToken,IntSemiToken})) where {T <: SortedSet} =
    SSExcludeLast(m, ii[1].address, ii[2].address)
@inline exclusive(m::T, i1::IntSemiToken, i2::IntSemiToken) where {T <: SAContainer} =
    exclusive(m, (i1,i2))
@inline inclusive(m::T, ii::(Tuple{IntSemiToken,IntSemiToken})) where {T <: SDMContainer} =
    SDMIncludeLast(m, ii[1].address, ii[2].address)
@inline inclusive(m::T, ii::(Tuple{IntSemiToken,IntSemiToken})) where {T <: SortedSet} =
    SSIncludeLast(m, ii[1].address, ii[2].address)
@inline inclusive(m::T, i1::IntSemiToken, i2::IntSemiToken) where {T <: SAContainer} =
    inclusive(m, (i1,i2))
@inline keys(ba::SortedDict{K,D,Ord}) where {K, D, Ord <: Ordering} = SDMKeyIteration(ba)
@inline keys(ba::T) where {T <: SDMIterableTypesBase} = SDMKeyIteration(ba)
in(k, keyit::SDMKeyIteration{SortedDict{K,D,Ord}}) where {K,D,Ord <: Ordering} =
    haskey(extractcontainer(keyit.base), k)
in(k, keyit::SDMKeyIteration{SortedMultiDict{K,D,Ord}}) where {K,D,Ord <: Ordering} =
    haskey(extractcontainer(keyit.base), k)
@inline values(ba::SortedDict{K,D,Ord}) where {K, D, Ord <: Ordering} = SDMValIteration(ba)
@inline values(ba::T) where {T <: SDMIterableTypesBase} = SDMValIteration(ba)
@inline semitokens(ba::T) where {T <: SDMIterableTypesBase} = SDMSemiTokenIteration(ba)
@inline semitokens(ba::T) where {T <: SSIterableTypesBase} = SSSemiTokenIteration(ba)
@inline semitokens(ki::SDMKeyIteration{T}) where {T <: SDMIterableTypesBase} =
                   SDMSemiTokenKeyIteration(ki.base)
@inline semitokens(vi::SDMValIteration{T}) where {T <: SDMIterableTypesBase} =
                   SDMSemiTokenValIteration(vi.base)
@inline onlysemitokens(ba::T) where {T <: SAIterableTypesBase} = SAOnlySemiTokensIteration(ba)
@inline start(m::SAContainer) = SAIterationState(nextloc0(m.bt,1), 2)
@inline start(e::SACompoundIterable) = start(e.base)
function start(e::AbstractExcludeLast)
    (!(e.first in e.m.bt.useddatacells) || e.first == 1 ||
        !(e.pastlast in e.m.bt.useddatacells)) &&
        throw(BoundsError())
    if compareInd(e.m.bt, e.first, e.pastlast) < 0
        return SAIterationState(e.first, e.pastlast)
    else
        return SAIterationState(2, 2)
    end
end
function start(e::AbstractIncludeLast)
    (!(e.first in e.m.bt.useddatacells) || e.first == 1 ||
        !(e.last in e.m.bt.useddatacells) || e.last == 2) &&
        throw(BoundsError())
    if compareInd(e.m.bt, e.first, e.last) <= 0
        return SAIterationState(e.first, nextloc0(e.m.bt, e.last))
    else
        return SAIterationState(2, 2)
    end
end
@inline function next(u::SAOnlySemiTokensIteration, state::SAIterationState)
    sn = state.next
    (sn < 3 || !(sn in extractcontainer(u).bt.useddatacells)) && throw(BoundsError())
    IntSemiToken(sn),
    SAIterationState(nextloc0(extractcontainer(u).bt, sn), state.final)
end
@inline function nexthelper(u, state::SAIterationState)
    sn = state.next
    (sn < 3 || !(sn in extractcontainer(u).bt.useddatacells)) && throw(BoundsError())
    extractcontainer(u).bt.data[sn], sn,
    SAIterationState(nextloc0(extractcontainer(u).bt, sn), state.final)
end
@inline function next(u::SDMIterableTypesBase, state::SAIterationState)
    dt, t, ni = nexthelper(u, state)
    (dt.k => dt.d), ni
end
@inline function next(u::SSIterableTypesBase, state::SAIterationState)
    dt, t, ni = nexthelper(u, state)
    dt.k, ni
end
@inline function next(u::SDMKeyIteration, state::SAIterationState)
    dt, t, ni = nexthelper(u, state)
    dt.k, ni
end
@inline function next(u::SDMValIteration, state::SAIterationState)
    dt, t, ni = nexthelper(u, state)
    dt.d, ni
end
@inline function next(u::SDMSemiTokenIteration, state::SAIterationState)
    dt, t, ni = nexthelper(u, state)
    (IntSemiToken(t), dt.k, dt.d), ni
end
@inline function next(u::SSSemiTokenIteration, state::SAIterationState)
    dt, t, ni = nexthelper(u, state)
    (IntSemiToken(t), dt.k), ni
end
@inline function next(u::SDMSemiTokenKeyIteration, state::SAIterationState)
    dt, t, ni = nexthelper(u, state)
    (IntSemiToken(t), dt.k), ni
end
@inline function next(u::SDMSemiTokenValIteration, state::SAIterationState)
    dt, t, ni = nexthelper(u, state)
    (IntSemiToken(t), dt.d), ni
end
eachindex(sd::SortedDict) = keys(sd)
eachindex(sdm::SortedMultiDict) = onlysemitokens(sdm)
eachindex(ss::SortedSet) = onlysemitokens(ss)
eachindex(sd::SDMExcludeLast{SortedDict{K,D,Ord}}) where {K,D,Ord <: Ordering} = keys(sd)
eachindex(smd::SDMExcludeLast{SortedMultiDict{K,D,Ord}}) where {K,D,Ord <: Ordering} =
     onlysemitokens(smd)
eachindex(ss::SSExcludeLast) = onlysemitokens(ss)
eachindex(sd::SDMIncludeLast{SortedDict{K,D,Ord}}) where {K,D,Ord <: Ordering} = keys(sd)
eachindex(smd::SDMIncludeLast{SortedMultiDict{K,D,Ord}}) where {K,D,Ord <: Ordering} =
     onlysemitokens(smd)
eachindex(ss::SSIncludeLast) = onlysemitokens(ss)
empty!(m::SAContainer) =  empty!(m.bt)
@inline length(m::SAContainer) = length(m.bt.data) - length(m.bt.freedatainds) - 2
@inline isempty(m::SAContainer) = length(m) == 0
import Base: sort, sort!
function sort!(d::OrderedDict; byvalue::Bool=false, args...)
    if d.ndel > 0
        rehash!(d)
    end
    if byvalue
        p = sortperm(d.vals; args...)
    else
        p = sortperm(d.keys; args...)
    end
    for (i,key) in enumerate(d.keys)
        idx = ht_keyindex(d, key, false)
        d.slots[idx] = p[i]
    end
    d.keys = d.keys[p]
    d.vals = d.vals[p]
    return d
end
sort(d::OrderedDict; args...) = sort!(copy(d); args...)
sort(d::Dict; args...) = sort!(OrderedDict(d); args...)
    export
        CircularBuffer,
        capacity,
        isfull
"""
New items are pushed to the back of the list, overwriting values in a circular fashion.
"""
mutable struct CircularBuffer{T} <: AbstractVector{T}
    capacity::Int
    first::Int
    buffer::Vector{T}
    CircularBuffer{T}(capacity::Int) where {T} = new{T}(capacity, 1, T[])
end
function Base.empty!(cb::CircularBuffer)
    cb.first = 1
    empty!(cb.buffer)
end
function _buffer_index(cb::CircularBuffer, i::Int)
    n = length(cb)
    if i < 1 || i > n
        throw(BoundsError("CircularBuffer out of range. cb=$cb i=$i"))
    end
    idx = cb.first + i - 1
    if idx > n
        idx - n
    else
        idx
    end
end
function Base.getindex(cb::CircularBuffer, i::Int)
    cb.buffer[_buffer_index(cb, i)]
end
function Base.setindex!(cb::CircularBuffer, data, i::Int)
    cb.buffer[_buffer_index(cb, i)] = data
    cb
end
function Base.push!(cb::CircularBuffer, data)
    # if full, increment and overwrite, otherwise push
    if length(cb) == cb.capacity
        cb.first = (cb.first == cb.capacity ? 1 : cb.first + 1)
        cb[length(cb)] = data
    else
        push!(cb.buffer, data)
    end
    cb
end
function Base.unshift!(cb::CircularBuffer, data)
    # if full, decrement and overwrite, otherwise unshift
    if length(cb) == cb.capacity
        cb.first = (cb.first == 1 ? cb.capacity : cb.first - 1)
        cb[1] = data
    else
        unshift!(cb.buffer, data)
    end
    cb
end
function Base.append!(cb::CircularBuffer, datavec::AbstractVector)
    # push at most last `capacity` items
    n = length(datavec)
    for i in max(1, n-capacity(cb)+1):n
        push!(cb, datavec[i])
    end
    cb
end
Base.length(cb::CircularBuffer) = length(cb.buffer)
Base.size(cb::CircularBuffer) = (length(cb),)
Base.convert(::Type{Array}, cb::CircularBuffer{T}) where {T} = T[x for x in cb]
Base.isempty(cb::CircularBuffer) = isempty(cb.buffer)
capacity(cb::CircularBuffer) = cb.capacity
isfull(cb::CircularBuffer) = length(cb) == cb.capacity
    export status
    export deref_key, deref_value, deref, advance, regress
    export PriorityQueue, peek
"""
    PriorityQueue(K, V, [ord])
Construct a new [`PriorityQueue`](@ref), with keys of type
`K` and values/priorites of type `V`.
If an order is not given, the priority queue is min-ordered using
the default comparison for `V`.
A `PriorityQueue` acts like a `Dict`, mapping values to their
priorities, with the addition of a `dequeue!` function to remove the
lowest priority element.
```jldoctest
julia> a = PriorityQueue(["a","b","c"],[2,3,1],Base.Order.Forward)
PriorityQueue{String,Int64,Base.Order.ForwardOrdering} with 3 entries:
  "c" => 1
  "b" => 3
  "a" => 2
```
"""
mutable struct PriorityQueue{K,V,O<:Ordering} <: AbstractDict{K,V}
    # Binary heap of (element, priority) pairs.
    xs::Array{Pair{K,V}, 1}
    o::O
    # Map elements to their index in xs
    index::Dict{K, Int}
    function PriorityQueue{K,V,O}(o::O) where {K,V,O<:Ordering}
        new{K,V,O}(Vector{Pair{K,V}}(), o, Dict{K, Int}())
    end
    function PriorityQueue{K,V,O}(o::O, itr) where {K,V,O<:Ordering}
        xs = Vector{Pair{K,V}}(uninitialized, length(itr))
        index = Dict{K, Int}()
        for (i, (k, v)) in enumerate(itr)
            xs[i] = Pair{K,V}(k, v)
            if haskey(index, k)
                throw(ArgumentError("PriorityQueue keys must be unique"))
            end
            index[k] = i
        end
        pq = new{K,V,O}(xs, o, index)
        # heapify
        for i in heapparent(length(pq.xs)):-1:1
            percolate_down!(pq, i)
        end
        pq
    end
end
PriorityQueue(o::Ordering=Forward) = PriorityQueue{Any,Any,typeof(o)}(o)
PriorityQueue(ps::Pair...) = PriorityQueue(Forward, ps)
PriorityQueue(o::Ordering, ps::Pair...) = PriorityQueue(o, ps)
PriorityQueue{K,V}(ps::Pair...) where {K,V} = PriorityQueue{K,V,ForwardOrdering}(Forward, ps)
PriorityQueue{K,V}(o::Ord, ps::Pair...) where {K,V,Ord<:Ordering} = PriorityQueue{K,V,Ord}(o, ps)
PriorityQueue{K,V}(kv) where {K,V} = PriorityQueue{K,V}(Forward, kv)
function PriorityQueue{K,V}(o::Ord, kv) where {K,V,Ord<:Ordering}
    try
        PriorityQueue{K,V,Ord}(o, kv)
    catch e
        if not_iterator_of_pairs(kv)
            throw(ArgumentError("PriorityQueue(kv): kv needs to be an iterator of tuples or pairs"))
        else
            rethrow(e)
        end
    end
end
PriorityQueue(o1::Ordering, o2::Ordering) = throw(ArgumentError("PriorityQueue with two parameters must be called with an Ordering and an interable of pairs"))
PriorityQueue(kv, o::Ordering=Forward) = PriorityQueue(o, kv)
function PriorityQueue(o::Ordering, kv)
    try
        _priority_queue_with_eltype(o, kv, eltype(kv))
    catch e
        if not_iterator_of_pairs(kv)
            throw(ArgumentError("PriorityQueue(kv): kv needs to be an iterator of tuples or pairs"))
        else
            rethrow(e)
        end
    end
end
_priority_queue_with_eltype(o::Ord, ps, ::Type{Pair{K,V}} ) where {K,V,Ord} = PriorityQueue{  K,  V,Ord}(o, ps)
_priority_queue_with_eltype(o::Ord, kv, ::Type{Tuple{K,V}}) where {K,V,Ord} = PriorityQueue{  K,  V,Ord}(o, kv)
_priority_queue_with_eltype(o::Ord, ps, ::Type{Pair{K}}   ) where {K,  Ord} = PriorityQueue{  K,Any,Ord}(o, ps)
_priority_queue_with_eltype(o::Ord, kv, ::Type            ) where {    Ord} = PriorityQueue{Any,Any,Ord}(o, kv)
length(pq::PriorityQueue) = length(pq.xs)
isempty(pq::PriorityQueue) = isempty(pq.xs)
haskey(pq::PriorityQueue, key) = haskey(pq.index, key)
"""
    peek(pq)
Return the lowest priority key from a priority queue without removing that
key from the queue.
"""
peek(pq::PriorityQueue) = pq.xs[1]
function percolate_down!(pq::PriorityQueue, i::Integer)
    x = pq.xs[i]
    @inbounds while (l = heapleft(i)) <= length(pq)
        r = heapright(i)
        j = r > length(pq) || lt(pq.o, pq.xs[l].second, pq.xs[r].second) ? l : r
        if lt(pq.o, pq.xs[j].second, x.second)
            pq.index[pq.xs[j].first] = i
            pq.xs[i] = pq.xs[j]
            i = j
        else
            break
        end
    end
    pq.index[x.first] = i
    pq.xs[i] = x
end
function percolate_up!(pq::PriorityQueue, i::Integer)
    x = pq.xs[i]
    @inbounds while i > 1
        j = heapparent(i)
        if lt(pq.o, x.second, pq.xs[j].second)
            pq.index[pq.xs[j].first] = i
            pq.xs[i] = pq.xs[j]
            i = j
        else
            break
        end
    end
    pq.index[x.first] = i
    pq.xs[i] = x
end
function force_up!(pq::PriorityQueue, i::Integer)
    x = pq.xs[i]
    @inbounds while i > 1
        j = heapparent(i)
        pq.index[pq.xs[j].first] = i
        pq.xs[i] = pq.xs[j]
        i = j
    end
    pq.index[x.first] = i
    pq.xs[i] = x
end
function getindex(pq::PriorityQueue{K,V}, key) where {K,V}
    pq.xs[pq.index[key]].second
end
function get(pq::PriorityQueue{K,V}, key, deflt) where {K,V}
    i = get(pq.index, key, 0)
    i == 0 ? deflt : pq.xs[i].second
end
function setindex!(pq::PriorityQueue{K, V}, value, key) where {K,V}
    if haskey(pq, key)
        i = pq.index[key]
        oldvalue = pq.xs[i].second
        pq.xs[i] = Pair{K,V}(key, value)
        if lt(pq.o, oldvalue, value)
            percolate_down!(pq, i)
        else
            percolate_up!(pq, i)
        end
    else
        enqueue!(pq, key, value)
    end
    value
end
"""
    enqueue!(pq, k=>v)
Insert the a key `k` into a priority queue `pq` with priority `v`.
```jldoctest
julia> a = PriorityQueue(PriorityQueue("a"=>1, "b"=>2, "c"=>3))
PriorityQueue{String,Int64,Base.Order.ForwardOrdering} with 3 entries:
  "c" => 3
  "b" => 2
  "a" => 1
julia> enqueue!(a, "d"=>4)
PriorityQueue{String,Int64,Base.Order.ForwardOrdering} with 4 entries:
  "c" => 3
  "b" => 2
  "a" => 1
  "d" => 4
```
"""
function enqueue!(pq::PriorityQueue{K,V}, pair::Pair{K,V}) where {K,V}
    key = pair.first
    if haskey(pq, key)
        throw(ArgumentError("PriorityQueue keys must be unique"))
    end
    push!(pq.xs, pair)
    pq.index[key] = length(pq)
    percolate_up!(pq, length(pq))
    return pq
end
"""
enqueue!(pq, k, v)
Insert the a key `k` into a priority queue `pq` with priority `v`.
"""
enqueue!(pq::PriorityQueue, key, value) = enqueue!(pq, key=>value)
enqueue!(pq::PriorityQueue{K,V}, kv) where {K,V} = enqueue!(pq, Pair{K,V}(kv.first, kv.second))
"""
    dequeue!(pq)
Remove and return the lowest priority key from a priority queue.
```jldoctest
julia> a = PriorityQueue(["a","b","c"],[2,3,1],Base.Order.Forward)
PriorityQueue{String,Int64,Base.Order.ForwardOrdering} with 3 entries:
  "c" => 1
  "b" => 3
  "a" => 2
julia> dequeue!(a)
"c"
julia> a
PriorityQueue{String,Int64,Base.Order.ForwardOrdering} with 2 entries:
  "b" => 3
  "a" => 2
```
"""
function dequeue!(pq::PriorityQueue)
    x = pq.xs[1]
    y = pop!(pq.xs)
    if !isempty(pq)
        pq.xs[1] = y
        pq.index[y.first] = 1
        percolate_down!(pq, 1)
    end
    delete!(pq.index, x.first)
    x.first
end
function dequeue!(pq::PriorityQueue, key)
    idx = pq.index[key]
    force_up!(pq, idx)
    dequeue!(pq)
    key
end
"""
    dequeue_pair!(pq)
Remove and return a the lowest priority key and value from a priority queue as a pair.
```jldoctest
julia> a = PriorityQueue(["a","b","c"],[2,3,1],Base.Order.Forward)
PriorityQueue{String,Int64,Base.Order.ForwardOrdering} with 3 entries:
  "c" => 1
  "b" => 3
  "a" => 2
julia> dequeue_pair!(a)
"c" => 1
julia> a
PriorityQueue{String,Int64,Base.Order.ForwardOrdering} with 2 entries:
  "b" => 3
  "a" => 2
```
"""
function dequeue_pair!(pq::PriorityQueue)
    x = pq.xs[1]
    y = pop!(pq.xs)
    if !isempty(pq)
        pq.xs[1] = y
        pq.index[y.first] = 1
        percolate_down!(pq, 1)
    end
    delete!(pq.index, x.first)
    x
end
function dequeue_pair!(pq::PriorityQueue, key)
    idx = pq.index[key]
    force_up!(pq, idx)
    dequeue_pair!(pq)
end
start(pq::PriorityQueue) = start(pq.index)
done(pq::PriorityQueue, i) = done(pq.index, i)
function next(pq::PriorityQueue{K,V}, i) where {K,V}
    (k, idx), i = next(pq.index, i)
    return (pq.xs[idx], i)
end
    # Deprecations
    # Remove when Julia 0.7 (or whatever version is after v0.6) is released
    @deprecate DefaultDictBase(default, ks::AbstractArray, vs::AbstractArray) DefaultDictBase(default, zip(ks, vs))
    @deprecate DefaultDictBase(default, ks, vs) DefaultDictBase(default, zip(ks, vs))
    @deprecate DefaultDictBase(::Type{K}, ::Type{V}, default) where {K,V} DefaultDictBase{K,V}(default)
    @deprecate DefaultDict(default, ks, vs) DefaultDict(default, zip(ks, vs))
    @deprecate DefaultDict(::Type{K}, ::Type{V}, default) where {K,V} DefaultDict{K,V}(default)
    @deprecate DefaultOrderedDict(default, ks, vs) DefaultOrderedDict(default, zip(ks, vs))
    @deprecate DefaultOrderedDict(::Type{K}, ::Type{V}, default) where {K,V} DefaultOrderedDict{K,V}(default)
    function SortedMultiDict(ks::AbstractVector{K},
                             vs::AbstractVector{V},
                             o::Ordering=Forward) where {K,V}
        Base.depwarn("SortedMultiDict(ks, vs, o::Ordering=Forward) is deprecated.\n" * "Use SortedMultiDict(o, zip(ks,vs)) or SortedMultiDict(zip(ks, vs))", :SortedMultiDict)
        if length(ks) != length(vs)
            throw(ArgumentError("SortedMultiDict(ks,vs,o): ks and vs arrays must be the same length"))
        end
        SortedMultiDict(o, zip(ks,vs))
    end
    @deprecate PriorityQueue(::Type{K}, ::Type{V}) where {K,V} PriorityQueue{K,V}()
    @deprecate PriorityQueue(::Type{K}, ::Type{V}, o::Ordering) where {K,V} PriorityQueue{K,V,typeof(o)}(o)
    @deprecate (PriorityQueue{K,V,ForwardOrdering}() where {K,V}) PriorityQueue{K,V}()
    function PriorityQueue(ks::AbstractVector{K},
                           vs::AbstractVector{V},
                           o::Ordering=Forward) where {K,V}
        Base.depwarn("PriorityQueue(ks, vs, o::Ordering=Forward) is deprecated.\n" * "Use PriorityQueue(o, zip(ks,vs)) or PriorityQueue(zip(ks, vs))", :PriorityQueue)
        if length(ks) != length(vs)
            throw(ArgumentError("PriorityQueue(ks,vs,o): ks and vs arrays must be the same length"))
        end
        PriorityQueue(o, zip(ks,vs))
    end
end
