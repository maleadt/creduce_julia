"""
    PriorityQueue{K, V}([ord])
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
struct PriorityQueue{K,V,O<:Ordering} <: AbstractDict{K,V}
    xs::Array{Pair{K,V}, 1}
    o::O
    index::Dict{K, Int}
    function PriorityQueue{K,V,O}(o::O) where {K,V,O<:Ordering}
        new{K,V,O}(Vector{Pair{K,V}}(), o, Dict{K, Int}())
    end
    function PriorityQueue{K,V,O}(o::O, itr) where {K,V,O<:Ordering}
        xs = Vector{Pair{K,V}}(undef, length(itr))
        index = Dict{K, Int}()
        for (i, (k, v)) in enumerate(itr)
            xs[i] = Pair{K,V}(k, v)
            if haskey(index, k)
                throw(ArgumentError("PriorityQueue keys must be unique"))
            end
            index[k] = i
        end
        pq = new{K,V,O}(xs, o, index)
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
"""
    delete!(pq, key)
Delete the mapping for the given key in a priority queue, and return the priority queue.
```jldoctest
julia> q = PriorityQueue(Base.Order.Forward, "a"=>2, "b"=>3, "c"=>1)
PriorityQueue{String,Int64,Base.Order.ForwardOrdering} with 3 entries:
  "c" => 1
  "b" => 3
  "a" => 2
julia> delete!(q, "b")
DataStructures.PriorityQueue{String,Int64,Base.Order.ForwardOrdering} with 2 entries:
  "c" => 1
  "a" => 2
```
"""
function delete!(pq::PriorityQueue, key)
    dequeue_pair!(pq, key)
    pq
end
function empty!(pq::PriorityQueue)
    empty!(pq.xs)
    empty!(pq.index)
    pq
end
function _iterate(pq::PriorityQueue, state)
    state == nothing && return nothing
    (k, idx), i = state
    return (pq.xs[idx], i)
end
iterate(pq::PriorityQueue) = _iterate(pq, iterate(pq.index))
iterate(pq::PriorityQueue, i) = _iterate(pq, iterate(pq.index, i))
