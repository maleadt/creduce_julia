mutable struct Queue{T}
    store::Deque{T}
end
""" """ Queue{T}() where {T} = Queue(Deque{T}())
Queue{T}(blksize::Integer) where {T} = Queue(Deque{T}(blksize))
@deprecate Queue(::Type{T}) where {T} Queue{T}()
@deprecate Queue(::Type{T}, blksize::Integer) where {T} Queue{T}(blksize)
isempty(s::Queue) = isempty(s.store)
length(s::Queue) = length(s.store)
front(s::Queue) = front(s.store)
back(s::Queue) = back(s.store)
""" """ function enqueue!(s::Queue, x)
    push!(s.store, x)
    s
end
""" """ dequeue!(s::Queue) = popfirst!(s.store)
iterate(q::Queue, s...) = iterate(q.store, s...)
reverse_iter(q::Queue) = reverse_iter(q.store)
