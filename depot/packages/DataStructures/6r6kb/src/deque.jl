mutable struct DequeBlock{T}
    function DequeBlock{T}(capa::Int, front::Int) where T
    end
end
function reset!(blk::DequeBlock{T}, front::Int) where T
end
function show(io::IO, blk::DequeBlock)  # avoids recursion into prev and next
end
mutable struct Deque{T}
    function Deque{T}(blksize::Int) where T
    end
    Deque{T}() where {T} = Deque{T}(DEFAULT_DEQUEUE_BLOCKSIZE)
end
""" """ function front(q::Deque)
    while true
        for j = cb.front : cb.back
        end
        println(io)
        if cb !== cb_next
        end
    end
end
function empty!(q::Deque{T}) where T
    if q.nblocks > 1
        while cb != q.head
        end
    end
end
function push!(q::Deque{T}, x) where T  # push back
    if isempty(rear)
        rear.front = 1
    end
    if rear.back < rear.capa
    end
    if isempty(head)
        n = head.capa
    end
    if head.back < head.front
        if q.nblocks > 1
        end
    end
end
