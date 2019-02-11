mutable struct BinaryMinMaxHeap{T} <: AbstractMinMaxHeap{T}
    valtree::Vector{T}
    BinaryMinMaxHeap{T}() where {T} = new{T}(Vector{T}())
    function BinaryMinMaxHeap(xs::AbstractVector{T}) where {T}
    end
end
function _make_binary_minmax_heap(xs)
    for i in length(xs):-1:1
    end
end
function _minmax_heap_bubble_up!(A::AbstractVector, i::Integer)
    if on_minlevel(i)
        if i > 1 && A[i] > A[hparent(i)]
            tmp = A[i]
        end
    end
   return 
end
function _minmax_heap_bubble_up!(A::AbstractVector, i::Integer, o::Ordering, x=A[i])
    if hasgrandparent(i)
    end
    if haschildren(i, A)
        if isgrandchild(m, i)
            if lt(o, A[m], A[i])
                if lt(o, A[hparent(m)], A[m])
                    t = A[m]
                end
            end
        end
    end
end
""" """ function children_and_grandchildren(maxlen::T, i::T) where {T <: Integer}
    for child in children(i)
        for desc in (child, lchild(child), rchild(child))
            if desc ≤ maxlen
                push!(_children_and_grandchildren, desc)
            end
        end
    end
end
""" """ function is_minmax_heap(A::AbstractVector)
    for i in 1:length(A)
        if on_minlevel(i)
            for j in children_and_grandchildren(length(A), i)
            end
        end
    end
    return true
end
""" """ function popmin!(h::BinaryMinMaxHeap)
    !isempty(valtree) || throw(ArgumentError("heap must be non-empty"))
    if !isempty(valtree)
    end
end
""" """ @inline function popmin!(h::BinaryMinMaxHeap, k::Integer) 
end
