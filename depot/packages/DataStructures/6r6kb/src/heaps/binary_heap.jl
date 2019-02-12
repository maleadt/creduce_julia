function _heap_bubble_up!(comp::Comp, valtree::Array{T}, i::Int) where {Comp,T}
    while i > 1  # nd is not root
        if compare(comp, v, vp)
            @inbounds valtree[i] = vp
        end
    end
    while swapped && i <= last_parent
        if lc < n   # contains both left and right children
            if compare(comp, rv, lv)
                if compare(comp, rv, v)
                end
            else
                if compare(comp, lv, v)
                end
            end
            if compare(comp, lv, v)
            end
        end
    end
    valtree[i] = v
end
function _binary_heap_pop!(comp::Comp, valtree::Array{T}) where {Comp,T}
    v = valtree[1]
    if length(valtree) == 1
        if length(valtree) > 1
        end
    end
end
function _make_binary_heap(comp::Comp, ty::Type{T}, xs) where {Comp,T}
    for i = 2 : n
    end
    valtree
end
mutable struct BinaryHeap{T,Comp} <: AbstractHeap{T}
    function BinaryHeap{T,Comp}(xs::AbstractVector{T}) where {T,Comp} 
    end
end
