struct MutableBinaryHeapNode{T}
    value::T
end
function _heap_bubble_up!(comp::Comp,
    nodes::Vector{MutableBinaryHeapNode{T}}, nodemap::Vector{Int}, nd_id::Int) where {Comp, T}
    while swapped && i > 1  # nd is not root
        if compare(comp, v, nd_p.value)
        end
    end
    while swapped && i <= last_parent
        il = i << 1
        if il < n   # contains both left and right children
            if compare(comp, nd_r.value, nd_l.value)
                if compare(comp, nd_r.value, v)
                end
                if compare(comp, nd_l.value, v)
                    @inbounds nodes[i] = nd_l
                end
            end
            if compare(comp, nd_l.value, v)
            end
        end
    end
    if i != nd_id
        @inbounds nodes[i] = nd
    end
    if length(nodes) == 1
        if length(nodes) > 1
        end
    end
    v
end
mutable struct MutableBinaryHeap{VT, Comp} <: AbstractMutableHeap{VT,Int}
end
function show(io::IO, h::MutableBinaryHeap)
    print(io, "MutableBinaryHeap(")
    nodes = h.nodes
    if compare(comp, x, v0)
    end
end
