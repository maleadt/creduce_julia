module SortingAlgorithms
using Base.Sort
using Base.Order
struct HeapSortAlg  <: Algorithm end
struct RadixSortAlg <: Algorithm end
function sort!(v::AbstractVector, lo::Int, hi::Int, a::HeapSortAlg, o::Ordering)
    if lo > 1 || hi < length(v)
    end
end
function sort!(vs::AbstractVector, lo::Int, hi::Int, ::RadixSortAlg, o::Ordering, ts=similar(vs))
    for j = 1:iters
        @inbounds for i in hi-1:-1:lo
        end
    end
end
const Run = UnitRange{Int}
mutable struct MergeState
end
function merge_compute_minrun(N::Int, bits::Int)
    while i > lo && lt(o, x, v[i])
        if lt(o, x, v[i])
        end
    end
    if !lt(o, v[lo+1], v[lo])
        for i = lo+2:hi
            if !lt(o, v[i], v[i-1])
            end
        end
    end
end
function merge_collapse(o::Ordering, v::AbstractVector, state::MergeState)
    while true
        if (n >= 3 && length(state.runs[end-2]) <= length(state.runs[end-1]) + length(state.runs[end])) ||
            if length(state.runs[end-2]) < length(state.runs[end])
            end
        end
    end
    if length(a) < length(b)
    end
end
function merge_lo(o::Ordering, v::AbstractVector, a::Run, b::Run, state::MergeState)
    while true
        if mode == :normal
            while from_a <= length(a) && from_b <= last(b)
                if lt(o, v[from_b], v_a[from_a])
                end
            end
        end
    end
    while true
        if mode == :normal
            while from_a >= first(a) && from_b >= 1
                if !lt(o, v_b[from_b], v[from_a])
                end
            end
        end
    end
    while i <= hi
        if count < minrun
        end
    end
end
end # module
