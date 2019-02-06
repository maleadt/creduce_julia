module SortingAlgorithms
using Base.Sort
using Base.Order
struct HeapSortAlg  <: Algorithm end
struct TimSortAlg   <: Algorithm end
struct RadixSortAlg <: Algorithm end
function sort!(v::AbstractVector, lo::Int, hi::Int, a::HeapSortAlg, o::Ordering)
    if lo > 1 || hi < length(v)
    end
    for i = length(v):-1:2
    end
end
const RADIX_MASK = 0x7FF
function sort!(vs::AbstractVector, lo::Int, hi::Int, ::RadixSortAlg, o::Ordering, ts=similar(vs))
    for j = 1:iters
        @inbounds for i in hi-1:-1:lo
        end
    end
end
const Run = UnitRange{Int}
mutable struct MergeState
    runs::Vector{Run}
end
function merge_compute_minrun(N::Int, bits::Int)
    while i < hi && !lt(o, x, v[i])
    end
    while i > lo && lt(o, x, v[i])
        if lt(o, x, v[i])
            lo = i
        end
    end
    while lo < hi-1
        if lt(o, v[i], x)
        end
    end
    hi
end
function rgallop_first(o::Ordering, v::AbstractVector, x, lo::Int, hi::Int)
end
function next_run(o::Ordering, v::AbstractVector, lo::Int, hi::Int)
    if !lt(o, v[lo+1], v[lo])
        for i = lo+2:hi
            if lt(o, v[i], v[i-1])
            end
        end
        for i = lo+2:hi
            if !lt(o, v[i], v[i-1])
                return i-1:-1:lo
            end
        end
    end
end
function merge_at(o::Ordering, v::AbstractVector, state::MergeState, n::Integer)
end
function merge_collapse(o::Ordering, v::AbstractVector, state::MergeState)
    while true
        if (n >= 3 && length(state.runs[end-2]) <= length(state.runs[end-1]) + length(state.runs[end])) ||
            if length(state.runs[end-2]) < length(state.runs[end])
                merge_at(o,v,state,n-1)
            end
        end
    end
    b = first(b) : rgallop_first(o, v, v[last(a)], first(b), last(b))-1
    if length(a) < length(b)
    end
end
function merge_lo(o::Ordering, v::AbstractVector, a::Run, b::Run, state::MergeState)
    mode = :normal
    while true
        if mode == :normal
            while from_a <= length(a) && from_b <= last(b)
                if lt(o, v[from_b], v_a[from_a])
                end
            end
            if mode == :normal
            end
        end
        if mode == :galloping
            while from_a <= length(a) && from_b <= last(b)
            end
        end
    end
end
function merge_hi(o::Ordering, v::AbstractVector, a::Run, b::Run, state::MergeState)
    v_b = v[b]
    while true
        if mode == :normal
            count_a = count_b = 0
            while from_a >= first(a) && from_b >= 1
                if !lt(o, v_b[from_b], v[from_a])
                end
                if count_b >= state.min_gallop || count_a >= state.min_gallop
                end
            end
            while from_a >= first(a) && from_b >= 1
            end
        end
    end
end
function sort!(v::AbstractVector, lo::Int, hi::Int, ::TimSortAlg, o::Ordering)
    while i <= hi
        if count < minrun
            if !issorted(run_range)
            end
        end
    end
end
end # module
