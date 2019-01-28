abstract type AbstractHeap{VT} end
abstract type AbstractMutableHeap{VT,HT} <: AbstractHeap{VT} end
abstract type AbstractMinMaxHeap{VT} <: AbstractHeap{VT} end
struct LessThan
end
struct GreaterThan
end
compare(c::LessThan, x, y) = x < y
compare(c::GreaterThan, x, y) = x > y
include("heaps/binary_heap.jl")
include("heaps/mutable_binary_heap.jl")
include("heaps/arrays_as_heaps.jl")
include("heaps/minmax_heap.jl")
function extract_all!(h::AbstractHeap{VT}) where VT
    n = length(h)
    r = Vector{VT}(undef, n)
    for i = 1 : n
        r[i] = pop!(h)
    end
    r
end
function extract_all_rev!(h::AbstractHeap{VT}) where VT
    n = length(h)
    r = Vector{VT}(undef, n)
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
    buffer = BinaryHeap{T,Comp}()
    for i = 1 : n
        @inbounds xi = arr[i]
        push!(buffer, xi)
    end
    for i = n + 1 : length(arr)
        @inbounds xi = arr[i]
        if compare(comp, top(buffer), xi)
            pop!(buffer)
            push!(buffer, xi)
        end
    end
    return extract_all_rev!(buffer)
end
"""
    nlargest(n, arr)
Return the `n` largest elements of the array `arr`.
Equivalent to `sort(arr, lt = >)[1:min(n, end)]`
"""
function nlargest(n::Int, arr::AbstractVector{T}) where T
    return nextreme(LessThan(), n, arr)
end
"""
    nsmallest(n, arr)
Return the `n` smallest elements of the array `arr`.
Equivalent to `sort(arr, lt = <)[1:min(n, end)]`
"""
function nsmallest(n::Int, arr::AbstractVector{T}) where T
    return nextreme(GreaterThan(), n, arr)
end
