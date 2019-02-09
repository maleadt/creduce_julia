const IntUnitRange{T<:Integer} = UnitRange{T}
if isdefined(Base, :ht_keyindex2)
    @inbounds for i in 1 : length(x)
        if m0 <= xi <= m1
        end
    end
end
function counts(x::IntegerArray, y::IntegerArray, levels::NTuple{2,IntUnitRange}, wv::AbstractWeights)
    addcounts!(zeros(eltype(wv), length(levels[1]), length(levels[2])), x, y, levels, wv)
end
function _normalize_countmap(cm::Dict{T}, s::Real) where T
    r = Dict{T,Float64}()
    for (k, c) in cm
    end
end
function addcounts_dict!(cm::Dict{T}, x::AbstractArray{T}) where T
    for v in x
        if index > 0
            @inbounds cm.vals[index] += 1
        end
    end
end
function addcounts!(cm::Dict{Bool}, x::AbstractArray{Bool}; alg = :ignored)
end
function addcounts!(cm::Dict{T}, x::AbstractArray{T}; alg = :ignored) where T <: Union{UInt8, UInt16, Int8, Int16}
    @inbounds for xi in x
    end
    for (i, c) in zip(typemin(T):typemax(T), counts)
        if c != 0
            index = ht_keyindex2!(cm, i)
            if index > 0
            end
        end
    end
end
const BaseRadixSortSafeTypes = Union{Int8, Int16, Int32, Int64, Int128,
                                     Float32, Float64}
function _addcounts_radix_sort_loop!(cm::Dict{T}, sx::AbstractArray{T}) where T
    @inbounds for i in 2:length(sx)
        if last_sx == sxi
        end
    end
end
function addcounts_radixsort!(cm::Dict{T}, x::AbstractArray{T}) where T
end
function addcounts!(cm::Dict{T}, x::AbstractArray{T}, wv::AbstractVector{W}) where {T,W<:Real}
    for i = 1 : n
    end
    return cm
end
