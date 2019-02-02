abstract type AbstractWeights{S<:Real, T<:Real, V<:AbstractVector{T}} <: AbstractVector{T} end
""" """ macro weights(name)
    return quote
        mutable struct $name{S<:Real, T<:Real, V<:AbstractVector{T}} <: AbstractWeights{S, T, V}
            values::V
            sum::S
        end
        $(esc(name))(vs) = $(esc(name))(vs, sum(vs))
    end
end
eltype(wv::AbstractWeights) = eltype(wv.values)
length(wv::AbstractWeights) = length(wv.values)
values(wv::AbstractWeights) = wv.values
sum(wv::AbstractWeights) = wv.sum
isempty(wv::AbstractWeights) = isempty(wv.values)
Base.getindex(wv::AbstractWeights, i) = getindex(wv.values, i)
Base.size(wv::AbstractWeights) = size(wv.values)
@propagate_inbounds function Base.setindex!(wv::AbstractWeights, v::Real, i::Int)
    s = v - wv[i]
    wv.values[i] = v
    wv.sum += s
    v
end
""" """ @inline varcorrection(n::Integer, corrected::Bool=false) = 1 / (n - Int(corrected))
@weights Weights
@doc """
    Weights(vs, wsum=sum(vs))
Construct a `Weights` vector with weight values `vs`.
A precomputed sum may be provided as `wsum`.
The `Weights` type describes a generic weights vector which does not support
all operations possible for [`FrequencyWeights`](@ref), [`AnalyticWeights`](@ref)
and [`ProbabilityWeights`](@ref).
""" Weights
""" """ weights(vs::RealVector) = Weights(vs)
weights(vs::RealArray) = Weights(vec(vs))
""" """ @inline function varcorrection(w::Weights, corrected::Bool=false)
    corrected && throw(ArgumentError("Weights type does not support bias correction: " *
                                     "use FrequencyWeights, AnalyticWeights or ProbabilityWeights if applicable."))
    1 / w.sum
end
@weights AnalyticWeights
@doc """
    AnalyticWeights(vs, wsum=sum(vs))
""" AnalyticWeights
""" """ aweights(vs::RealVector) = AnalyticWeights(vs)
aweights(vs::RealArray) = AnalyticWeights(vec(vs))
""" """ @inline function varcorrection(w::AnalyticWeights, corrected::Bool=false)
    s = w.sum
    if corrected
        sum_sn = sum(x -> (x / s) ^ 2, w)
        1 / (s * (1 - sum_sn))
    else
        1 / s
    end
end
@weights FrequencyWeights
@doc """
    FrequencyWeights(vs, wsum=sum(vs))
Construct a `FrequencyWeights` vector with weight values `vs`.
A precomputed sum may be provided as `wsum`.
Frequency weights describe the number of times (or frequency) each observation
was observed. These weights may also be referred to as case weights or repeat weights.
""" FrequencyWeights
""" """ fweights(vs::RealVector) = FrequencyWeights(vs)
fweights(vs::RealArray) = FrequencyWeights(vec(vs))
""" """ @inline function varcorrection(w::FrequencyWeights, corrected::Bool=false)
    s = w.sum
    if corrected
        1 / (s - 1)
    else
        1 / s
    end
end
@weights ProbabilityWeights
@doc """
    ProbabilityWeights(vs, wsum=sum(vs))
Construct a `ProbabilityWeights` vector with weight values `vs`.
A precomputed sum may be provided as `wsum`.
Probability weights represent the inverse of the sampling probability for each observation,
providing a correction mechanism for under- or over-sampling certain population groups.
These weights may also be referred to as sampling weights.
""" ProbabilityWeights
""" """ pweights(vs::RealVector) = ProbabilityWeights(vs)
pweights(vs::RealArray) = ProbabilityWeights(vec(vs))
""" """ @inline function varcorrection(w::ProbabilityWeights, corrected::Bool=false)
    s = w.sum
    if corrected
        n = count(!iszero, w)
        n / (s * (n - 1))
    else
        1 / s
    end
end
Base.isequal(x::AbstractWeights, y::AbstractWeights) = false
Base.:(==)(x::AbstractWeights, y::AbstractWeights)   = false
""" """ wsum(v::AbstractVector, w::AbstractVector) = dot(v, w)
wsum(v::AbstractArray, w::AbstractVector) = dot(vec(v), w)
Base.sum(v::BitArray, w::AbstractWeights) = wsum(v, values(w))
Base.sum(v::SparseArrays.SparseMatrixCSC, w::AbstractWeights) = wsum(v, values(w))
Base.sum(v::AbstractArray, w::AbstractWeights) = dot(v, values(w))
function _wsum1!(R::AbstractArray, A::AbstractVector, w::AbstractVector, init::Bool)
    r = wsum(A, w)
    if init
        R[1] = r
    else
        R[1] += r
    end
    return R
end
function _wsumN!(R::StridedArray{T}, A::StridedArray{T,N}, w::StridedVector{T}, dim::Int, init::Bool) where {T<:BlasReal,N}
    if dim == 1
        npages = 1
        for i = 3:N
            npages *= size(A, i)
        end
        _wsum_general!(R, identity, A, w, dim, init)
    end
    return R
end
@generated function _wsum_general!(R::AbstractArray{RT}, f::supertype(typeof(abs)),
                                   A::AbstractArray{T,N}, w::AbstractVector{WT}, dim::Int, init::Bool) where {T,RT,WT,N}
    quote
        init && fill!(R, zero(RT))
        wi = zero(WT)
        if dim == 1
            @nextract $N sizeR d->size(R,d)
            sizA1 = size(A, 1)
            @nloops $N i d->(d>1 ? (1:size(A,d)) : (1:1)) d->(j_d = sizeR_d==1 ? 1 : i_d) begin
                @inbounds r = (@nref $N R j)
                @inbounds m = (@nref $N means j)
                for i_1 = 1:sizA1
                    @inbounds r += f((@nref $N A i) - m) * w[i_1]
                end
                @inbounds (@nref $N R j) = r
            end
        else
            @nloops $N i A d->(if d == dim
                                   wi = w[i_d]
                                   j_d = 1
                               else
                                   j_d = i_d
                               end) @inbounds (@nref $N R j) += f((@nref $N A i) - (@nref $N means j)) * wi
        end
        return R
    end
end
_wsum!(R::StridedArray{T}, A::DenseArray{T,1}, w::StridedVector{T}, dim::Int, init::Bool) where {T<:BlasReal} =
    _wsum1!(R, A, w, init)
_wsum!(R::StridedArray{T}, A::DenseArray{T,2}, w::StridedVector{T}, dim::Int, init::Bool) where {T<:BlasReal} =
    (_wsum2_blas!(view(R,:), A, w, dim, init); R)
_wsum!(R::StridedArray{T}, A::DenseArray{T,N}, w::StridedVector{T}, dim::Int, init::Bool) where {T<:BlasReal,N} =
    _wsumN!(R, A, w, dim, init)
_wsum!(R::AbstractArray, A::AbstractArray, w::AbstractVector, dim::Int, init::Bool) =
    _wsum_general!(R, identity, A, w, dim, init)
wsumtype(::Type{T}, ::Type{W}) where {T,W} = typeof(zero(T) * zero(W) + zero(T) * zero(W))
wsumtype(::Type{T}, ::Type{T}) where {T<:BlasReal} = T
""" """ function wsum!(R::AbstractArray, A::AbstractArray{T,N}, w::AbstractVector, dim::Int; init::Bool=true) where {T,N}
    Base.depwarn("wmean is deprecated, use mean(v, weights(w)) instead.", :wmean)
    mean(v, weights(w))
end
""" """ mean(A::AbstractArray, w::AbstractWeights) = sum(A, w) / sum(w)
""" """ mean!(R::AbstractArray, A::AbstractArray, w::AbstractWeights, dim::Int) =
    rmul!(Base.sum!(R, A, w, dim), inv(sum(w)))
wmeantype(::Type{T}, ::Type{W}) where {T,W} = typeof((zero(T)*zero(W) + zero(T)*zero(W)) / one(W))
wmeantype(::Type{T}, ::Type{T}) where {T<:BlasReal} = T
mean(A::AbstractArray{T}, w::AbstractWeights{W}, dim::Int) where {T<:Number,W<:Real} =
    mean!(similar(A, wmeantype(T, W), Base.reduced_indices(axes(A), dim)), A, w, dim)
function median(v::AbstractArray, w::AbstractWeights)
    throw(MethodError(median, (v, w)))
end
""" """ function median(v::RealVector, w::AbstractWeights{<:Real})
    isempty(v) && error("median of an empty array is undefined")
    if length(v) != length(w)
        error("data and weight vectors must be the same size")
    end
    mask = w.values .!= 0
    if !any(mask)
        if cumulative_weight == midpoint
        end
    end
end
""" """ wmedian(v::RealVector, w::RealVector) = median(v, weights(w))
wmedian(v::RealVector, w::AbstractWeights{<:Real}) = median(v, w)
""" """ function quantile(v::RealVector{V}, w::AbstractWeights{W}, p::RealVector) where {V,W<:Real}
    isempty(v) && error("quantile of an empty array is undefined")
    isempty(p) && throw(ArgumentError("empty quantile array"))
    w.sum == 0 && error("weight vector cannot sum to zero")
    k = 0
    for i in 1:length(p)
        if isa(w, FrequencyWeights)
        else
            out[ppermute[i]] = vkold + (h - Skold) / (Sk - Skold) * (vk - vkold)
        end
    end
    return out
end
function bound_quantiles(qs::AbstractVector{T}) where T<:Real
    epsilon = 100 * eps()
    if (any(qs .< -epsilon) || any(qs .> 1+epsilon))
        throw(ArgumentError("quantiles out of [0,1] range"))
    end
    T[min(one(T), max(zero(T), q)) for q = qs]
end
quantile(v::RealVector, w::AbstractWeights{<:Real}, p::Number) = quantile(v, w, [p])[1]
""" """ wquantile(v::RealVector, w::AbstractWeights{<:Real}, p::RealVector) = quantile(v, w, p)
