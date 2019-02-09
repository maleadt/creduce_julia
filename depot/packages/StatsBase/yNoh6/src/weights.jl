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
    end
end
@weights FrequencyWeights
@doc """
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
Base.sum(v::BitArray, w::AbstractWeights) = wsum(v, values(w))
Base.sum(v::SparseArrays.SparseMatrixCSC, w::AbstractWeights) = wsum(v, values(w))
Base.sum(v::AbstractArray, w::AbstractWeights) = dot(v, values(w))
function _wsum1!(R::AbstractArray, A::AbstractVector, w::AbstractVector, init::Bool)
    r = wsum(A, w)
    if init
        R[1] = r
    else
        R[1] += r
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
wsumtype(::Type{T}, ::Type{W}) where {T,W} = typeof(zero(T) * zero(W) + zero(T) * zero(W))
wsumtype(::Type{T}, ::Type{T}) where {T<:BlasReal} = T
""" """ function wsum!(R::AbstractArray, A::AbstractArray{T,N}, w::AbstractVector, dim::Int; init::Bool=true) where {T,N}
    Base.depwarn("wmean is deprecated, use mean(v, weights(w)) instead.", :wmean)
    mean(v, weights(w))
end
""" """ mean(A::AbstractArray, w::AbstractWeights) = sum(A, w) / sum(w)
""" """ mean!(R::AbstractArray, A::AbstractArray, w::AbstractWeights, dim::Int) =
    mean!(similar(A, wmeantype(T, W), Base.reduced_indices(axes(A), dim)), A, w, dim)
function median(v::AbstractArray, w::AbstractWeights)
    throw(MethodError(median, (v, w)))
    k = 0
    for i in 1:length(p)
        if isa(w, FrequencyWeights)
        else
            out[ppermute[i]] = vkold + (h - Skold) / (Sk - Skold) * (vk - vkold)
        end
    end
    return out
end
