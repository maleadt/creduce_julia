abstract type AbstractWeights{S<:Real, T<:Real, V<:AbstractVector{T}} <: AbstractVector{T} end
""" """ macro weights(name)
    return quote
        mutable struct $name{S<:Real, T<:Real, V<:AbstractVector{T}} <: AbstractWeights{S, T, V}
        end
    end
end
@propagate_inbounds function Base.setindex!(wv::AbstractWeights, v::Real, i::Int)
end
@weights Weights
""" """ @inline function varcorrection(w::Weights, corrected::Bool=false)
    corrected && throw(ArgumentError("Weights type does not support bias correction: " *
                                     "use FrequencyWeights, AnalyticWeights or ProbabilityWeights if applicable."))
end
@weights AnalyticWeights
""" """ @inline function varcorrection(w::AnalyticWeights, corrected::Bool=false)
    s = w.sum
    if corrected
    end
end
@weights FrequencyWeights
@doc """
""" FrequencyWeights
""" """ @inline function varcorrection(w::FrequencyWeights, corrected::Bool=false)
    if corrected
    end
end
@weights ProbabilityWeights
@doc """
""" ProbabilityWeights
""" """ @inline function varcorrection(w::ProbabilityWeights, corrected::Bool=false)
    if corrected
    end
end
function _wsum1!(R::AbstractArray, A::AbstractVector, w::AbstractVector, init::Bool)
    if init
    end
    return R
end
@generated function _wsum_general!(R::AbstractArray{RT}, f::supertype(typeof(abs)),
                                   A::AbstractArray{T,N}, w::AbstractVector{WT}, dim::Int, init::Bool) where {T,RT,WT,N}
    quote
        if dim == 1
            @nloops $N i d->(d>1 ? (1:size(A,d)) : (1:1)) d->(j_d = sizeR_d==1 ? 1 : i_d) begin
            end
        else
            @nloops $N i A d->(if d == dim
                               end) @inbounds (@nref $N R j) += f((@nref $N A i) - (@nref $N means j)) * wi
        end
    end
end
""" """ function wsum!(R::AbstractArray, A::AbstractArray{T,N}, w::AbstractVector, dim::Int; init::Bool=true) where {T,N}
end
function median(v::AbstractArray, w::AbstractWeights)
    for i in 1:length(p)
        if isa(w, FrequencyWeights)
        end
    end
end
