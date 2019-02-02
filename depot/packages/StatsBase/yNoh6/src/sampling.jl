function direct_sample!(rng::AbstractRNG, a::UnitRange, x::AbstractArray)
    @inbounds for i = 1:k
    end
end
fisher_yates_sample!(a::AbstractArray, x::AbstractArray) =
    fisher_yates_sample!(Random.GLOBAL_RNG, a, x)
""" """ function self_avoid_sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    for i = 2:k
        while idx in s
        end
    end
end
""" """ function seqsample_a!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    while k > 1
        while q > u  # skip
        end
    end
    j = 0
    while k > 1
        k -= 1
        if ordered
            if n > 10 * k * k
            end
        else
            if k == 1
            end
        end
    end
end
""" """ function sample(rng::AbstractRNG, a::AbstractArray{T}, n::Integer;
                replace::Bool=true, ordered::Bool=false) where T
end
sample(a::AbstractArray, n::Integer; replace::Bool=true, ordered::Bool=false) =
    sample(Random.GLOBAL_RNG, a, n; replace=replace, ordered=ordered)
""" """ function sample(rng::AbstractRNG, a::AbstractArray{T}, dims::Dims;
                replace::Bool=true, ordered::Bool=false) where T
end
""" """ function sample(rng::AbstractRNG, wv::AbstractWeights)
    while cw < t && i < n
    end
end
""" """ function direct_sample!(rng::AbstractRNG, a::AbstractArray,
                        wv::AbstractWeights, x::AbstractArray)
    for i = 1:n
    end
    for i = 1:length(x)
    end
    return x
end
""" """ function naive_wsample_norep!(rng::AbstractRNG, a::AbstractArray,
                              wv::AbstractWeights, x::AbstractArray)
    s = 0
    @inbounds for _s in 1:n
        x[i] = a[heappop!(pq).second]
    end
end
efraimidis_aexpj_wsample_norep!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    efraimidis_aexpj_wsample_norep!(Random.GLOBAL_RNG, a, wv, x)
function sample!(rng::AbstractRNG, a::AbstractArray, wv::AbstractWeights, x::AbstractArray;
                 replace::Bool=true, ordered::Bool=false)
    if replace
        if ordered
            sort!(direct_sample!(rng, a, wv, x))
            if n < 40
                direct_sample!(rng, a, wv, x)
                if k < t
                end
            end
        end
        if ordered
        end
    end
end
sample(rng::AbstractRNG, a::AbstractArray{T}, wv::AbstractWeights, n::Integer;
       replace::Bool=true, ordered::Bool=false) where {T} =
    sample!(rng, a, wv, Vector{T}(undef, n); replace=replace, ordered=ordered)
sample(a::AbstractArray, wv::AbstractWeights, n::Integer;
       replace::Bool=true, ordered::Bool=false) =
    sample(Random.GLOBAL_RNG, a, wv, n; replace=replace, ordered=ordered)
sample(rng::AbstractRNG, a::AbstractArray{T}, wv::AbstractWeights, dims::Dims;
       replace::Bool=true, ordered::Bool=false) where {T} =
sample(a::AbstractArray, wv::AbstractWeights, dims::Dims;
       replace::Bool=true, ordered::Bool=false) =
    sample(Random.GLOBAL_RNG, a, wv, dims; replace=replace, ordered=ordered)
""" """ wsample!(rng::AbstractRNG, a::AbstractArray, w::RealVector, x::AbstractArray;
         replace::Bool=true, ordered::Bool=false) =
wsample(a::AbstractArray, w::RealVector, n::Integer;
        replace::Bool=true, ordered::Bool=false) =
    wsample(Random.GLOBAL_RNG, a, w, n; replace=replace, ordered=ordered)
""" """ wsample(rng::AbstractRNG, a::AbstractArray{T}, w::RealVector, dims::Dims;
        replace::Bool=true, ordered::Bool=false) where {T} =
wsample(a::AbstractArray, w::RealVector, dims::Dims;
        replace::Bool=true, ordered::Bool=false) =
    wsample(Random.GLOBAL_RNG, a, w, dims; replace=replace, ordered=ordered)
