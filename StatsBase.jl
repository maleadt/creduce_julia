__precompile__()
module StatsBase
    import Base: length, isempty, eltype, values, sum, mean, mean!, show, quantile
    import Base: rand, rand!
    import Base.LinAlg: BlasReal, BlasFloat
    import Base.Cartesian: @nloops, @nref, @nextract
    import DataStructures: heapify!, heappop!, percolate_down!
    # import SpecialFunctions: erfcinv
    using Compat, SortingAlgorithms, Missings
    if VERSION >= v"0.7.0-DEV.3052"
        using Printf
    end
    ## tackle compatibility issues
    export
    ## weights
    AbstractWeights,    # abstract type to represent any weight vector
    Weights,            # to represent a generic weight vector
    AnalyticWeights,    # to represent an analytic/precision/reliability weight vector
    FrequencyWeights,   # to representing a frequency/case/repeat weight vector
    ProbabilityWeights, # to representing a probability/sampling weight vector
    weights,            # construct a generic Weights vector
    aweights,           # construct an AnalyticWeights vector
    fweights,           # construct a FrequencyWeights vector
    pweights,           # construct a ProbabilityWeights vector
    wsum,               # weighted sum with vector as second argument
    wsum!,              # weighted sum across dimensions with provided storage
    wmean,              # weighted mean
    wmean!,             # weighted mean across dimensions with provided storage
    wmedian,            # weighted median
    wquantile,          # weighted quantile
    ## moments
    skewness,       # (standardized) skewness
    kurtosis,       # (excessive) kurtosis
    moment,         # central moment of given order
    mean_and_var,   # (mean, var)
    mean_and_std,   # (mean, std)
    mean_and_cov,   # (mean, cov)
    ## scalarstats
    geomean,     # geometric mean
    harmmean,    # harmonic mean
    genmean,     # generalized/power mean
    middle,      # the mean of two real numbers
    mode,        # find a mode from data (the first one)
    modes,       # find all modes from data
    zscore,      # compute Z-scores
    zscore!,     # compute Z-scores inplace or to a pre-allocated array
    percentile,  # quantile using percentage (instead of fraction) as argument
    nquantile,   # quantiles at [0:n]/n
    span,        # The range minimum(x):maximum(x)
    variation,   # ratio of standard deviation to mean
    sem,         # standard error of the mean, i.e. sqrt(var / n)
    mad,         # median absolute deviation
    iqr,         # interquatile range
    entropy,        # the entropy of a probability vector
    renyientropy,   # the Rényi (generalised) entropy of a probability vector
    crossentropy,   # cross entropy between two probability vectors
    kldivergence,   # K-L divergence between two probability vectors
    summarystats,   # summary statistics
    describe,       # print the summary statistics
    # deviation
    counteq,        # count the number of equal pairs
    countne,        # count the number of non-equal pairs
    sqL2dist,       # squared L2 distance between two arrays
    L2dist,         # L2 distance between two arrays
    L1dist,         # L1 distance between two arrays
    Linfdist,       # L-inf distance between two arrays
    gkldiv,         # (Generalized) Kullback-Leibler divergence between two vectors
    meanad,         # mean absolute deviation
    maxad,          # maximum absolute deviation
    msd,            # mean squared deviation
    rmsd,           # root mean squared deviation
    psnr,           # peak signal-to-noise ratio (in dB)
    # cov
    scattermat,     # scatter matrix (i.e. unnormalized covariance)
    cov2cor,        # converts a covariance matrix to a correlation matrix
    cor2cov,        # converts a correlation matrix to a covariance matrix
    ## counts
    addcounts!,     # add counts to an accumulating array or map
    counts,         # count integer values in given arrays
    proportions,    # proportions of integer values in given arrays
                    # (normalized version of counts)
    countmap,       # count distinct values and return a map
    proportionmap,  # proportions of distinct values returned as a map
    ## ranking
    ordinalrank,    # ordinal ranking ("1234" ranking)
    competerank,    # competition ranking ("1 2 2 4" ranking)
    denserank,      # dense ranking ("1 2 2 3" ranking)
    tiedrank,       # tied ranking ("1 2.5 2.5 4" ranking)
    ## rankcorr
    corspearman,       # spearman's rank correlation
    corkendall,        # kendall's rank correlation
    ## signalcorr
    autocov!, autocov,      # auto covariance
    autocor!, autocor,      # auto correlation
    crosscov!, crosscov,    # cross covariance
    crosscor!, crosscor,    # cross correlation
    pacf!, pacf,            # partial auto-correlation
    ## sampling
    samplepair,     # draw a pair of distinct elements   
    sample,         # sampling from a population
    sample!,        # sampling from a population, with pre-allocated output
    wsample,        # sampling from a population with weights
    wsample!,       # weighted sampling, with pre-allocated output
    ## empirical
    ecdf,           # empirical cumulative distribution function
    AbstractHistogram,
    Histogram,
    hist,
    # histrange,
    ## robust
    trim,           # trimmed set
    trim!,          # trimmed set
    winsor,         # Winsorized set
    winsor!,        # Winsorized set
    trimvar,        # variance of the mean of a trimmed set
    ## misc
    rle,            # run-length encoding
    inverse_rle,    # inverse run-length encoding
    indexmap,       # construct a map from element to index
    levelsmap,      # construct a map from n unique elements to [1, ..., n]
    findat,         # find the position within a for elements in b
    indicatormat,   # construct indicator matrix
    # statistical models
    CoefTable,
    StatisticalModel,
    RegressionModel,
    adjr2,
    adjr²,
    aic,
    aicc,
    bic,
    coef,
    coefnames,
    coeftable,
    confint,
    deviance,
    dof,
    dof_residual,
    fit,
    fit!,
    fitted,
    loglikelihood,
    modelmatrix,
    nobs,
    nulldeviance,
    nullloglikelihood,
    stderr,
    vcov,
    predict,
    predict!,
    residuals,
    r2,
    r²,
    model_response,
    ConvergenceException
@static if !isdefined(Base, :midpoints)
    export midpoints
end
    # source files
const RealArray{T<:Real,N} = AbstractArray{T,N}
const RealVector{T<:Real} = AbstractArray{T,1}
const RealMatrix{T<:Real} = AbstractArray{T,2}
const IntegerArray{T<:Integer,N} = AbstractArray{T,N}
const IntegerVector{T<:Integer} = AbstractArray{T,1}
const IntegerMatrix{T<:Integer} = AbstractArray{T,2}
const RealFP = Union{Float32, Float64}
fptype(::Type{T}) where {T<:Union{Float32,Bool,Int8,UInt8,Int16,UInt16}} = Float32
fptype(::Type{T}) where {T<:Union{Float64,Int32,UInt32,Int64,UInt64,Int128,UInt128}} = Float64
fptype(::Type{Complex64}) = Complex64
fptype(::Type{Complex128}) = Complex128
const DepBool = Union{Bool, Void}
function depcheck(fname::Symbol, b::DepBool)
    if b == nothing
        msg = "$fname will default to corrected=true in the future. Use corrected=false for previous behaviour."
        Base.depwarn(msg, fname)
        false
    else
        b
    end
end
if !isdefined(Base, :axes)
    const axes = Base.indices
end
abstract type AbstractWeights{S<:Real, T<:Real, V<:AbstractVector{T}} <: AbstractVector{T} end
"""
    @weights name
Generates a new generic weight type with specified `name`, which subtypes `AbstractWeights`
and stores the `values` (`V<:RealVector`) and `sum` (`S<:Real`).
"""
macro weights(name)
    return quote
        struct $name{S<:Real, T<:Real, V<:AbstractVector{T}} <: AbstractWeights{S, T, V}
            values::V
            sum::S
        end
    end
end
eltype(wv::AbstractWeights) = eltype(wv.values)
length(wv::AbstractWeights) = length(wv.values)
values(wv::AbstractWeights) = wv.values
sum(wv::AbstractWeights) = wv.sum
isempty(wv::AbstractWeights) = isempty(wv.values)
Base.getindex(wv::AbstractWeights, i) = getindex(wv.values, i)
Base.size(wv::AbstractWeights) = size(wv.values)
"""
    varcorrection(n::Integer, corrected=false)
Compute a bias correction factor for calculating `var`, `std` and `cov` with
`n` observations. Returns ``\\frac{1}{n - 1}`` when `corrected=true`
(i.e. [Bessel's correction](https://en.wikipedia.org/wiki/Bessel's_correction)),
otherwise returns ``\\frac{1}{n}`` (i.e. no correction).
"""
@inline varcorrection(n::Integer, corrected::Bool=false) = 1 / (n - Int(corrected))
@weights Weights
"""
    Weights(vs, wsum=sum(vs))
Construct a `Weights` vector with weight values `vs`.
A precomputed sum may be provided as `wsum`.
The `Weights` type describes a generic weights vector which does not support
all operations possible for [`FrequencyWeights`](@ref), [`AnalyticWeights`](@ref)
and [`ProbabilityWeights`](@ref).
"""
Weights(vs::V, s::S=sum(vs)) where {S<:Real, V<:RealVector} = Weights{S, eltype(vs), V}(vs, s)
"""
    weights(vs)
Construct a `Weights` vector from array `vs`.
See the documentation for [`Weights`](@ref) for more details.
"""
weights(vs::RealVector) = Weights(vs)
weights(vs::RealArray) = Weights(vec(vs))
"""
    varcorrection(w::Weights, corrected=false)
Returns ``\\frac{1}{\\sum w}`` when `corrected=false` and throws an `ArgumentError`
if `corrected=true`.
"""
@inline function varcorrection(w::Weights, corrected::Bool=false)
    corrected && throw(ArgumentError("Weights type does not support bias correction: " *
                                     "use FrequencyWeights, AnalyticWeights or ProbabilityWeights if applicable."))
    1 / w.sum
end
@weights AnalyticWeights
"""
    AnalyticWeights(vs, wsum=sum(vs))
Construct an `AnalyticWeights` vector with weight values `vs`.
A precomputed sum may be provided as `wsum`.
Analytic weights describe a non-random relative importance (usually between 0 and 1)
for each observation. These weights may also be referred to as reliability weights,
precision weights or inverse variance weights. These are typically used when the observations
being weighted are aggregate values (e.g., averages) with differing variances.
"""
AnalyticWeights(vs::V, s::S=sum(vs)) where {S<:Real, V<:RealVector} =
    AnalyticWeights{S, eltype(vs), V}(vs, s)
"""
    aweights(vs)
Construct an `AnalyticWeights` vector from array `vs`.
See the documentation for [`AnalyticWeights`](@ref) for more details.
"""
aweights(vs::RealVector) = AnalyticWeights(vs)
aweights(vs::RealArray) = AnalyticWeights(vec(vs))
"""
    varcorrection(w::AnalyticWeights, corrected=false)
* `corrected=true`: ``\\frac{1}{\\sum w - \\sum {w^2} / \\sum w}``
* `corrected=false`: ``\\frac{1}{\\sum w}``
"""
@inline function varcorrection(w::AnalyticWeights, corrected::Bool=false)
    s = w.sum
    if corrected
        sum_sn = sum(x -> (x / s) ^ 2, w)
        1 / (s * (1 - sum_sn))
    else
        1 / s
    end
end
@weights FrequencyWeights
"""
    FrequencyWeights(vs, wsum=sum(vs))
Construct a `FrequencyWeights` vector with weight values `vs`.
A precomputed sum may be provided as `wsum`.
Frequency weights describe the number of times (or frequency) each observation
was observed. These weights may also be referred to as case weights or repeat weights.
"""
FrequencyWeights(vs::V, s::S=sum(vs)) where {S<:Real, V<:RealVector} =
    FrequencyWeights{S, eltype(vs), V}(vs, s)
"""
    fweights(vs)
Construct a `FrequencyWeights` vector from a given array.
See the documentation for [`FrequencyWeights`](@ref) for more details.
"""
fweights(vs::RealVector) = FrequencyWeights(vs)
fweights(vs::RealArray) = FrequencyWeights(vec(vs))
"""
    varcorrection(w::FrequencyWeights, corrected=false)
* `corrected=true`: ``\\frac{1}{\\sum{w} - 1}``
* `corrected=false`: ``\\frac{1}{\\sum w}``
"""
@inline function varcorrection(w::FrequencyWeights, corrected::Bool=false)
    s = w.sum
    if corrected
        1 / (s - 1)
    else
        1 / s
    end
end
@weights ProbabilityWeights
"""
    ProbabilityWeights(vs, wsum=sum(vs))
Construct a `ProbabilityWeights` vector with weight values `vs`.
A precomputed sum may be provided as `wsum`.
Probability weights represent the inverse of the sampling probability for each observation,
providing a correction mechanism for under- or over-sampling certain population groups.
These weights may also be referred to as sampling weights.
"""
ProbabilityWeights(vs::V, s::S=sum(vs)) where {S<:Real, V<:RealVector} =
    ProbabilityWeights{S, eltype(vs), V}(vs, s)
"""
    pweights(vs)
Construct a `ProbabilityWeights` vector from a given array.
See the documentation for [`ProbabilityWeights`](@ref) for more details.
"""
pweights(vs::RealVector) = ProbabilityWeights(vs)
pweights(vs::RealArray) = ProbabilityWeights(vec(vs))
"""
    varcorrection(w::ProbabilityWeights, corrected=false)
* `corrected=true`: ``\\frac{n}{(n - 1) \\sum w}`` where ``n`` equals `count(!iszero, w)`
* `corrected=false`: ``\\frac{1}{\\sum w}``
"""
@inline function varcorrection(w::ProbabilityWeights, corrected::Bool=false)
    s = w.sum
    if corrected
        n = count(!iszero, w)
        n / (s * (n - 1))
    else
        1 / s
    end
end
for w in (AnalyticWeights, FrequencyWeights, ProbabilityWeights, Weights)
    @eval begin
        Base.isequal(x::$w, y::$w) = isequal(x.sum, y.sum) && isequal(x.values, y.values)
        Base.:(==)(x::$w, y::$w)   = (x.sum == y.sum) && (x.values == y.values)
    end
end
Base.isequal(x::AbstractWeights, y::AbstractWeights) = false
Base.:(==)(x::AbstractWeights, y::AbstractWeights)   = false
"""
    wsum(v, w::AbstractVector, [dim])
Compute the weighted sum of an array `v` with weights `w`, optionally over the dimension `dim`.
"""
wsum(v::AbstractVector, w::AbstractVector) = dot(v, w)
wsum(v::AbstractArray, w::AbstractVector) = dot(vec(v), w)
Base.sum(v::BitArray, w::AbstractWeights) = wsum(v, values(w))
Base.sum(v::SparseMatrixCSC, w::AbstractWeights) = wsum(v, values(w))
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
function _wsum2_blas!(R::StridedVector{T}, A::StridedMatrix{T}, w::StridedVector{T}, dim::Int, init::Bool) where T<:BlasReal
    beta = ifelse(init, zero(T), one(T))
    trans = dim == 1 ? 'T' : 'N'
    BLAS.gemv!(trans, one(T), A, w, beta, R)
    return R
end
function _wsumN!(R::StridedArray{T}, A::StridedArray{T,N}, w::StridedVector{T}, dim::Int, init::Bool) where {T<:BlasReal,N}
    if dim == 1
        m = size(A, 1)
        n = div(length(A), m)
        _wsum2_blas!(view(R,:), reshape(A, (m, n)), w, 1, init)
    elseif dim == N
        n = size(A, N)
        m = div(length(A), n)
        _wsum2_blas!(view(R,:), reshape(A, (m, n)), w, 2, init)
    else # 1 < dim < N
        m = 1
        for i = 1:dim-1; m *= size(A, i); end
        n = size(A, dim)
        k = 1
        for i = dim+1:N; k *= size(A, i); end
        Av = reshape(A, (m, n, k))
        Rv = reshape(R, (m, k))
        for i = 1:k
            _wsum2_blas!(view(Rv,:,i), view(Av,:,:,i), w, 2, init)
        end
    end
    return R
end
function _wsumN!(R::StridedArray{T}, A::DenseArray{T,N}, w::StridedVector{T}, dim::Int, init::Bool) where {T<:BlasReal,N}
    @assert N >= 3
    if dim <= 2
        m = size(A, 1)
        n = size(A, 2)
        npages = 1
        for i = 3:N
            npages *= size(A, i)
        end
        rlen = ifelse(dim == 1, n, m)
        Rv = reshape(R, (rlen, npages))
        for i = 1:npages
            _wsum2_blas!(view(Rv,:,i), view(A,:,:,i), w, dim, init)
        end
    else
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
                for i_1 = 1:sizA1
                    @inbounds r += f(@nref $N A i) * w[i_1]
                end
                @inbounds (@nref $N R j) = r
            end
        else
            @nloops $N i A d->(if d == dim
                                   wi = w[i_d]
                                   j_d = 1
                               else
                                   j_d = i_d
                               end) @inbounds (@nref $N R j) += f(@nref $N A i) * wi
        end
        return R
    end
end
@generated function _wsum_centralize!(R::AbstractArray{RT}, f::supertype(typeof(abs)),
                                      A::AbstractArray{T,N}, w::AbstractVector{WT}, means,
                                      dim::Int, init::Bool) where {T,RT,WT,N}
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
"""
    wsum!(R, A, w, dim; init=true)
Compute the weighted sum of `A` with weights `w` over the dimension `dim` and store
the result in `R`. If `init=false`, the sum is added to `R` rather than starting
from zero.
"""
function wsum!(R::AbstractArray, A::AbstractArray{T,N}, w::AbstractVector, dim::Int; init::Bool=true) where {T,N}
    1 <= dim <= N || error("dim should be within [1, $N]")
    ndims(R) <= N || error("ndims(R) should not exceed $N")
    length(w) == size(A,dim) || throw(DimensionMismatch("Inconsistent array dimension."))
    # TODO: more careful examination of R's size
    _wsum!(R, A, w, dim, init)
end
function wsum(A::AbstractArray{T}, w::AbstractVector{W}, dim::Int) where {T<:Number,W<:Real}
    length(w) == size(A,dim) || throw(DimensionMismatch("Inconsistent array dimension."))
    _wsum!(similar(A, wsumtype(T,W), Base.reduced_indices(axes(A), dim)), A, w, dim, true)
end
Base.sum!(R::AbstractArray, A::AbstractArray, w::AbstractWeights{<:Real}, dim::Int; init::Bool=true) =
    wsum!(R, A, values(w), dim; init=init)
Base.sum(A::AbstractArray{<:Number}, w::AbstractWeights{<:Real}, dim::Int) = wsum(A, values(w), dim)
"""
    wmean(v, w::AbstractVector)
Compute the weighted mean of an array `v` with weights `w`.
"""
function wmean(v::AbstractArray{<:Number}, w::AbstractVector)
    Base.depwarn("wmean is deprecated, use mean(v, weights(w)) instead.", :wmean)
    mean(v, weights(w))
end
"""
    mean(A::AbstractArray, w::AbstractWeights[, dim::Int])
Compute the weighted mean of array `A` with weight vector `w`
(of type `AbstractWeights`). If `dim` is provided, compute the
weighted mean along dimension `dim`.
```julia
w = rand(n)
mean(x, weights(w))
```
"""
Base.mean(A::AbstractArray, w::AbstractWeights) = sum(A, w) / sum(w)
"""
    mean(R::AbstractArray, , A::AbstractArray, w::AbstractWeights[, dim::Int])
Compute the weighted mean of array `A` with weight vector `w`
(of type `AbstractWeights`) along dimension `dim`, and write results to `R`.
"""
Base.mean!(R::AbstractArray, A::AbstractArray, w::AbstractWeights, dim::Int) =
    scale!(Base.sum!(R, A, w, dim), inv(sum(w)))
wmeantype(::Type{T}, ::Type{W}) where {T,W} = typeof((zero(T)*zero(W) + zero(T)*zero(W)) / one(W))
wmeantype(::Type{T}, ::Type{T}) where {T<:BlasReal} = T
Base.mean(A::AbstractArray{T}, w::AbstractWeights{W}, dim::Int) where {T<:Number,W<:Real} =
    mean!(similar(A, wmeantype(T, W), Base.reduced_indices(axes(A), dim)), A, w, dim)
function Base.median(v::AbstractArray, w::AbstractWeights)
    throw(MethodError(median, (v, w)))
end
"""
    median(v::RealVector, w::AbstractWeights)
Compute the weighted median of `x`, using weights given by a weight vector `w`
(of type `AbstractWeights`). The weight and data vectors must have the same length.
The weighted median ``x_k`` is the element of `x` that satisfies
``\\sum_{x_i < x_k} w_i \\le \\frac{1}{2} \\sum_{j} w_j`` and
``\\sum_{x_i > x_k} w_i \\le \\frac{1}{2} \\sum_{j} w_j``.
If a weight has value zero, then its associated data point is ignored.
If none of the weights are positive, an error is thrown.
`NaN` is returned if `x` contains any `NaN` values.
An error is raised if `w` contains any `NaN` values.
"""
function Base.median(v::RealVector, w::AbstractWeights{<:Real})
    isempty(v) && error("median of an empty array is undefined")
    if length(v) != length(w)
        error("data and weight vectors must be the same size")
    end
    @inbounds for x in w.values
        isnan(x) && error("weight vector cannot contain NaN entries")
    end
    @inbounds for x in v
        isnan(x) && return x
    end
    mask = w.values .!= 0
    if !any(mask)
        error("all weights are zero")
    end
    if all(w.values .<= 0)
        error("no positive weights found")
    end
    v = v[mask]
    wt = w[mask]
    midpoint = w.sum / 2
    maxval, maxind = findmax(wt)
    if maxval > midpoint
        v[maxind]
    else
        permute = sortperm(v)
        cumulative_weight = zero(eltype(wt))
        i = 0
        for (_i, p) in enumerate(permute)
            i = _i
            if cumulative_weight == midpoint
                i += 1
                break
            elseif cumulative_weight > midpoint
                cumulative_weight -= wt[p]
                break
            end
            cumulative_weight += wt[p]
        end
        if cumulative_weight == midpoint
            middle(v[permute[i-2]], v[permute[i-1]])
        else
            middle(v[permute[i-1]])
        end
    end
end
"""
    wmedian(v, w)
Compute the weighted median of an array `v` with weights `w`, given as either a
vector or an `AbstractWeights` vector.
"""
wmedian(v::RealVector, w::RealVector) = median(v, weights(w))
wmedian(v::RealVector, w::AbstractWeights{<:Real}) = median(v, w)
"""
    quantile(v, w::AbstractWeights, p)
Compute the weighted quantiles of a vector `x` at a specified set of probability
values `p`, using weights given by a weight vector `w` (of type `AbstractWeights`).
Weights must not be negative. The weights and data vectors must have the same length.
The quantile for `p` is defined as follows. Denoting
``S_k = (k-1)w_k + (n-1) \\sum_{i<k}w_i``, define ``x_{k+1}`` the smallest element of `x`
such that ``S_{k+1}/S_{n}`` is strictly superior to `p`. The function returns
``(1-\\gamma) x_k + \\gamma x_{k+1}`` with  ``\\gamma = (pS_n- S_k)/(S_{k+1}-S_k)``.
This corresponds to  R-7, Excel, SciPy-(1,1) and Maple-6 when `w` contains only ones
(see [Wikipedia](https://en.wikipedia.org/wiki/Quantile)).
"""
function quantile(v::RealVector{V}, w::AbstractWeights{W}, p::RealVector) where {V,W<:Real}
    # checks
    isempty(v) && error("quantile of an empty array is undefined")
    isempty(p) && throw(ArgumentError("empty quantile array"))
    w.sum == 0 && error("weight vector cannot sum to zero")
    length(v) == length(w) || error("data and weight vectors must be the same size, got $(length(v)) and $(length(w))")
    for x in w.values
        isnan(x) && error("weight vector cannot contain NaN entries")
        x < 0 && error("weight vector cannot contain negative entries")
    end
    # full sort
    vw = sort!(collect(zip(v, w.values)))
    wsum = w.sum
    # prepare percentiles
    ppermute = sortperm(p)
    p = p[ppermute]
    p = bound_quantiles(p)
    # prepare out vector
    N = length(vw)
    out = Vector{typeof(zero(V)/1)}(uninitialized, length(p))
    fill!(out, vw[end][1])
    # start looping on quantiles
    cumulative_weight, Sk, Skold =  zero(W), zero(W), zero(W)
    vk, vkold = zero(V), zero(V)
    k = 1
    for i in 1:length(p)
        h = p[i] * (N - 1) * wsum
        if h == 0
            # happens when N or p or wsum equal zero
            out[ppermute[i]] = vw[1][1]
        else
            while Sk <= h
                # happens in particular when k == 1
                vk, wk = vw[k]
                cumulative_weight += wk
                if k >= N
                    # out was initialized with maximum v
                    return out
                end
                k += 1
                Skold, vkold = Sk, vk
                vk, wk = vw[k]
                Sk = (k - 1) * wk + (N - 1) * cumulative_weight
            end
            # in particular, Sk is different from Skold
            g = (h - Skold) / (Sk - Skold)
            out[ppermute[i]] = vkold + g * (vk - vkold)
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
"""
    wquantile(v, w, p)
Compute the `p`th quantile(s) of `v` with weights `w`, given as either a vector
or an `AbstractWeights` vector.
"""
wquantile(v::RealVector, w::AbstractWeights{<:Real}, p::RealVector) = quantile(v, w, p)
wquantile(v::RealVector, w::AbstractWeights{<:Real}, p::Number) = quantile(v, w, [p])[1]
wquantile(v::RealVector, w::RealVector, p::RealVector) = quantile(v, weights(w), p)
wquantile(v::RealVector, w::RealVector, p::Number) = quantile(v, weights(w), [p])[1]
"""
    varm(x, w::AbstractWeights, m, [dim]; corrected=false)
Compute the variance of a real-valued array `x` with a known mean `m`, optionally
over a dimension `dim`. Observations in `x` are weighted using weight vector `w`.
The uncorrected (when `corrected=false`) sample variance is defined as:
```math
\\frac{1}{\\sum{w}} \\sum_{i=1}^n {w_i\\left({x_i - m}\\right)^2 }
```
where ``n`` is the length of the input. The unbiased estimate (when `corrected=true`) of
the population variance is computed by replacing
``\\frac{1}{\\sum{w}}`` with a factor dependent on the type of weights used:
* `AnalyticWeights`: ``\\frac{1}{\\sum w - \\sum {w^2} / \\sum w}``
* `FrequencyWeights`: ``\\frac{1}{\\sum{w} - 1}``
* `ProbabilityWeights`: ``\\frac{n}{(n - 1) \\sum w}`` where ``n`` equals `count(!iszero, w)`
* `Weights`: `ArgumentError` (bias correction not supported)
"""
Base.varm(v::RealArray, w::AbstractWeights, m::Real; corrected::DepBool=nothing) =
    _moment2(v, w, m; corrected=depcheck(:varm, corrected))
"""
    var(x, w::AbstractWeights, [dim]; mean=nothing, corrected=false)
Compute the variance of a real-valued array `x`, optionally over a dimension `dim`.
Observations in `x` are weighted using weight vector `w`.
The uncorrected (when `corrected=false`) sample variance is defined as:
```math
\\frac{1}{\\sum{w}} \\sum_{i=1}^n {w_i\\left({x_i - μ}\\right)^2 }
```
where ``n`` is the length of the input and ``μ`` is the mean.
The unbiased estimate (when `corrected=true`) of the population variance is computed by
replacing ``\\frac{1}{\\sum{w}}`` with a factor dependent on the type of weights used:
* `AnalyticWeights`: ``\\frac{1}{\\sum w - \\sum {w^2} / \\sum w}``
* `FrequencyWeights`: ``\\frac{1}{\\sum{w} - 1}``
* `ProbabilityWeights`: ``\\frac{n}{(n - 1) \\sum w}`` where ``n`` equals `count(!iszero, w)`
* `Weights`: `ArgumentError` (bias correction not supported)
"""
function Base.var(v::RealArray, w::AbstractWeights; mean=nothing,
                  corrected::DepBool=nothing)
    corrected = depcheck(:var, corrected)
    if mean == nothing
        varm(v, w, Base.mean(v, w); corrected=corrected)
    else
        varm(v, w, mean; corrected=corrected)
    end
end
function Base.varm!(R::AbstractArray, A::RealArray, w::AbstractWeights, M::RealArray,
                    dim::Int; corrected::DepBool=nothing)
    corrected = depcheck(:varm!, corrected)
    scale!(_wsum_centralize!(R, abs2, A, values(w), M, dim, true),
           varcorrection(w, corrected))
end
function var!(R::AbstractArray, A::RealArray, w::AbstractWeights, dim::Int;
              mean=nothing, corrected::DepBool=nothing)
    corrected = depcheck(:var!, corrected)
    if mean == 0
        Base.varm!(R, A, w, Base.reducedim_initarray(A, dim, 0, eltype(R)), dim;
                   corrected=corrected)
    elseif mean == nothing
        Base.varm!(R, A, w, Base.mean(A, w, dim), dim; corrected=corrected)
    else
        # check size of mean
        for i = 1:ndims(A)
            dA = size(A,i)
            dM = size(mean,i)
            if i == dim
                dM == 1 || throw(DimensionMismatch("Incorrect size of mean."))
            else
                dM == dA || throw(DimensionMismatch("Incorrect size of mean."))
            end
        end
        Base.varm!(R, A, w, mean, dim; corrected=corrected)
    end
end
function Base.varm(A::RealArray, w::AbstractWeights, M::RealArray, dim::Int;
                   corrected::DepBool=nothing)
    corrected = depcheck(:varm, corrected)
    Base.varm!(similar(A, Float64, Base.reduced_indices(indices(A), dim)), A, w, M,
               dim; corrected=corrected)
end
function Base.var(A::RealArray, w::AbstractWeights, dim::Int; mean=nothing,
                  corrected::DepBool=nothing)
    corrected = depcheck(:var, corrected)
    var!(similar(A, Float64, Base.reduced_indices(indices(A), dim)), A, w, dim;
         mean=mean, corrected=corrected)
end
"""
    stdm(v, w::AbstractWeights, m, [dim]; corrected=false)
Compute the standard deviation of a real-valued array `x` with a known mean `m`,
optionally over a dimension `dim`. Observations in `x` are weighted using weight vector `w`.
The uncorrected (when `corrected=false`) sample standard deviation is defined as:
```math
\\sqrt{\\frac{1}{\\sum{w}} \\sum_{i=1}^n {w_i\\left({x_i - m}\\right)^2 }}
```
where ``n`` is the length of the input. The unbiased estimate (when `corrected=true`) of the
population standard deviation is computed by replacing ``\\frac{1}{\\sum{w}}`` with a factor
dependent on the type of weights used:
* `AnalyticWeights`: ``\\frac{1}{\\sum w - \\sum {w^2} / \\sum w}``
* `FrequencyWeights`: ``\\frac{1}{\\sum{w} - 1}``
* `ProbabilityWeights`: ``\\frac{n}{(n - 1) \\sum w}`` where ``n`` equals `count(!iszero, w)`
* `Weights`: `ArgumentError` (bias correction not supported)
"""
Base.stdm(v::RealArray, w::AbstractWeights, m::Real; corrected::DepBool=nothing) =
    sqrt(varm(v, w, m, corrected=depcheck(:stdm, corrected)))
"""
    std(v, w::AbstractWeights, [dim]; mean=nothing, corrected=false)
Compute the standard deviation of a real-valued array `x`,
optionally over a dimension `dim`. Observations in `x` are weighted using weight vector `w`.
The uncorrected (when `corrected=false`) sample standard deviation is defined as:
```math
\\sqrt{\\frac{1}{\\sum{w}} \\sum_{i=1}^n {w_i\\left({x_i - μ}\\right)^2 }}
```
where ``n`` is the length of the input and ``μ`` is the mean.
The unbiased estimate (when `corrected=true`) of the population standard deviation is
computed by replacing ``\\frac{1}{\\sum{w}}`` with a factor dependent on the type of
weights used:
* `AnalyticWeights`: ``\\frac{1}{\\sum w - \\sum {w^2} / \\sum w}``
* `FrequencyWeights`: ``\\frac{1}{\\sum{w} - 1}``
* `ProbabilityWeights`: ``\\frac{n}{(n - 1) \\sum w}`` where ``n`` equals `count(!iszero, w)`
* `Weights`: `ArgumentError` (bias correction not supported)
"""
Base.std(v::RealArray, w::AbstractWeights; mean=nothing, corrected::DepBool=nothing) =
    sqrt.(var(v, w; mean=mean, corrected=depcheck(:std, corrected)))
Base.stdm(v::RealArray, m::RealArray, dim::Int; corrected::DepBool=nothing) =
    Base.sqrt!(varm(v, m, dim; corrected=depcheck(:stdm, corrected)))
Base.stdm(v::RealArray, w::AbstractWeights, m::RealArray, dim::Int;
          corrected::DepBool=nothing) =
    sqrt.(varm(v, w, m, dim; corrected=depcheck(:stdm, corrected)))
Base.std(v::RealArray, w::AbstractWeights, dim::Int; mean=nothing,
         corrected::DepBool=nothing) =
    sqrt.(var(v, w, dim; mean=mean, corrected=depcheck(:std, corrected)))
"""
    mean_and_var(x, [w::AbstractWeights], [dim]; corrected=false) -> (mean, var)
Return the mean and variance of a real-valued array `x`, optionally over a dimension
`dim`, as a tuple. Observations in `x` can be weighted using weight vector `w`.
Finally, bias correction is be applied to the variance calculation if `corrected=true`.
See [`var`](@ref) documentation for more details.
"""
function mean_and_var(A::RealArray; corrected::Bool=true)
    m = mean(A)
    v = varm(A, m; corrected=corrected)
    m, v
end
"""
    mean_and_std(x, [w::AbstractWeights], [dim]; corrected=false) -> (mean, std)
Return the mean and standard deviation of a real-valued array `x`, optionally
over a dimension `dim`, as a tuple. A weighting vector `w` can be specified
to weight the estimates. Finally, bias correction is applied to the
standard deviation calculation if `corrected=true`.
See [`std`](@ref) documentation for more details.
"""
function mean_and_std(A::RealArray; corrected::Bool=true)
    m = mean(A)
    s = stdm(A, m; corrected=corrected)
    m, s
end
function mean_and_var(A::RealArray, w::AbstractWeights; corrected::DepBool=nothing)
    m = mean(A, w)
    v = varm(A, w, m; corrected=depcheck(:mean_and_var, corrected))
    m, v
end
function mean_and_std(A::RealArray, w::AbstractWeights; corrected::DepBool=nothing)
    m = mean(A, w)
    s = stdm(A, w, m; corrected=depcheck(:mean_and_std, corrected))
    m, s
end
function mean_and_var(A::RealArray, dim::Int; corrected::Bool=true)
    m = mean(A, dim)
    v = varm(A, m, dim; corrected=corrected)
    m, v
end
function mean_and_std(A::RealArray, dim::Int; corrected::Bool=true)
    m = mean(A, dim)
    s = stdm(A, m, dim; corrected=corrected)
    m, s
end
function mean_and_var(A::RealArray, w::AbstractWeights, dim::Int;
                      corrected::DepBool=nothing)
    m = mean(A, w, dim)
    v = varm(A, w, m, dim; corrected=depcheck(:mean_and_var, corrected))
    m, v
end
function mean_and_std(A::RealArray, w::AbstractWeights, dim::Int;
                      corrected::DepBool=nothing)
    m = mean(A, w, dim)
    s = stdm(A, w, m, dim; corrected=depcheck(:mean_and_std, corrected))
    m, s
end
function _moment2(v::RealArray, m::Real; corrected=false)
    n = length(v)
    s = 0.0
    for i = 1:n
        @inbounds z = v[i] - m
        s += z * z
    end
    varcorrection(n, corrected) * s
end
function _moment2(v::RealArray, wv::AbstractWeights, m::Real; corrected=false)
    n = length(v)
    s = 0.0
    w = values(wv)
    for i = 1:n
        @inbounds z = v[i] - m
        @inbounds s += (z * z) * w[i]
    end
    varcorrection(wv, corrected) * s
end
function _moment3(v::RealArray, m::Real)
    n = length(v)
    s = 0.0
    for i = 1:n
        @inbounds z = v[i] - m
        s += z * z * z
    end
    s / n
end
function _moment3(v::RealArray, wv::AbstractWeights, m::Real)
    n = length(v)
    s = 0.0
    w = values(wv)
    for i = 1:n
        @inbounds z = v[i] - m
        @inbounds s += (z * z * z) * w[i]
    end
    s / sum(wv)
end
function _moment4(v::RealArray, m::Real)
    n = length(v)
    s = 0.0
    for i = 1:n
        @inbounds z = v[i] - m
        s += abs2(z * z)
    end
    s / n
end
function _moment4(v::RealArray, wv::AbstractWeights, m::Real)
    n = length(v)
    s = 0.0
    w = values(wv)
    for i = 1:n
        @inbounds z = v[i] - m
        @inbounds s += abs2(z * z) * w[i]
    end
    s / sum(wv)
end
function _momentk(v::RealArray, k::Int, m::Real)
    n = length(v)
    s = 0.0
    for i = 1:n
        @inbounds z = v[i] - m
        s += (z ^ k)
    end
    s / n
end
function _momentk(v::RealArray, k::Int, wv::AbstractWeights, m::Real)
    n = length(v)
    s = 0.0
    w = values(wv)
    for i = 1:n
        @inbounds z = v[i] - m
        @inbounds s += (z ^ k) * w[i]
    end
    s / sum(wv)
end
"""
    moment(v, k, [wv::AbstractWeights], m=mean(v))
Return the `k`th order central moment of a real-valued array `v`, optionally
specifying a weighting vector `wv` and a center `m`.
"""
function moment(v::RealArray, k::Int, m::Real)
    k == 2 ? _moment2(v, m) :
    k == 3 ? _moment3(v, m) :
    k == 4 ? _moment4(v, m) :
    _momentk(v, k, m)
end
function moment(v::RealArray, k::Int, wv::AbstractWeights, m::Real)
    k == 2 ? _moment2(v, wv, m) :
    k == 3 ? _moment3(v, wv, m) :
    k == 4 ? _moment4(v, wv, m) :
    _momentk(v, k, wv, m)
end
moment(v::RealArray, k::Int) = moment(v, k, mean(v))
function moment(v::RealArray, k::Int, wv::AbstractWeights)
    moment(v, k, wv, mean(v, wv))
end
"""
    skewness(v, [wv::AbstractWeights], m=mean(v))
Compute the standardized skewness of a real-valued array `v`, optionally
specifying a weighting vector `wv` and a center `m`.
"""
function skewness(v::RealArray, m::Real)
    n = length(v)
    cm2 = 0.0   # empirical 2nd centered moment (variance)
    cm3 = 0.0   # empirical 3rd centered moment
    for i = 1:n
        @inbounds z = v[i] - m
        z2 = z * z
        cm2 += z2
        cm3 += z2 * z
    end
    cm3 /= n
    cm2 /= n
    return cm3 / sqrt(cm2 * cm2 * cm2)  # this is much faster than cm2^1.5
end
function skewness(v::RealArray, wv::AbstractWeights, m::Real)
    n = length(v)
    length(wv) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    cm2 = 0.0   # empirical 2nd centered moment (variance)
    cm3 = 0.0   # empirical 3rd centered moment
    w = values(wv)
    @inbounds for i = 1:n
        x_i = v[i]
        w_i = w[i]
        z = x_i - m
        z2w = z * z * w_i
        cm2 += z2w
        cm3 += z2w * z
    end
    sw = sum(wv)
    cm3 /= sw
    cm2 /= sw
    return cm3 / sqrt(cm2 * cm2 * cm2)  # this is much faster than cm2^1.5
end
skewness(v::RealArray) = skewness(v, mean(v))
skewness(v::RealArray, wv::AbstractWeights) = skewness(v, wv, mean(v, wv))
"""
    kurtosis(v, [wv::AbstractWeights], m=mean(v))
Compute the excess kurtosis of a real-valued array `v`, optionally
specifying a weighting vector `wv` and a center `m`.
"""
function kurtosis(v::RealArray, m::Real)
    n = length(v)
    cm2 = 0.0  # empirical 2nd centered moment (variance)
    cm4 = 0.0  # empirical 4th centered moment
    for i = 1:n
        @inbounds z = v[i] - m
        z2 = z * z
        cm2 += z2
        cm4 += z2 * z2
    end
    cm4 /= n
    cm2 /= n
    return (cm4 / (cm2 * cm2)) - 3.0
end
function kurtosis(v::RealArray, wv::AbstractWeights, m::Real)
    n = length(v)
    length(wv) == n || throw(DimensionMismatch("Inconsistent array lengths."))
    cm2 = 0.0  # empirical 2nd centered moment (variance)
    cm4 = 0.0  # empirical 4th centered moment
    w = values(wv)
    @inbounds for i = 1 : n
        x_i = v[i]
        w_i = w[i]
        z = x_i - m
        z2 = z * z
        z2w = z2 * w_i
        cm2 += z2w
        cm4 += z2w * z2
    end
    sw = sum(wv)
    cm4 /= sw
    cm2 /= sw
    return (cm4 / (cm2 * cm2)) - 3.0
end
kurtosis(v::RealArray) = kurtosis(v, mean(v))
kurtosis(v::RealArray, wv::AbstractWeights) = kurtosis(v, wv, mean(v, wv))
"""
    geomean(a)
Return the geometric mean of a real-valued array.
"""
function geomean(a::RealArray)
    s = 0.0
    n = length(a)
    for i = 1 : n
        @inbounds s += log(a[i])
    end
    return exp(s / n)
end
"""
    harmmean(a)
Return the harmonic mean of a real-valued array.
"""
function harmmean(a::RealArray)
    s = 0.0
    n = length(a)
    for i in 1 : n
        @inbounds s += inv(a[i])
    end
    return n / s
end
"""
    genmean(a, p)
Return the generalized/power mean with exponent `p` of a real-valued array,
i.e. ``\\left( \\frac{1}{n} \\sum_{i=1}^n a_i^p \\right)^{\\frac{1}{p}}``, where `n = length(a)`.
It is taken to be the geometric mean when `p == 0`.
"""
function genmean(a::RealArray, p::Real)
    if p == 0
        return geomean(a)
    end
    s = 0.0
    n = length(a)
    for x in a
        #= At least one of `x` or `p` must not be an int to avoid domain errors when `p` is a negative int.
        We choose `x` in order to exploit exponentiation by squaring when `p` is an int. =#
        @inbounds s += convert(Float64, x)^p
    end
    return (s/n)^(1/p)
end
"""
    mode(a, [r])
Return the mode (most common number) of an array, optionally
over a specified range `r`. If several modes exist, the first
one (in order of appearance) is returned.
"""
function mode(a::AbstractArray{T}, r::UnitRange{T}) where T<:Integer
    isempty(a) && error("mode: input array cannot be empty.")
    len = length(a)
    r0 = r[1]
    r1 = r[end]
    cnts = zeros(Int, length(r))
    mc = 0    # maximum count
    mv = r0   # a value corresponding to maximum count
    for i = 1:len
        @inbounds x = a[i]
        if r0 <= x <= r1
            @inbounds c = (cnts[x - r0 + 1] += 1)
            if c > mc
                mc = c
                mv = x
            end
        end
    end
    return mv
end
"""
    modes(a, [r])::Vector
Return all modes (most common numbers) of an array, optionally over a
specified range `r`.
"""
function modes(a::AbstractArray{T}, r::UnitRange{T}) where T<:Integer
    r0 = r[1]
    r1 = r[end]
    n = length(r)
    cnts = zeros(Int, n)
    # find the maximum count
    mc = 0
    for i = 1:length(a)
        @inbounds x = a[i]
        if r0 <= x <= r1
            @inbounds c = (cnts[x - r0 + 1] += 1)
            if c > mc
                mc = c
            end
        end
    end
    # find all values corresponding to maximum count
    ms = T[]
    for i = 1:n
        @inbounds if cnts[i] == mc
            push!(ms, r[i])
        end
    end
    return ms
end
function mode(a::AbstractArray{T}) where T
    isempty(a) && error("mode: input array cannot be empty.")
    cnts = Dict{T,Int}()
    # first element
    mc = 1
    mv = a[1]
    cnts[mv] = 1
    # find the mode along with table construction
    for i = 2 : length(a)
        @inbounds x = a[i]
        if haskey(cnts, x)
            c = (cnts[x] += 1)
            if c > mc
                mc = c
                mv = x
            end
        else
            cnts[x] = 1
            # in this case: c = 1, and thus c > mc won't happen
        end
    end
    return mv
end
function modes(a::AbstractArray{T}) where T
    isempty(a) && error("modes: input array cannot be empty.")
    cnts = Dict{T,Int}()
    # first element
    mc = 1
    cnts[a[1]] = 1
    # find the mode along with table construction
    for i = 2 : length(a)
        @inbounds x = a[i]
        if haskey(cnts, x)
            c = (cnts[x] += 1)
            if c > mc
                mc = c
            end
        else
            cnts[x] = 1
            # in this case: c = 1, and thus c > mc won't happen
        end
    end
    # find values corresponding to maximum counts
    ms = T[]
    for (x, c) in cnts
        if c == mc
            push!(ms, x)
        end
    end
    return ms
end
"""
    percentile(v, p)
Return the `p`th percentile of a real-valued array `v`, i.e. `quantile(x, p / 100)`.
"""
percentile(v::AbstractArray{<:Real}, p) = quantile(v, p * 0.01)
quantile(v::AbstractArray{<:Real}) = quantile(v, [.0, .25, .5, .75, 1.0])
"""
    nquantile(v, n)
Return the n-quantiles of a real-valued array, i.e. the values which
partition `v` into `n` subsets of nearly equal size.
Equivalent to `quantile(v, [0:n]/n)`. For example, `nquantiles(x, 5)`
returns a vector of quantiles, respectively at `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`.
"""
nquantile(v::AbstractArray{<:Real}, n::Integer) = quantile(v, (0:n)/n)
"""
    span(x)
Return the span of an integer array, i.e. the range `minimum(x):maximum(x)`.
The minimum and maximum of `x` are computed in one-pass using `extrema`.
"""
span(x::AbstractArray{<:Integer}) = ((a, b) = extrema(x); a:b)
"""
    variation(x, m=mean(x))
Return the coefficient of variation of an array `x`, optionally specifying
a precomputed mean `m`. The coefficient of variation is the ratio of the
standard deviation to the mean.
"""
variation(x::AbstractArray{<:Real}, m::Real) = stdm(x, m) / m
variation(x::AbstractArray{<:Real}) = variation(x, mean(x))
"""
    sem(a)
Return the standard error of the mean of `a`, i.e. `sqrt(var(a) / length(a))`.
"""
sem(a::AbstractArray{<:Real}) = sqrt(var(a) / length(a))
"""
    mad(v; center=median(v), normalize=true)
Compute the median absolute deviation (MAD) of `v` around `center`
(by default, around the median).
If `normalize` is set to `true`, the MAD is multiplied by
`1 / quantile(Normal(), 3/4) ≈ 1.4826`, in order to obtain a consistent estimator
of the standard deviation under the assumption that the data is normally distributed.
"""
function mad(v::AbstractArray{T};
             center::Union{Real,Nothing}=nothing,
             normalize::Union{Bool, Nothing}=nothing) where T<:Real
    isempty(v) && throw(ArgumentError("mad is not defined for empty arrays"))
    S = promote_type(T, typeof(middle(first(v))))
    v2 = LinAlg.copy_oftype(v, S)
    if normalize === nothing
        Base.depwarn("the `normalize` keyword argument will be false by default in future releases: set it explicitly to silence this deprecation", :mad)
        normalize = true
    end
    mad!(v2, center=center === nothing ? median!(v2) : center, normalize=normalize)
end
"""
    StatsBase.mad!(v; center=median!(v), normalize=true)
Compute the median absolute deviation (MAD) of `v` around `center`
(by default, around the median), overwriting `v` in the process.
If `normalize` is set to `true`, the MAD is multiplied by
`1 / quantile(Normal(), 3/4) ≈ 1.4826`, in order to obtain a consistent estimator
of the standard deviation under the assumption that the data is normally distributed.
"""
function mad!(v::AbstractArray{T};
              center::Real=median!(v),
              normalize::Union{Bool,Nothing}=true,
              constant=nothing) where T<:Real
    for i in 1:length(v)
        @inbounds v[i] = abs(v[i]-center)
    end
    k = 1 / (-sqrt(2 * one(T)) * erfcinv(3 * one(T) / 2))
    if normalize === nothing
        Base.depwarn("the `normalize` keyword argument will be false by default in future releases: set it explicitly to silence this deprecation", :mad)
        normalize = true
    end
    if constant !== nothing
        Base.depwarn("keyword argument `constant` is deprecated, use `normalize` instead or apply the multiplication directly", :mad)
        constant * median!(v)
    elseif normalize
        k * median!(v)
    else
        one(k) * median!(v)
    end
end
"""
    iqr(v)
Compute the interquartile range (IQR) of an array, i.e. the 75th percentile
minus the 25th percentile.
"""
iqr(v::AbstractArray{<:Real}) = (q = quantile(v, [.25, .75]); q[2] - q[1])
function _zscore!(Z::AbstractArray, X::AbstractArray, μ::Real, σ::Real)
    # Z and X are assumed to have the same size
    iσ = inv(σ)
    if μ == zero(μ)
        for i = 1 : length(X)
            @inbounds Z[i] = X[i] * iσ
        end
    else
        for i = 1 : length(X)
            @inbounds Z[i] = (X[i] - μ) * iσ
        end
    end
    return Z
end
@generated function _zscore!(Z::AbstractArray{S,N}, X::AbstractArray{T,N},
                             μ::AbstractArray, σ::AbstractArray) where {S,T,N}
    quote
        # Z and X are assumed to have the same size
        # μ and σ are assumed to have the same size, that is compatible with size(X)
        siz1 = size(X, 1)
        @nextract $N ud d->size(μ, d)
        if size(μ, 1) == 1 && siz1 > 1
            @nloops $N i d->(d>1 ? (1:size(X,d)) : (1:1)) d->(j_d = ud_d ==1 ? 1 : i_d) begin
                v = (@nref $N μ j)
                c = inv(@nref $N σ j)
                for i_1 = 1:siz1
                    (@nref $N Z i) = ((@nref $N X i) - v) * c
                end
            end
        else
            @nloops $N i X d->(j_d = ud_d ==1 ? 1 : i_d) begin
                (@nref $N Z i) = ((@nref $N X i) - (@nref $N μ j)) / (@nref $N σ j)
            end
        end
        return Z
    end
end
function _zscore_chksize(X::AbstractArray, μ::AbstractArray, σ::AbstractArray)
    size(μ) == size(σ) || throw(DimensionMismatch("μ and σ should have the same size."))
    for i=1:ndims(X)
        dμ_i = size(μ,i)
        (dμ_i == 1 || dμ_i == size(X,i)) || throw(DimensionMismatch("X and μ have incompatible sizes."))
    end
end
"""
    zscore!([Z], X, μ, σ)
Compute the z-scores of an array `X` with mean `μ` and standard deviation `σ`.
z-scores are the signed number of standard deviations above the mean that an
observation lies, i.e. ``(x - μ) / σ``.
If a destination array `Z` is provided, the scores are stored
in `Z` and it must have the same shape as `X`. Otherwise `X` is overwritten.
"""
function zscore!(Z::AbstractArray{ZT}, X::AbstractArray{T}, μ::Real, σ::Real) where {ZT<:AbstractFloat,T<:Real}
    size(Z) == size(X) || throw(DimensionMismatch("Z and X must have the same size."))
    _zscore!(Z, X, μ, σ)
end
function zscore!(Z::AbstractArray{<:AbstractFloat}, X::AbstractArray{<:Real},
                 μ::AbstractArray{<:Real}, σ::AbstractArray{<:Real})
    size(Z) == size(X) || throw(DimensionMismatch("Z and X must have the same size."))
    _zscore_chksize(X, μ, σ)
    _zscore!(Z, X, μ, σ)
end
zscore!(X::AbstractArray{<:AbstractFloat}, μ::Real, σ::Real) = _zscore!(X, X, μ, σ)
zscore!(X::AbstractArray{<:AbstractFloat}, μ::AbstractArray{<:Real}, σ::AbstractArray{<:Real}) =
    (_zscore_chksize(X, μ, σ); _zscore!(X, X, μ, σ))
"""
    zscore(X, [μ, σ])
Compute the z-scores of `X`, optionally specifying a precomputed mean `μ` and
standard deviation `σ`. z-scores are the signed number of standard deviations
above the mean that an observation lies, i.e. ``(x - μ) / σ``.
`μ` and `σ` should be both scalars or both arrays. The computation is broadcasting.
In particular, when `μ` and `σ` are arrays, they should have the same size, and
`size(μ, i) == 1  || size(μ, i) == size(X, i)` for each dimension.
"""
function zscore(X::AbstractArray{T}, μ::Real, σ::Real) where T<:Real
    ZT = typeof((zero(T) - zero(μ)) / one(σ))
    _zscore!(Array{ZT}(uninitialized, size(X)), X, μ, σ)
end
function zscore(X::AbstractArray{T}, μ::AbstractArray{U}, σ::AbstractArray{S}) where {T<:Real,U<:Real,S<:Real}
    _zscore_chksize(X, μ, σ)
    ZT = typeof((zero(T) - zero(U)) / one(S))
    _zscore!(Array{ZT}(uninitialized, size(X)), X, μ, σ)
end
zscore(X::AbstractArray{<:Real}) = ((μ, σ) = mean_and_std(X); zscore(X, μ, σ))
zscore(X::AbstractArray{<:Real}, dim::Int) = ((μ, σ) = mean_and_std(X, dim); zscore(X, μ, σ))
"""
    entropy(p, [b])
Compute the entropy of an array `p`, optionally specifying a real number
`b` such that the entropy is scaled by `1/log(b)`.
"""
function entropy(p::AbstractArray{T}) where T<:Real
    s = zero(T)
    z = zero(T)
    for i = 1:length(p)
        @inbounds pi = p[i]
        if pi > z
            s += pi * log(pi)
        end
    end
    return -s
end
entropy(p::AbstractArray{<:Real}, b::Real) = entropy(p) / log(b)
"""
    renyientropy(p, α)
Compute the Rényi (generalized) entropy of order `α` of an array `p`.
"""
function renyientropy(p::AbstractArray{T}, α::Real) where T<:Real
    α < 0 && throw(ArgumentError("Order of Rényi entropy not legal, $(α) < 0."))
    s = zero(T)
    z = zero(T)
    scale = sum(p)
    if α ≈ 0
        for i = 1:length(p)
            @inbounds pi = p[i]
            if pi > z
                s += 1
            end
        end
        s = log(s / scale)
    elseif α ≈ 1
        for i = 1:length(p)
            @inbounds pi = p[i]
            if pi > z
                s -= pi * log(pi)
            end
        end
        s = s / scale
    elseif (isinf(α))
        s = -log(maximum(p))
    else # a normal Rényi entropy
        for i = 1:length(p)
            @inbounds pi = p[i]
            if pi > z
                s += pi ^ α
            end
        end
        s = log(s / scale) / (1 - α)
    end
    return s
end
"""
    crossentropy(p, q, [b])
Compute the cross entropy between `p` and `q`, optionally specifying a real
number `b` such that the result is scaled by `1/log(b)`.
"""
function crossentropy(p::AbstractArray{T}, q::AbstractArray{T}) where T<:Real
    length(p) == length(q) || throw(DimensionMismatch("Inconsistent array length."))
    s = 0.
    z = zero(T)
    for i = 1:length(p)
        @inbounds pi = p[i]
        @inbounds qi = q[i]
        if pi > z
            s += pi * log(qi)
        end
    end
    return -s
end
crossentropy(p::AbstractArray{T}, q::AbstractArray{T}, b::Real) where {T<:Real} =
    crossentropy(p,q) / log(b)
"""
    kldivergence(p, q, [b])
Compute the Kullback-Leibler divergence of `q` from `p`, optionally specifying
a real number `b` such that the divergence is scaled by `1/log(b)`.
"""
function kldivergence(p::AbstractArray{T}, q::AbstractArray{T}) where T<:Real
    length(p) == length(q) || throw(DimensionMismatch("Inconsistent array length."))
    s = 0.
    z = zero(T)
    for i = 1:length(p)
        @inbounds pi = p[i]
        @inbounds qi = q[i]
        if pi > z
            s += pi * log(pi / qi)
        end
    end
    return s
end
kldivergence(p::AbstractArray{T}, q::AbstractArray{T}, b::Real) where {T<:Real} =
    kldivergence(p,q) / log(b)
struct SummaryStats{T<:AbstractFloat}
    mean::T
    min::T
    q25::T
    median::T
    q75::T
    max::T
end
"""
    summarystats(a)
Compute summary statistics for a real-valued array `a`. Returns a
`SummaryStats` object containing the mean, minimum, 25th percentile,
median, 75th percentile, and maxmimum.
"""
function summarystats(a::AbstractArray{T}) where T<:Real
    m = mean(a)
    qs = quantile(a, [0.00, 0.25, 0.50, 0.75, 1.00])
    R = typeof(convert(AbstractFloat, zero(T)))
    SummaryStats{R}(
        convert(R, m),
        convert(R, qs[1]),
        convert(R, qs[2]),
        convert(R, qs[3]),
        convert(R, qs[4]),
        convert(R, qs[5]))
end
function Base.show(io::IO, ss::SummaryStats)
    println(io, "Summary Stats:")
    @printf(io, "Mean:           %.6f\n", ss.mean)
    @printf(io, "Minimum:        %.6f\n", ss.min)
    @printf(io, "1st Quartile:   %.6f\n", ss.q25)
    @printf(io, "Median:         %.6f\n", ss.median)
    @printf(io, "3rd Quartile:   %.6f\n", ss.q75)
    @printf(io, "Maximum:        %.6f\n", ss.max)
end
"""
    describe(a)
Pretty-print the summary statistics provided by [`summarystats`](@ref):
the mean, minimum, 25th percentile, median, 75th percentile, and
maximum.
"""
describe(a::AbstractArray) = describe(STDOUT, a)
function describe(io::IO, a::AbstractArray{T}) where T<:Real
    show(io, summarystats(a))
    println(io, "Length:         $(length(a))")
    println(io, "Type:           $(string(eltype(a)))")
end
function describe(io::IO, a::AbstractArray)
    println(io, "Summary Stats:")
    println(io, "Length:         $(length(a))")
    println(io, "Type:           $(string(eltype(a)))")
    println(io, "Number Unique:  $(length(unique(a)))")
    return
end
"""
    trim(x; prop=0.0, count=0)
Return a copy of `x` with either `count` or proportion `prop` of the highest
and lowest elements removed.  To compute the trimmed mean of `x` use
`mean(trim(x))`; to compute the variance use `trimvar(x)` (see [`trimvar`](@ref)).
```julia
julia> trim([1,2,3,4,5], prop=0.2)
3-element Array{Int64,1}:
 2
 3
 4
```
"""
function trim(x::AbstractVector; prop::Real=0.0, count::Integer=0)
    trim!(copy(x); prop=prop, count=count)
end
"""
    trim!(x; prop=0.0, count=0)
A variant of [`trim`](@ref) that modifies `x` in place.
"""
function trim!(x::AbstractVector; prop::Real=0.0, count::Integer=0)
    n = length(x)
    n > 0 || throw(ArgumentError("x can not be empty."))
    if count == 0
        0 <= prop < 0.5 || throw(ArgumentError("prop must satisfy 0 ≤ prop < 0.5."))
        count = floor(Int, n * prop)
    else
        prop == 0 || throw(ArgumentError("prop and count can not both be > 0."))
        0 <= count < n/2 || throw(ArgumentError("count must satisfy 0 ≤ count < length(x)/2."))
    end
    select!(x, (n-count+1):n)
    select!(x, 1:count)
    deleteat!(x, (n-count+1):n)
    deleteat!(x, 1:count)
    return x
end
"""
    winsor(x; prop=0.0, count=0)
Return a copy of `x` with either `count` or proportion `prop` of the lowest
elements of `x` replaced with the next-lowest, and an equal number of the
highest elements replaced with the previous-highest.  To compute the Winsorized
mean of `x` use `mean(winsor(x))`.
```julia
julia> winsor([1,2,3,4,5], prop=0.2)
5-element Array{Int64,1}:
 2
 2
 3
 4
 4
```
"""
function winsor(x::AbstractVector; prop::Real=0.0, count::Integer=0)
    winsor!(copy(x); prop=prop, count=count)
end
"""
    winsor!(x; prop=0.0, count=0)
A variant of [`winsor`](@ref) that modifies vector `x` in place.
"""
function winsor!(x::AbstractVector; prop::Real=0.0, count::Integer=0)
    n = length(x)
    n > 0 || throw(ArgumentError("x can not be empty."))
    if count == 0
        0 <= prop < 0.5 || throw(ArgumentError("prop must satisfy 0 ≤ prop < 0.5."))
        count = floor(Int, n * prop)
    else
        prop == 0 || throw(ArgumentError("prop and count can not both be > 0."))
        0 <= count < n/2 || throw(ArgumentError("count must satisfy 0 ≤ count < length(x)/2."))
    end
    select!(x, (n-count+1):n)
    select!(x, 1:count)
    x[1:count] = x[count+1]
    x[n-count+1:end] = x[n-count]
    return x
end
"""
    trimvar(x; prop=0.0, count=0)
Compute the variance of the trimmed mean of `x`. This function uses
the Winsorized variance, as described in Wilcox (2010).
"""
function trimvar(x::AbstractVector; prop::Real=0.0, count::Integer=0)
    n = length(x)
    n > 0 || throw(ArgumentError("x can not be empty."))
    if count == 0
        0 <= prop < 0.5 || throw(ArgumentError("prop must satisfy 0 ≤ prop < 0.5."))
        count = floor(Int, n * prop)
    else
        0 <= count < n/2 || throw(ArgumentError("count must satisfy 0 ≤ count < length(x)/2."))
        prop = count/n
    end
    return var(winsor(x, count=count)) / (n * (1 - 2prop)^2)
end
"""
    counteq(a, b)
Count the number of indices at which the elements of the arrays
`a` and `b` are equal.
"""
function counteq(a::AbstractArray, b::AbstractArray)
    n = length(a)
    length(b) == n || throw(DimensionMismatch("Inconsistent lengths."))
    c = 0
    for i = 1:n
        @inbounds if a[i] == b[i]
            c += 1
        end
    end
    return c
end
"""
    countne(a, b)
Count the number of indices at which the elements of the arrays
`a` and `b` are not equal.
"""
function countne(a::AbstractArray, b::AbstractArray)
    n = length(a)
    length(b) == n || throw(DimensionMismatch("Inconsistent lengths."))
    c = 0
    for i = 1:n
        @inbounds if a[i] != b[i]
            c += 1
        end
    end
    return c
end
"""
    sqL2dist(a, b)
Compute the squared L2 distance between two arrays: ``\\sum_{i=1}^n |a_i - b_i|^2``.
Efficient equivalent of `sumabs2(a - b)`.
"""
function sqL2dist(a::AbstractArray{T}, b::AbstractArray{T}) where T<:Number
    n = length(a)
    length(b) == n || throw(DimensionMismatch("Input dimension mismatch"))
    r = 0.0
    for i = 1:n
        @inbounds r += abs2(a[i] - b[i])
    end
    return r
end
"""
    L2dist(a, b)
Compute the L2 distance between two arrays: ``\\sqrt{\\sum_{i=1}^n |a_i - b_i|^2}``.
Efficient equivalent of `sqrt(sumabs2(a - b))`.
"""
L2dist(a::AbstractArray{T}, b::AbstractArray{T}) where {T<:Number} = sqrt(sqL2dist(a, b))
"""
    L1dist(a, b)
Compute the L1 distance between two arrays: ``\\sum_{i=1}^n |a_i - b_i|``.
Efficient equivalent of `sum(abs, a - b)`.
"""
function L1dist(a::AbstractArray{T}, b::AbstractArray{T}) where T<:Number
    n = length(a)
    length(b) == n || throw(DimensionMismatch("Input dimension mismatch"))
    r = 0.0
    for i = 1:n
        @inbounds r += abs(a[i] - b[i])
    end
    return r
end
"""
    Linfdist(a, b)
Compute the L∞ distance, also called the Chebyshev distance, between
two arrays: ``\\max_{i\\in1:n} |a_i - b_i|``.
Efficient equivalent of `maxabs(a - b)`.
"""
function Linfdist(a::AbstractArray{T}, b::AbstractArray{T}) where T<:Number
    n = length(a)
    length(b) == n || throw(DimensionMismatch("Input dimension mismatch"))
    r = 0.0
    for i = 1:n
        @inbounds v = abs(a[i] - b[i])
        if r < v
            r = v
        end
    end
    return r
end
"""
    gkldiv(a, b)
Compute the generalized Kullback-Leibler divergence between two arrays:
``\\sum_{i=1}^n (a_i \\log(a_i/b_i) - a_i + b_i)``.
Efficient equivalent of `sum(a*log(a/b)-a+b)`.
"""
function gkldiv(a::AbstractArray{T}, b::AbstractArray{T}) where T<:AbstractFloat
    n = length(a)
    r = 0.0
    for i = 1:n
        @inbounds ai = a[i]
        @inbounds bi = b[i]
        if ai > 0
            r += (ai * log(ai / bi) - ai + bi)
        else
            r += bi
        end
    end
    return r::Float64
end
"""
    meanad(a, b)
Return the mean absolute deviation between two arrays: `mean(abs(a - b))`.
"""
meanad(a::AbstractArray{T}, b::AbstractArray{T}) where {T<:Number} =
    L1dist(a, b) / length(a)
"""
    maxad(a, b)
Return the maximum absolute deviation between two arrays: `maxabs(a - b)`.
"""
maxad(a::AbstractArray{T}, b::AbstractArray{T}) where {T<:Number} = Linfdist(a, b)
"""
    msd(a, b)
Return the mean squared deviation between two arrays: `mean(abs2(a - b))`.
"""
msd(a::AbstractArray{T}, b::AbstractArray{T}) where {T<:Number} =
    sqL2dist(a, b) / length(a)
"""
    rmsd(a, b; normalize=false)
Return the root mean squared deviation between two optionally
normalized arrays. The root mean squared deviation is computed
as `sqrt(msd(a, b))`.
"""
function rmsd(a::AbstractArray{T}, b::AbstractArray{T}; normalize::Bool=false) where T<:Number
    v = sqrt(msd(a, b))
    if normalize
        amin, amax = extrema(a)
        v /= (amax - amin)
    end
    return v
end
"""
    psnr(a, b, maxv)
Compute the peak signal-to-noise ratio between two arrays `a` and `b`.
`maxv` is the maximum possible value either array can take. The PSNR
is computed as `10 * log10(maxv^2 / msd(a, b))`.
"""
function psnr(a::AbstractArray{T}, b::AbstractArray{T}, maxv::Real) where T<:Real
    20. * log10(maxv) - 10. * log10(msd(a, b))
end
function _symmetrize!(a::DenseMatrix)
    m, n = size(a)
    m == n || error("a must be a square matrix.")
    for j = 1:n
        @inbounds for i = j+1:n
            vl = a[i,j]
            vr = a[j,i]
            a[i,j] = a[j,i] = middle(vl, vr)
        end
    end
    return a
end
function _scalevars(x::DenseMatrix, s::DenseVector, vardim::Int)
    vardim == 1 ? Diagonal(s) * x :
    vardim == 2 ? x * Diagonal(s) :
    error("vardim should be either 1 or 2.")
end
scattermat_zm(x::DenseMatrix, vardim::Int) = Base.unscaled_covzm(x, vardim)
scattermat_zm(x::DenseMatrix, wv::AbstractWeights, vardim::Int) =
    _symmetrize!(Base.unscaled_covzm(x, _scalevars(x, values(wv), vardim), vardim))
"""
    scattermat(X, [wv::AbstractWeights]; mean=nothing, vardim=1)
Compute the scatter matrix, which is an unnormalized covariance matrix.
A weighting vector `wv` can be specified to weight
the estimate.
* `mean=nothing`: a known mean value. `nothing` indicates that the mean is
  unknown, and the function will compute the mean. Specifying `mean=0` indicates
  that the data are centered and hence there's no need to subtract the mean.
* `vardim=1`: the dimension along which the variables are organized.
  When `vardim = 1`, the variables are considered columns with observations in rows;
  when `vardim = 2`, variables are in rows with observations in columns.
"""
function scattermat end
"""
    cov(X, w::AbstractWeights; mean=nothing, vardim=1, corrected=false)
Compute the weighted covariance matrix. Similar to `var` and `std` the biased covariance
matrix (`corrected=false`) is computed by multiplying `scattermat(X, w)` by
``\\frac{1}{\\sum{w}}`` to normalize. However, the unbiased covariance matrix
(`corrected=true`) is dependent on the type of weights used:
* `AnalyticWeights`: ``\\frac{1}{\\sum w - \\sum {w^2} / \\sum w}``
* `FrequencyWeights`: ``\\frac{1}{\\sum{w} - 1}``
* `ProbabilityWeights`: ``\\frac{n}{(n - 1) \\sum w}`` where ``n`` equals `count(!iszero, w)`
* `Weights`: `ArgumentError` (bias correction not supported)
"""
cov
"""
    mean_and_cov(x, [wv::AbstractWeights]; vardim=1, corrected=false) -> (mean, cov)
Return the mean and covariance matrix as a tuple. A weighting
vector `wv` can be specified. `vardim` that designates whether
the variables are columns in the matrix (`1`) or rows (`2`).
Finally, bias correction is applied to the covariance calculation if
`corrected=true`. See [`cov`](@ref) documentation for more details.
"""
function mean_and_cov end
scattermatm(x::DenseMatrix, mean, vardim::Int=1) =
    scattermat_zm(x .- mean, vardim)
scattermatm(x::DenseMatrix, mean, wv::AbstractWeights, vardim::Int=1) =
    scattermat_zm(x .- mean, wv, vardim)
scattermat(x::DenseMatrix, vardim::Int=1) =
    scattermatm(x, Base.mean(x, vardim), vardim)
scattermat(x::DenseMatrix, wv::AbstractWeights, vardim::Int=1) =
    scattermatm(x, Base.mean(x, wv, vardim), wv, vardim)
Base.covm(x::DenseMatrix, mean, w::AbstractWeights, vardim::Int=1;
          corrected::DepBool=nothing) =
    scale!(scattermatm(x, mean, w, vardim), varcorrection(w, depcheck(:covm, corrected)))
Base.cov(x::DenseMatrix, w::AbstractWeights, vardim::Int=1; corrected::DepBool=nothing) =
    Base.covm(x, Base.mean(x, w, vardim), w, vardim; corrected=depcheck(:cov, corrected))
function Base.corm(x::DenseMatrix, mean, w::AbstractWeights, vardim::Int=1)
    c = Base.covm(x, mean, w, vardim; corrected=false)
    s = Base.stdm(x, w, mean, vardim; corrected=false)
    Base.cov2cor!(c, s)
end
"""
    cor(X, w::AbstractWeights, vardim=1)
Compute the Pearson correlation matrix of `X` along the dimension
`vardim` with a weighting `w` .
"""
Base.cor(x::DenseMatrix, w::AbstractWeights, vardim::Int=1) =
    Base.corm(x, Base.mean(x, w, vardim), w, vardim)
if VERSION >= v"0.7.0-DEV.755"
    function mean_and_cov(x::DenseMatrix, vardim::Int=1; corrected::Bool=true)
        m = mean(x, vardim)
        return m, Base.covm(x, m, vardim, corrected=corrected)
    end
else
    function mean_and_cov(x::DenseMatrix, vardim::Int=1; corrected::Bool=true)
        m = mean(x, vardim)
        return m, Base.covm(x, m, vardim, corrected)
    end
end
function mean_and_cov(x::DenseMatrix, wv::AbstractWeights, vardim::Int=1;
                      corrected::DepBool=nothing)
    m = mean(x, wv, vardim)
    return m, Base.cov(x, wv, vardim; corrected=depcheck(:mean_and_cov, corrected))
end
"""
    cov2cor(C, s)
Compute the correlation matrix from the covariance matrix `C` and a vector of standard
deviations `s`. Use `Base.cov2cor!` for an in-place version.
"""
cov2cor(C::AbstractMatrix, s::AbstractArray) = Base.cov2cor!(copy(C), s)
"""
    cor2cov(C, s)
Compute the covariance matrix from the correlation matrix `C` and a vector of standard
deviations `s`. Use `StatsBase.cor2cov!` for an in-place version.
"""
cor2cov(C::AbstractMatrix, s::AbstractArray) = cor2cov!(copy(C), s)
"""
    cor2cov!(C, s)
Converts the correlation matrix `C` to a covariance matrix in-place using a vector of
standard deviations `s`.
"""
function cor2cov!(C::AbstractMatrix, s::AbstractArray)
    n = length(s)
    size(C) == (n, n) || throw(DimensionMismatch("inconsistent dimensions"))
    for i in CartesianRange(size(C))
        @inbounds C[i] *= s[i[1]] * s[i[2]]
    end
    return C
end
const IntUnitRange{T<:Integer} = UnitRange{T}
if isdefined(Base, :ht_keyindex2)
    const ht_keyindex2! = Base.ht_keyindex2
else
    using Base: ht_keyindex2!
end
"""
    addcounts!(r, x, levels::UnitRange{<:Int}, [wv::AbstractWeights])
Add the number of occurrences in `x` of each value in `levels` to an existing
array `r`. If a weighting vector `wv` is specified, the sum of weights is used
rather than the raw counts.
"""
function addcounts!(r::AbstractArray, x::IntegerArray, levels::IntUnitRange)
    # add counts of integers from x to r
    k = length(levels)
    length(r) == k || throw(DimensionMismatch())
    m0 = levels[1]
    m1 = levels[end]
    b = m0 - 1
    @inbounds for i in 1 : length(x)
        xi = x[i]
        if m0 <= xi <= m1
            r[xi - b] += 1
        end
    end
    return r
end
function addcounts!(r::AbstractArray, x::IntegerArray, levels::IntUnitRange, wv::AbstractWeights)
    k = length(levels)
    length(r) == k || throw(DimensionMismatch())
    m0 = levels[1]
    m1 = levels[end]
    b = m0 - 1
    w = values(wv)
    @inbounds for i in 1 : length(x)
        xi = x[i]
        if m0 <= xi <= m1
            r[xi - b] += w[i]
        end
    end
    return r
end
"""
    counts(x, [wv::AbstractWeights])
    counts(x, levels::UnitRange{<:Integer}, [wv::AbstractWeights])
    counts(x, k::Integer, [wv::AbstractWeights])
Count the number of times each value in `x` occurs. If `levels` is provided, only values
falling in that range will be considered (the others will be ignored without
raising an error or a warning). If an integer `k` is provided, only values in the
range `1:k` will be considered.
If a weighting vector `wv` is specified, the sum of the weights is used rather than the
raw counts.
The output is a vector of length `length(levels)`.
"""
function counts end
counts(x::IntegerArray, levels::IntUnitRange) =
    addcounts!(zeros(Int, length(levels)), x, levels)
counts(x::IntegerArray, levels::IntUnitRange, wv::AbstractWeights) =
    addcounts!(zeros(eltype(wv), length(levels)), x, levels, wv)
counts(x::IntegerArray, k::Integer) = counts(x, 1:k)
counts(x::IntegerArray, k::Integer, wv::AbstractWeights) = counts(x, 1:k, wv)
counts(x::IntegerArray) = counts(x, span(x))
counts(x::IntegerArray, wv::AbstractWeights) = counts(x, span(x), wv)
"""
    proportions(x, levels=span(x), [wv::AbstractWeights])
Return the proportion of values in the range `levels` that occur in `x`.
Equivalent to `counts(x, levels) / length(x)`. If a weighting vector `wv`
is specified, the sum of the weights is used rather than the raw counts.
"""
proportions(x::IntegerArray, levels::IntUnitRange) = counts(x, levels) .* inv(length(x))
proportions(x::IntegerArray, levels::IntUnitRange, wv::AbstractWeights) =
    counts(x, levels, wv) .* inv(sum(wv))
"""
    proportions(x, k::Integer, [wv::AbstractWeights])
Return the proportion of integers in 1 to `k` that occur in `x`.
"""
proportions(x::IntegerArray, k::Integer) = proportions(x, 1:k)
proportions(x::IntegerArray, k::Integer, wv::AbstractWeights) = proportions(x, 1:k, wv)
proportions(x::IntegerArray) = proportions(x, span(x))
proportions(x::IntegerArray, wv::AbstractWeights) = proportions(x, span(x), wv)
function addcounts!(r::AbstractArray, x::IntegerArray, y::IntegerArray, levels::NTuple{2,IntUnitRange})
    # add counts of integers from x to r
    n = length(x)
    length(y) == n || throw(DimensionMismatch())
    xlevels, ylevels = levels
    kx = length(xlevels)
    ky = length(ylevels)
    size(r) == (kx, ky) || throw(DimensionMismatch())
    mx0 = xlevels[1]
    mx1 = xlevels[end]
    my0 = ylevels[1]
    my1 = ylevels[end]
    bx = mx0 - 1
    by = my0 - 1
    for i = 1:n
        xi = x[i]
        yi = y[i]
        if (mx0 <= xi <= mx1) && (my0 <= yi <= my1)
            r[xi - bx, yi - by] += 1
        end
    end
    return r
end
function addcounts!(r::AbstractArray, x::IntegerArray, y::IntegerArray,
                    levels::NTuple{2,IntUnitRange}, wv::AbstractWeights)
    # add counts of integers from x to r
    n = length(x)
    length(y) == length(wv) == n || throw(DimensionMismatch())
    xlevels, ylevels = levels
    kx = length(xlevels)
    ky = length(ylevels)
    size(r) == (kx, ky) || throw(DimensionMismatch())
    mx0 = xlevels[1]
    mx1 = xlevels[end]
    my0 = ylevels[1]
    my1 = ylevels[end]
    bx = mx0 - 1
    by = my0 - 1
    w = values(wv)
    for i = 1:n
        xi = x[i]
        yi = y[i]
        if (mx0 <= xi <= mx1) && (my0 <= yi <= my1)
            r[xi - bx, yi - by] += w[i]
        end
    end
    return r
end
function counts(x::IntegerArray, y::IntegerArray, levels::NTuple{2,IntUnitRange})
    addcounts!(zeros(Int, length(levels[1]), length(levels[2])), x, y, levels)
end
function counts(x::IntegerArray, y::IntegerArray, levels::NTuple{2,IntUnitRange}, wv::AbstractWeights)
    addcounts!(zeros(eltype(wv), length(levels[1]), length(levels[2])), x, y, levels, wv)
end
counts(x::IntegerArray, y::IntegerArray, levels::IntUnitRange) =
    counts(x, y, (levels, levels))
counts(x::IntegerArray, y::IntegerArray, levels::IntUnitRange, wv::AbstractWeights) =
    counts(x, y, (levels, levels), wv)
counts(x::IntegerArray, y::IntegerArray, ks::NTuple{2,Integer}) =
    counts(x, y, (1:ks[1], 1:ks[2]))
counts(x::IntegerArray, y::IntegerArray, ks::NTuple{2,Integer}, wv::AbstractWeights) =
    counts(x, y, (1:ks[1], 1:ks[2]), wv)
counts(x::IntegerArray, y::IntegerArray, k::Integer) = counts(x, y, (1:k, 1:k))
counts(x::IntegerArray, y::IntegerArray, k::Integer, wv::AbstractWeights) =
    counts(x, y, (1:k, 1:k), wv)
counts(x::IntegerArray, y::IntegerArray) = counts(x, y, (span(x), span(y)))
counts(x::IntegerArray, y::IntegerArray, wv::AbstractWeights) = counts(x, y, (span(x), span(y)), wv)
proportions(x::IntegerArray, y::IntegerArray, levels::NTuple{2,IntUnitRange}) =
    counts(x, y, levels) .* inv(length(x))
proportions(x::IntegerArray, y::IntegerArray, levels::NTuple{2,IntUnitRange}, wv::AbstractWeights) =
    counts(x, y, levels, wv) .* inv(sum(wv))
proportions(x::IntegerArray, y::IntegerArray, ks::NTuple{2,Integer}) =
    proportions(x, y, (1:ks[1], 1:ks[2]))
proportions(x::IntegerArray, y::IntegerArray, ks::NTuple{2,Integer}, wv::AbstractWeights) =
    proportions(x, y, (1:ks[1], 1:ks[2]), wv)
proportions(x::IntegerArray, y::IntegerArray, k::Integer) = proportions(x, y, (1:k, 1:k))
proportions(x::IntegerArray, y::IntegerArray, k::Integer, wv::AbstractWeights) =
    proportions(x, y, (1:k, 1:k), wv)
proportions(x::IntegerArray, y::IntegerArray) = proportions(x, y, (span(x), span(y)))
proportions(x::IntegerArray, y::IntegerArray, wv::AbstractWeights) =
    proportions(x, y, (span(x), span(y)), wv)
function _normalize_countmap(cm::Dict{T}, s::Real) where T
    r = Dict{T,Float64}()
    for (k, c) in cm
        r[k] = c / s
    end
    return r
end
"""
    addcounts!(dict, x[, wv]; alg = :auto)
Add counts based on `x` to a count map. New entries will be added if new values come up.
If a weighting vector `wv` is specified, the sum of the weights is used rather than the
raw counts.
`alg` can be one of:
- `:auto` (default): if `StatsBase.radixsort_safe(eltype(x)) == true` then use
                     `:radixsort`, otherwise use `:dict`.
- `:radixsort`:      if `radixsort_safe(eltype(x)) == true` then use the
                     [radix sort](https://en.wikipedia.org/wiki/Radix_sort)
                     algorithm to sort the input vector which will generally lead to
                     shorter running time. However the radix sort algorithm creates a
                     copy of the input vector and hence uses more RAM. Choose `:dict`
                     if the amount of available RAM is a limitation.
- `:dict`:           use `Dict`-based method which is generally slower but uses less
                     RAM and is safe for any data type.
"""
function addcounts!(cm::Dict{T}, x::AbstractArray{T}; alg = :auto) where T
    # if it's safe to be sorted using radixsort then it should be faster
    # albeit using more RAM
    if radixsort_safe(T) && (alg == :auto || alg == :radixsort)
        addcounts_radixsort!(cm, x)
    elseif alg == :radixsort
        throw(ArgumentError("`alg = :radixsort` is chosen but type `radixsort_safe($T)` did not return `true`; use `alg = :auto` or `alg = :dict` instead"))
    else
        addcounts_dict!(cm,x)
    end
    return cm
end
"""Dict-based addcounts method"""
function addcounts_dict!(cm::Dict{T}, x::AbstractArray{T}) where T
    for v in x
        index = ht_keyindex2!(cm, v)
        if index > 0
            @inbounds cm.vals[index] += 1
        else
            @inbounds Base._setindex!(cm, 1, v, -index)
        end
    end
    return cm
end
function addcounts!(cm::Dict{Bool}, x::AbstractArray{Bool}; alg = :ignored)
    sumx = sum(x)
    cm[true] = get(cm, true, 0) + sumx
    cm[false] = get(cm, false, 0) + length(x) - sumx
    cm
end
function addcounts!(cm::Dict{T}, x::AbstractArray{T}; alg = :ignored) where T <: Union{UInt8, UInt16, Int8, Int16}
    counts = zeros(Int, 2^(8sizeof(T)))
    @inbounds for xi in x
        counts[Int(xi) - typemin(T) + 1] += 1
    end
    for (i, c) in zip(typemin(T):typemax(T), counts)
        if c != 0
            index = ht_keyindex2!(cm, i)
            if index > 0
                @inbounds cm.vals[index] += c
            else
                @inbounds Base._setindex!(cm, c, i, -index)
            end
        end
    end
    cm
end
const BaseRadixSortSafeTypes = Union{Int8, Int16, Int32, Int64, Int128,
                                     UInt8, UInt16, UInt32, UInt64, UInt128,
                                     Float32, Float64}
"Can the type be safely sorted by radixsort"
radixsort_safe(::Type{T}) where {T<:BaseRadixSortSafeTypes} = true
radixsort_safe(::Type) = false
function addcounts_radixsort!(cm::Dict{T}, x::AbstractArray{T}) where T
    # sort the x using radixsort
    sx = sort(x, alg = RadixSort)::typeof(x)
    tmpcount = 1
    last_sx = sx[1]
    # now the data is sorted: can just run through and accumulate values before
    # adding into the Dict
    for i in 2:length(sx)
        sxi = sx[i]
        if last_sx == sxi
            tmpcount += 1
        else
            cm[last_sx] = tmpcount
            last_sx = sxi
            tmpcount = 1
        end
    end
    cm[sx[end]] = tmpcount
    return cm
end
function addcounts!(cm::Dict{T}, x::AbstractArray{T}, wv::AbstractVector{W}) where {T,W<:Real}
    n = length(x)
    length(wv) == n || throw(DimensionMismatch())
    w = values(wv)
    z = zero(W)
    for i = 1 : n
        @inbounds xi = x[i]
        @inbounds wi = w[i]
        cm[xi] = get(cm, xi, z) + wi
    end
    return cm
end
"""
    countmap(x; alg = :auto)
Return a dictionary mapping each unique value in `x` to its number
of occurrences.
- `:auto` (default): if `StatsBase.radixsort_safe(eltype(x)) == true` then use
                     `:radixsort`, otherwise use `:dict`.
- `:radixsort`:      if `radixsort_safe(eltype(x)) == true` then use the
                     [radix sort](https://en.wikipedia.org/wiki/Radix_sort)
                     algorithm to sort the input vector which will generally lead to
                     shorter running time. However the radix sort algorithm creates a
                     copy of the input vector and hence uses more RAM. Choose `:dict`
                     if the amount of available RAM is a limitation.
- `:dict`:           use `Dict`-based method which is generally slower but uses less
                     RAM and is safe for any data type.
"""
countmap(x::AbstractArray{T}; alg = :auto) where {T} = addcounts!(Dict{T,Int}(), x; alg = alg)
countmap(x::AbstractArray{T}, wv::AbstractVector{W}) where {T,W<:Real} = addcounts!(Dict{T,W}(), x, wv)
"""
    proportionmap(x)
Return a dictionary mapping each unique value in `x` to its
proportion in `x`.
"""
proportionmap(x::AbstractArray) = _normalize_countmap(countmap(x), length(x))
proportionmap(x::AbstractArray, wv::AbstractWeights) = _normalize_countmap(countmap(x, wv), sum(wv))
function _check_randparams(rks, x, p)
    n = length(rks)
    length(x) == length(p) == n || raise_dimerror()
    return n
end
function ordinalrank!(rks::AbstractArray, x::AbstractArray, p::IntegerArray)
    n = _check_randparams(rks, x, p)
    if n > 0
        i = 1
        while i <= n
            rks[p[i]] = i
            i += 1
        end
    end
    return rks
end
"""
    ordinalrank(x; lt = isless, rev::Bool = false)
Return the [ordinal ranking](https://en.wikipedia.org/wiki/Ranking#Ordinal_ranking_.28.221234.22_ranking.29)
("1234" ranking) of an array. The `lt` keyword allows providing a custom "less
than" function; use `rev=true` to reverse the sorting order.
All items in `x` are given distinct, successive ranks based on their
position in `sort(x; lt = lt, rev = rev)`.
Missing values are assigned rank `missing`.
"""
ordinalrank(x::AbstractArray; lt = isless, rev::Bool = false) =
    ordinalrank!(Array{Int}(uninitialized, size(x)), x, sortperm(x; lt = lt, rev = rev))
function competerank!(rks::AbstractArray, x::AbstractArray, p::IntegerArray)
    n = _check_randparams(rks, x, p)
    if n > 0
        p1 = p[1]
        v = x[p1]
        rks[p1] = k = 1
        i = 2
        while i <= n
            pi = p[i]
            xi = x[pi]
            if xi == v
                rks[pi] = k
            else
                rks[pi] = k = i
                v = xi
            end
            i += 1
        end
    end
    return rks
end
"""
    competerank(x; lt = isless, rev::Bool = false)
Return the [standard competition ranking](http://en.wikipedia.org/wiki/Ranking#Standard_competition_ranking_.28.221224.22_ranking.29)
("1224" ranking) of an array. The `lt` keyword allows providing a custom "less
than" function; use `rev=true` to reverse the sorting order.
Items that compare equal are given the same rank, then a gap is left
in the rankings the size of the number of tied items - 1.
Missing values are assigned rank `missing`.
"""
competerank(x::AbstractArray; lt = isless, rev::Bool = false) =
    competerank!(Array{Int}(uninitialized, size(x)), x, sortperm(x; lt = lt, rev = rev))
function denserank!(rks::AbstractArray, x::AbstractArray, p::IntegerArray)
    n = _check_randparams(rks, x, p)
    if n > 0
        p1 = p[1]
        v = x[p1]
        rks[p1] = k = 1
        i = 2
        while i <= n
            pi = p[i]
            xi = x[pi]
            if xi == v
                rks[pi] = k
            else
                rks[pi] = (k += 1)
                v = xi
            end
            i += 1
        end
    end
    return rks
end
"""
    denserank(x)
Return the [dense ranking](http://en.wikipedia.org/wiki/Ranking#Dense_ranking_.28.221223.22_ranking.29)
("1223" ranking) of an array. The `lt` keyword allows providing a custom "less
than" function; use `rev=true` to reverse the sorting order. Items that
compare equal receive the same ranking, and the next subsequent rank is
assigned with no gap.
Missing values are assigned rank `missing`.
"""
denserank(x::AbstractArray; lt = isless, rev::Bool = false) =
    denserank!(Array{Int}(uninitialized, size(x)), x, sortperm(x; lt = lt, rev = rev))
function tiedrank!(rks::AbstractArray, x::AbstractArray, p::IntegerArray)
    n = _check_randparams(rks, x, p)
    if n > 0
        v = x[p[1]]
        s = 1  # starting index of current range
        e = 2  # pass-by-end index of current range
        while e <= n
            cx = x[p[e]]
            if cx != v
                # fill average rank to s : e-1
                ar = (s + e - 1) / 2
                for i = s : e-1
                    rks[p[i]] = ar
                end
                # switch to next range
                s = e
                v = cx
            end
            e += 1
        end
        # the last range (e == n+1)
        ar = (s + n) / 2
        for i = s : n
            rks[p[i]] = ar
        end
    end
    return rks
end
"""
    tiedrank(x)
Return the [tied ranking](http://en.wikipedia.org/wiki/Ranking#Fractional_ranking_.28.221_2.5_2.5_4.22_ranking.29),
also called fractional or "1 2.5 2.5 4" ranking,
of an array. The `lt` keyword allows providing a custom "less
than" function; use `rev=true` to reverse the sorting order.
Items that compare equal receive the mean of the
rankings they would have been assigned under ordinal ranking.
Missing values are assigned rank `missing`.
"""
tiedrank(x::AbstractArray; lt = isless, rev::Bool = false) =
    tiedrank!(Array{Float64}(uninitialized, size(x)), x, sortperm(x; lt = lt, rev = rev))
for (f, f!, S) in zip([:ordinalrank, :competerank, :denserank, :tiedrank],
                      [:ordinalrank!, :competerank!, :denserank!, :tiedrank!],
                      [Int, Int, Int, Float64])
    @eval begin
        function $f(x::AbstractArray{>: Missing}; lt = isless, rev::Bool = false)
            inds = find(!ismissing, x)
            xv = disallowmissing(view(x, inds))
            sp = sortperm(xv; lt = lt, rev = rev)
            rks = missings($S, length(x))
            $(f!)(view(rks, inds), xv, sp)
            rks
        end
    end
end
function durbin!(r::AbstractVector{T}, y::AbstractVector{T}) where T<:BlasReal
    n = length(r)
    n <= length(y) || throw(DimensionMismatch("Auxiliary vector cannot be shorter than data vector"))
    y[1] = -r[1]
    β = one(T)
    α = -r[1]
    for k = 1:n-1
        β *= one(T) - α*α
        α = -r[k+1]
        for j = 1:k
            α -= r[k-j+1]*y[j]
        end
        α /= β
        for j = 1:div(k,2)
            tmp = y[j]
            y[j] += α*y[k-j+1]
            y[k-j+1] += α*tmp
        end
        if isodd(k) y[div(k,2)+1] *= one(T) + α end
        y[k+1] = α
    end
    return y
end
durbin(r::AbstractVector{T}) where {T<:BlasReal} = durbin!(r, zeros(T, length(r)))
function levinson!(r::AbstractVector{T}, b::AbstractVector{T}, x::AbstractVector{T}) where T<:BlasReal
    n = length(b)
    n == length(r) || throw(DimensionMismatch("Vectors must have same length"))
    n <= length(x) || throw(DimensionMismatch("Auxiliary vector cannot be shorter than data vector"))
    x[1] = b[1]
    b[1] = -r[2]/r[1]
    β = one(T)
    α = -r[2]/r[1]
    for k = 1:n-1
        β *= one(T) - α*α
        μ = b[k+1]
        for j = 2:k+1
            μ -= r[j]/r[1]*x[k-j+2]
        end
        μ /= β
        for j = 1:k
            x[j] += μ*b[k-j+1]
        end
        x[k+1] = μ
        if k < n - 1
            α = -r[k+2]
            for j = 2:k+1
                α -= r[j]*b[k-j+2]
            end
            α /= β*r[1]
            for j = 1:div(k,2)
                tmp = b[j]
                b[j] += α*b[k-j+1]
                b[k-j+1] += α*tmp
            end
            if isodd(k) b[div(k,2)+1] *= one(T) + α end
            b[k+1] = α
        end
    end
    for i = 1:n
        x[i] /= r[1]
    end
    return x
end
levinson(r::AbstractVector{T}, b::AbstractVector{T}) where {T<:BlasReal} = levinson!(r, copy(b), zeros(T, length(b)))
"""
    corspearman(x, y=x)
Compute Spearman's rank correlation coefficient. If `x` and `y` are vectors, the
output is a float, otherwise it's a matrix corresponding to the pairwise correlations
of the columns of `x` and `y`.
"""
corspearman(x::RealVector, y::RealVector) = cor(tiedrank(x), tiedrank(y))
corspearman(X::RealMatrix, Y::RealMatrix) =
    cor(mapslices(tiedrank, X, 1), mapslices(tiedrank, Y, 1))
corspearman(X::RealMatrix, y::RealVector) = cor(mapslices(tiedrank, X, 1), tiedrank(y))
corspearman(x::RealVector, Y::RealMatrix) = cor(tiedrank(x), mapslices(tiedrank, Y, 1))
corspearman(X::RealMatrix) = (Z = mapslices(tiedrank, X, 1); cor(Z, Z))
function corkendall!(x::RealVector, y::RealVector)
    if any(isnan, x) || any(isnan, y) return NaN end
    n = length(x)
    if n != length(y) error("Vectors must have same length") end
    # Initial sorting
    pm = sortperm(y)
    x[:] = x[pm]
    y[:] = y[pm]
    pm[:] = sortperm(x)
    x[:] = x[pm]
    # Counting ties in x and y
    iT = 1
    nT = 0
    iU = 1
    nU = 0
    for i = 2:n
        if x[i] == x[i-1]
            iT += 1
        else
            nT += iT*(iT - 1)
            iT = 1
        end
        if y[i] == y[i-1]
            iU += 1
        else
            nU += iU*(iU - 1)
            iU = 1
        end
    end
    if iT > 1 nT += iT*(iT - 1) end
    nT = div(nT,2)
    if iU > 1 nU += iU*(iU - 1) end
    nU = div(nU,2)
    # Sort y after x
    y[:] = y[pm]
    # Calculate double ties
    iV = 1
    nV = 0
    jV = 1
    for i = 2:n
        if x[i] == x[i-1] && y[i] == y[i-1]
            iV += 1
        else
            nV += iV*(iV - 1)
            iV = 1
        end
    end
    if iV > 1 nV += iV*(iV - 1) end
    nV = div(nV,2)
    nD = div(n*(n - 1),2)
    return (nD - nT - nU + nV - 2swaps!(y)) / (sqrt(nD - nT) * sqrt(nD - nU))
end
"""
    corkendall(x, y=x)
Compute Kendall's rank correlation coefficient, τ. `x` and `y` must both be either
matrices or vectors.
"""
corkendall(x::RealVector, y::RealVector) = corkendall!(float(copy(x)), float(copy(y)))
corkendall(X::RealMatrix, y::RealVector) = Float64[corkendall!(float(X[:,i]), float(copy(y))) for i in 1:size(X, 2)]
corkendall(x::RealVector, Y::RealMatrix) = (n = size(Y,2); reshape(Float64[corkendall!(float(copy(x)), float(Y[:,i])) for i in 1:n], 1, n))
corkendall(X::RealMatrix, Y::RealMatrix) = Float64[corkendall!(float(X[:,i]), float(Y[:,j])) for i in 1:size(X, 2), j in 1:size(Y, 2)]
function corkendall(X::RealMatrix)
    n = size(X, 2)
    C = eye(n)
    for j = 2:n
        for i = 1:j-1
            C[i,j] = corkendall!(X[:,i],X[:,j])
            C[j,i] = C[i,j]
        end
    end
    return C
end
function swaps!(x::RealVector)
    n = length(x)
    if n == 1 return 0 end
    n2 = div(n, 2)
    xl = view(x, 1:n2)
    xr = view(x, n2+1:n)
    nsl = swaps!(xl)
    nsr = swaps!(xr)
    sort!(xl)
    sort!(xr)
    return nsl + nsr + mswaps(xl,xr)
end
function mswaps(x::RealVector, y::RealVector)
    i = 1
    j = 1
    nSwaps = 0
    n = length(x)
    while i <= n && j <= length(y)
        if y[j] < x[i]
            nSwaps += n - i + 1
            j += 1
        else
            i += 1
        end
    end
    return nSwaps
end
default_laglen(lx::Int) = min(lx-1, round(Int,10*log10(lx)))
check_lags(lx::Int, lags::AbstractVector) = (maximum(lags) < lx || error("lags must be less than the sample length."))
function demean_col!(z::AbstractVector{T}, x::AbstractMatrix{T}, j::Int, demean::Bool) where T<:RealFP
    m = size(x, 1)
    @assert m == length(z)
    b = m * (j-1)
    if demean
        s = zero(T)
        for i = 1 : m
            s += x[b + i]
        end
        mv = s / m
        for i = 1 : m
            z[i] = x[b + i] - mv
        end
    else
        copy!(z, 1, x, b+1, m)
    end
    z
end
default_autolags(lx::Int) = 0 : default_laglen(lx)
_autodot(x::AbstractVector{<:RealFP}, lx::Int, l::Int) = dot(x, 1:lx-l, x, 1+l:lx)
"""
    autocov!(r, x, lags; demean=true)
Compute the autocovariance of a vector or matrix `x` at `lags` and store the result
in `r`. `demean` denotes whether the mean of `x` should be subtracted from `x`
before computing the autocovariance.
If `x` is a vector, `r` must be a vector of the same length as `x`.
If `x` is a matrix, `r` must be a matrix of size `(length(lags), size(x,2))`, and
where each column in the result will correspond to a column in `x`.
The output is not normalized. See [`autocor!`](@ref) for a method with normalization.
"""
function autocov!(r::RealVector, x::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = length(x)
    m = length(lags)
    length(r) == m || throw(DimensionMismatch())
    check_lags(lx, lags)
    z::Vector{T} = demean ? x .- mean(x) : x
    for k = 1 : m  # foreach lag value
        r[k] = _autodot(z, lx, lags[k]) / lx
    end
    return r
end
function autocov!(r::RealMatrix, x::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = size(x, 1)
    ns = size(x, 2)
    m = length(lags)
    size(r) == (m, ns) || throw(DimensionMismatch())
    check_lags(lx, lags)
    z = Vector{T}(uninitialized, lx)
    for j = 1 : ns
        demean_col!(z, x, j, demean)
        for k = 1 : m
            r[k,j] = _autodot(z, lx, lags[k]) / lx
        end
    end
    return r
end
"""
    autocov(x, [lags]; demean=true)
Compute the autocovariance of a vector or matrix `x`, optionally specifying
the `lags` at which to compute the autocovariance. `demean` denotes whether
the mean of `x` should be subtracted from `x` before computing the autocovariance.
If `x` is a vector, return a vector of the same length as `x`.
If `x` is a matrix, return a matrix of size `(length(lags), size(x,2))`,
where each column in the result corresponds to a column in `x`.
When left unspecified, the lags used are the integers from 0 to
`min(size(x,1)-1, 10*log10(size(x,1)))`.
The output is not normalized. See [`autocor`](@ref) for a function with normalization.
"""
function autocov(x::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    autocov!(Vector{fptype(T)}(uninitialized, length(lags)), float(x), lags; demean=demean)
end
function autocov(x::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    autocov!(Matrix{fptype(T)}(uninitialized, length(lags), size(x,2)), float(x), lags; demean=demean)
end
autocov(x::AbstractVecOrMat{<:Real}; demean::Bool=true) = autocov(x, default_autolags(size(x,1)); demean=demean)
"""
    autocor!(r, x, lags; demean=true)
Compute the autocorrelation function (ACF) of a vector or matrix `x` at `lags`
and store the result in `r`. `demean` denotes whether the mean of `x` should
be subtracted from `x` before computing the ACF.
If `x` is a vector, `r` must be a vector of the same length as `x`.
If `x` is a matrix, `r` must be a matrix of size `(length(lags), size(x,2))`, and
where each column in the result will correspond to a column in `x`.
The output is normalized by the variance of `x`, i.e. so that the lag 0
autocorrelation is 1. See [`autocov!`](@ref) for the unnormalized form.
"""
function autocor!(r::RealVector, x::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = length(x)
    m = length(lags)
    length(r) == m || throw(DimensionMismatch())
    check_lags(lx, lags)
    z::Vector{T} = demean ? x .- mean(x) : x
    zz = dot(z, z)
    for k = 1 : m  # foreach lag value
        r[k] = _autodot(z, lx, lags[k]) / zz
    end
    return r
end
function autocor!(r::RealMatrix, x::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = size(x, 1)
    ns = size(x, 2)
    m = length(lags)
    size(r) == (m, ns) || throw(DimensionMismatch())
    check_lags(lx, lags)
    z = Vector{T}(uninitialized, lx)
    for j = 1 : ns
        demean_col!(z, x, j, demean)
        zz = dot(z, z)
        for k = 1 : m
            r[k,j] = _autodot(z, lx, lags[k]) / zz
        end
    end
    return r
end
"""
    autocor(x, [lags]; demean=true)
Compute the autocorrelation function (ACF) of a vector or matrix `x`,
optionally specifying the `lags`. `demean` denotes whether the mean
of `x` should be subtracted from `x` before computing the ACF.
If `x` is a vector, return a vector of the same length as `x`.
If `x` is a matrix, return a matrix of size `(length(lags), size(x,2))`,
where each column in the result corresponds to a column in `x`.
When left unspecified, the lags used are the integers from 0 to
`min(size(x,1)-1, 10*log10(size(x,1)))`.
The output is normalized by the variance of `x`, i.e. so that the lag 0
autocorrelation is 1. See [`autocov`](@ref) for the unnormalized form.
"""
function autocor(x::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    autocor!(Vector{fptype(T)}(uninitialized, length(lags)), float(x), lags; demean=demean)
end
function autocor(x::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    autocor!(Matrix{fptype(T)}(uninitialized, length(lags), size(x,2)), float(x), lags; demean=demean)
end
autocor(x::AbstractVecOrMat{<:Real}; demean::Bool=true) = autocor(x, default_autolags(size(x,1)); demean=demean)
default_crosslags(lx::Int) = (l=default_laglen(lx); -l:l)
_crossdot(x::AbstractVector{T}, y::AbstractVector{T}, lx::Int, l::Int) where {T<:RealFP} =
    (l >= 0 ? dot(x, 1:lx-l, y, 1+l:lx) : dot(x, 1-l:lx, y, 1:lx+l))
"""
    crosscov!(r, x, y, lags; demean=true)
Compute the cross covariance function (CCF) between real-valued vectors or matrices
`x` and `y` at `lags` and store the result in `r`. `demean` specifies whether the
respective means of `x` and `y` should be subtracted from them before computing their
CCF.
If both `x` and `y` are vectors, `r` must be a vector of the same length as
`lags`. If either `x` is a matrix and `y` is a vector, `r` must be a matrix of size
`(length(lags), size(x, 2))`; if `x` is a vector and `y` is a matrix, `r` must be a matrix
of size `(length(lags), size(y, 2))`. If both `x` and `y` are matrices, `r` must be a
three-dimensional array of size `(length(lags), size(x, 2), size(y, 2))`.
The output is not normalized. See [`crosscor!`](@ref) for a function with normalization.
"""
function crosscov!(r::RealVector, x::AbstractVector{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = length(x)
    m = length(lags)
    (length(y) == lx && length(r) == m) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zx::Vector{T} = demean ? x .- mean(x) : x
    zy::Vector{T} = demean ? y .- mean(y) : y
    for k = 1 : m  # foreach lag value
        r[k] = _crossdot(zx, zy, lx, lags[k]) / lx
    end
    return r
end
function crosscov!(r::RealMatrix, x::AbstractMatrix{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = size(x, 1)
    ns = size(x, 2)
    m = length(lags)
    (length(y) == lx && size(r) == (m, ns)) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zx = Vector{T}(uninitialized, lx)
    zy::Vector{T} = demean ? y .- mean(y) : y
    for j = 1 : ns
        demean_col!(zx, x, j, demean)
        for k = 1 : m
            r[k,j] = _crossdot(zx, zy, lx, lags[k]) / lx
        end
    end
    return r
end
function crosscov!(r::RealMatrix, x::AbstractVector{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = length(x)
    ns = size(y, 2)
    m = length(lags)
    (size(y, 1) == lx && size(r) == (m, ns)) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zx::Vector{T} = demean ? x .- mean(x) : x
    zy = Vector{T}(uninitialized, lx)
    for j = 1 : ns
        demean_col!(zy, y, j, demean)
        for k = 1 : m
            r[k,j] = _crossdot(zx, zy, lx, lags[k]) / lx
        end
    end
    return r
end
function crosscov!(r::AbstractArray{T,3}, x::AbstractMatrix{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = size(x, 1)
    nx = size(x, 2)
    ny = size(y, 2)
    m = length(lags)
    (size(y, 1) == lx && size(r) == (m, nx, ny)) || throw(DimensionMismatch())
    check_lags(lx, lags)
    # cached (centered) columns of x
    zxs = Vector{Vector{T}}(0)
    sizehint!(zxs, nx)
    for j = 1 : nx
        xj = x[:,j]
        if demean
            mv = mean(xj)
            for i = 1 : lx
                xj[i] -= mv
            end
        end
        push!(zxs, xj)
    end
    zx = Vector{T}(uninitialized, lx)
    zy = Vector{T}(uninitialized, lx)
    for j = 1 : ny
        demean_col!(zy, y, j, demean)
        for i = 1 : nx
            zx = zxs[i]
            for k = 1 : m
                r[k,i,j] = _crossdot(zx, zy, lx, lags[k]) / lx
            end
        end
    end
    return r
end
"""
    crosscov(x, y, [lags]; demean=true)
Compute the cross covariance function (CCF) between real-valued vectors or
matrices `x` and `y`, optionally specifying the `lags`. `demean` specifies
whether the respective means of `x` and `y` should be subtracted from them
before computing their CCF.
If both `x` and `y` are vectors, return a vector of the same length as
`lags`. Otherwise, compute cross covariances between each pairs of columns in `x` and `y`.
When left unspecified, the lags used are the integers from
`-min(size(x,1)-1, 10*log10(size(x,1)))` to `min(size(x,1), 10*log10(size(x,1)))`.
The output is not normalized. See [`crosscor`](@ref) for a function with normalization.
"""
function crosscov(x::AbstractVector{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscov!(Vector{fptype(T)}(uninitialized, length(lags)), float(x), float(y), lags; demean=demean)
end
function crosscov(x::AbstractMatrix{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscov!(Matrix{fptype(T)}(uninitialized, length(lags), size(x,2)), float(x), float(y), lags; demean=demean)
end
function crosscov(x::AbstractVector{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscov!(Matrix{fptype(T)}(uninitialized, length(lags), size(y,2)), float(x), float(y), lags; demean=demean)
end
function crosscov(x::AbstractMatrix{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscov!(Array{fptype(T),3}(uninitialized, length(lags), size(x,2), size(y,2)), float(x), float(y), lags; demean=demean)
end
crosscov(x::AbstractVecOrMat{T}, y::AbstractVecOrMat{T}; demean::Bool=true) where {T<:Real} = crosscov(x, y, default_crosslags(size(x,1)); demean=demean)
"""
    crosscor!(r, x, y, lags; demean=true)
Compute the cross correlation between real-valued vectors or matrices `x` and `y` at
`lags` and store the result in `r`. `demean` specifies whether the respective means of
`x` and `y` should be subtracted from them before computing their cross correlation.
If both `x` and `y` are vectors, `r` must be a vector of the same length as
`lags`. If either `x` is a matrix and `y` is a vector, `r` must be a matrix of size
`(length(lags), size(x, 2))`; if `x` is a vector and `y` is a matrix, `r` must be a matrix
of size `(length(lags), size(y, 2))`. If both `x` and `y` are matrices, `r` must be a
three-dimensional array of size `(length(lags), size(x, 2), size(y, 2))`.
The output is normalized by `sqrt(var(x)*var(y))`. See [`crosscov!`](@ref) for the
unnormalized form.
"""
function crosscor!(r::RealVector, x::AbstractVector{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = length(x)
    m = length(lags)
    (length(y) == lx && length(r) == m) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zx::Vector{T} = demean ? x .- mean(x) : x
    zy::Vector{T} = demean ? y .- mean(y) : y
    sc = sqrt(dot(zx, zx) * dot(zy, zy))
    for k = 1 : m  # foreach lag value
        r[k] = _crossdot(zx, zy, lx, lags[k]) / sc
    end
    return r
end
function crosscor!(r::RealMatrix, x::AbstractMatrix{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = size(x, 1)
    ns = size(x, 2)
    m = length(lags)
    (length(y) == lx && size(r) == (m, ns)) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zx = Vector{T}(uninitialized, lx)
    zy::Vector{T} = demean ? y .- mean(y) : y
    yy = dot(zy, zy)
    for j = 1 : ns
        demean_col!(zx, x, j, demean)
        sc = sqrt(dot(zx, zx) * yy)
        for k = 1 : m
            r[k,j] = _crossdot(zx, zy, lx, lags[k]) / sc
        end
    end
    return r
end
function crosscor!(r::RealMatrix, x::AbstractVector{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = length(x)
    ns = size(y, 2)
    m = length(lags)
    (size(y, 1) == lx && size(r) == (m, ns)) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zx::Vector{T} = demean ? x .- mean(x) : x
    zy = Vector{T}(uninitialized, lx)
    xx = dot(zx, zx)
    for j = 1 : ns
        demean_col!(zy, y, j, demean)
        sc = sqrt(xx * dot(zy, zy))
        for k = 1 : m
            r[k,j] = _crossdot(zx, zy, lx, lags[k]) / sc
        end
    end
    return r
end
function crosscor!(r::AbstractArray{T,3}, x::AbstractMatrix{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = size(x, 1)
    nx = size(x, 2)
    ny = size(y, 2)
    m = length(lags)
    (size(y, 1) == lx && size(r) == (m, nx, ny)) || throw(DimensionMismatch())
    check_lags(lx, lags)
    # cached (centered) columns of x
    zxs = Vector{Vector{T}}(0)
    sizehint!(zxs, nx)
    xxs = Vector{T}(uninitialized, nx)
    for j = 1 : nx
        xj = x[:,j]
        if demean
            mv = mean(xj)
            for i = 1 : lx
                xj[i] -= mv
            end
        end
        push!(zxs, xj)
        xxs[j] = dot(xj, xj)
    end
    zx = Vector{T}(uninitialized, lx)
    zy = Vector{T}(uninitialized, lx)
    for j = 1 : ny
        demean_col!(zy, y, j, demean)
        yy = dot(zy, zy)
        for i = 1 : nx
            zx = zxs[i]
            sc = sqrt(xxs[i] * yy)
            for k = 1 : m
                r[k,i,j] = _crossdot(zx, zy, lx, lags[k]) / sc
            end
        end
    end
    return r
end
"""
    crosscor(x, y, [lags]; demean=true)
Compute the cross correlation between real-valued vectors or matrices `x` and `y`,
optionally specifying the `lags`. `demean` specifies whether the respective means of
`x` and `y` should be subtracted from them before computing their cross correlation.
If both `x` and `y` are vectors, return a vector of the same length as
`lags`. Otherwise, compute cross covariances between each pairs of columns in `x` and `y`.
When left unspecified, the lags used are the integers from
`-min(size(x,1)-1, 10*log10(size(x,1)))` to `min(size(x,1), 10*log10(size(x,1)))`.
The output is normalized by `sqrt(var(x)*var(y))`. See [`crosscov`](@ref) for the
unnormalized form.
"""
function crosscor(x::AbstractVector{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscor!(Vector{fptype(T)}(uninitialized, length(lags)), float(x), float(y), lags; demean=demean)
end
function crosscor(x::AbstractMatrix{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscor!(Matrix{fptype(T)}(uninitialized, length(lags), size(x,2)), float(x), float(y), lags; demean=demean)
end
function crosscor(x::AbstractVector{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscor!(Matrix{fptype(T)}(uninitialized, length(lags), size(y,2)), float(x), float(y), lags; demean=demean)
end
function crosscor(x::AbstractMatrix{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscor!(Array{fptype(T),3}(uninitialized, length(lags), size(x,2), size(y,2)), float(x), float(y), lags; demean=demean)
end
crosscor(x::AbstractVecOrMat{T}, y::AbstractVecOrMat{T}; demean::Bool=true) where {T<:Real} = crosscor(x, y, default_crosslags(size(x,1)); demean=demean)
function pacf_regress!(r::RealMatrix, X::AbstractMatrix{T}, lags::IntegerVector, mk::Integer) where T<:RealFP
    lx = size(X, 1)
    tmpX = ones(T, lx, mk + 1)
    for j = 1 : size(X,2)
        for l = 1 : mk
            for i = 1+l:lx
                tmpX[i,l+1] = X[i-l,j]
            end
        end
        for i = 1 : length(lags)
            l = lags[i]
            sX = view(tmpX, 1+l:lx, 1:l+1)
            r[i,j] = l == 0 ? 1 : (cholfact!(sX'sX)\(sX'view(X, 1+l:lx, j)))[end]
        end
    end
    r
end
function pacf_yulewalker!(r::RealMatrix, X::AbstractMatrix{T}, lags::IntegerVector, mk::Integer) where T<:RealFP
    tmp = Vector{T}(uninitialized, mk)
    for j = 1 : size(X,2)
        acfs = autocor(X[:,j], 1:mk)
        for i = 1 : length(lags)
            l = lags[i]
            r[i,j] = l == 0 ? 1 : l == 1 ? acfs[i] : -durbin!(view(acfs, 1:l), tmp)[l]
        end
    end
end
"""
    pacf!(r, X, lags; method=:regression)
Compute the partial autocorrelation function (PACF) of a matrix `X` at `lags` and
store the result in `r`. `method` designates the estimation method. Recognized values
are `:regression`, which computes the partial autocorrelations via successive
regression models, and `:yulewalker`, which computes the partial autocorrelations
using the Yule-Walker equations.
`r` must be a matrix of size `(length(lags), size(x, 2))`.
"""
function pacf!(r::RealMatrix, X::AbstractMatrix{T}, lags::IntegerVector; method::Symbol=:regression) where T<:RealFP
    lx = size(X, 1)
    m = length(lags)
    minlag, maxlag = extrema(lags)
    (0 <= minlag && 2maxlag < lx) || error("Invalid lag value.")
    size(r) == (m, size(X,2)) || throw(DimensionMismatch())
    if method == :regression
        pacf_regress!(r, X, lags, maxlag)
    elseif method == :yulewalker
        pacf_yulewalker!(r, X, lags, maxlag)
    else
        error("Invalid method: $method")
    end
    return r
end
"""
    pacf(X, lags; method=:regression)
Compute the partial autocorrelation function (PACF) of a real-valued vector
or matrix `X` at `lags`. `method` designates the estimation method. Recognized
values are `:regression`, which computes the partial autocorrelations via successive
regression models, and `:yulewalker`, which computes the partial autocorrelations
using the Yule-Walker equations.
If `x` is a vector, return a vector of the same length as `lags`.
If `x` is a matrix, return a matrix of size `(length(lags), size(x, 2))`,
where each column in the result corresponds to a column in `x`.
"""
function pacf(X::AbstractMatrix{T}, lags::IntegerVector; method::Symbol=:regression) where T<:Real
    pacf!(Matrix{fptype(T)}(uninitialized, length(lags), size(X,2)), float(X), lags; method=method)
end
function pacf(x::AbstractVector{T}, lags::IntegerVector; method::Symbol=:regression) where T<:Real
    vec(pacf(reshape(x, length(x), 1), lags, method=method))
end
"""
    ecdf(X)
Return an empirical cumulative distribution function (ECDF) based on a vector of samples
given in `X`.
Note: this is a higher-level function that returns a function, which can then be applied
to evaluate CDF values on other samples.
"""
function ecdf(X::RealVector{T}) where T<:Real
    Xs = sort(X)
    n = length(X)
    ef(x::Real) = searchsortedlast(Xs, x) / n
    function ef(v::RealVector)
        ord = sortperm(v)
        m = length(v)
        r = Vector{T}(uninitialized, m)
        r0 = 0
        i = 1
        for x in Xs
            while i <= m && x > v[ord[i]]
                r[ord[i]] = r0
                i += 1
            end
            r0 += 1
            if i > m
                break
            end
        end
        while i <= m
            r[ord[i]] = n
            i += 1
        end
        return r / n
    end
    return ef
end
using Base.Cartesian
import Base: show, ==, push!, append!, float, norm, normalize, normalize!
function _check_closed_arg(closed::Symbol, funcsym)
    if closed == :default_left
        Base.depwarn("Default for keyword argument \"closed\" has changed from :right to :left.\n" *
                     "To avoid this warning, specify closed=:right or closed=:left as appropriate.",
                     funcsym)
        :left
    else
        closed
    end
end
@inline Base.@propagate_inbounds @generated function _multi_getindex(i::Integer, c::AbstractArray...)
    N = length(c)
    result_expr = Expr(:tuple)
    for j in 1:N
        push!(result_expr.args, :(c[$j][i]))
    end
    result_expr
end
@generated function _promote_edge_types(edges::NTuple{N,AbstractVector}) where N
    promote_type(map(eltype, edges.parameters)...)
end
function histrange(v::AbstractArray{T}, n::Integer, closed::Symbol=:default_left) where T
    closed = _check_closed_arg(closed,:histrange)
    F = float(T)
    nv = length(v)
    if nv == 0 && n < 0
        throw(ArgumentError("number of bins must be ≥ 0 for an empty array, got $n"))
    elseif nv > 0 && n < 1
        throw(ArgumentError("number of bins must be ≥ 1 for a non-empty array, got $n"))
    elseif nv == 0
        return zero(F):zero(F)
    end
    lo, hi = extrema(v)
    histrange(F(lo), F(hi), n, closed)
end
function histrange(lo::F, hi::F, n::Integer, closed::Symbol=:default_left) where F
    closed = _check_closed_arg(closed,:histrange)
    if hi == lo
        start = F(hi)
        step = one(F)
        divisor = one(F)
        len = one(F)
    else
        bw = (F(hi) - F(lo)) / n
        lbw = log10(bw)
        if lbw >= 0
            step = exp10(floor(lbw))
            r = bw / step
            if r <= 1.1
                nothing
            elseif r <= 2.2
                step *= 2
            elseif r <= 5.5
                step *= 5
            else
                step *= 10
            end
            divisor = one(F)
            start = step*floor(lo/step)
            len = ceil((hi - start)/step)
        else
            divisor = exp10(-floor(lbw))
            r = bw * divisor
            if r <= 1.1
                nothing
            elseif r <= 2.2
                divisor /= 2
            elseif r <= 5.5
                divisor /= 5
            else
                divisor /= 10
            end
            step = one(F)
            start = floor(lo*divisor)
            len = ceil(hi*divisor - start)
        end
    end
    # fix up endpoints
    if closed == :right #(,]
        while lo <= start/divisor
            start -= step
        end
        while (start + (len-1)*step)/divisor < hi
            len += one(F)
        end
    else
        while lo < start/divisor
            start -= step
        end
        while (start + (len-1)*step)/divisor <= hi
            len += one(F)
        end
    end
    Base.floatrange(start,step,len,divisor)
end
histrange(vs::NTuple{N,AbstractVector},nbins::NTuple{N,Integer},closed::Symbol) where {N} =
    map((v,n) -> histrange(v,n,closed),vs,nbins)
histrange(vs::NTuple{N,AbstractVector},nbins::Integer,closed::Symbol) where {N} =
    map(v -> histrange(v,nbins,closed),vs)
function sturges(n)  # Sturges' formula
    n==0 && return one(n)
    ceil(Integer, log2(n))+1
end
abstract type AbstractHistogram{T<:Real,N,E} end
mutable struct Histogram{T<:Real,N,E} <: AbstractHistogram{T,N,E}
    edges::E
    weights::Array{T,N}
    closed::Symbol
    isdensity::Bool
    function Histogram{T,N,E}(edges::NTuple{N,AbstractArray}, weights::Array{T,N},
                              closed::Symbol, isdensity::Bool=false) where {T,N,E}
        closed == :right || closed == :left || error("closed must :left or :right")
        isdensity && !(T <: AbstractFloat) && error("Density histogram must have float-type weights")
        _edges_nbins(edges) == size(weights) || error("Histogram edge vectors must be 1 longer than corresponding weight dimensions")
        new{T,N,E}(edges,weights,closed,isdensity)
    end
end
Histogram(edges::NTuple{N,AbstractVector}, weights::AbstractArray{T,N},
          closed::Symbol=:default_left, isdensity::Bool=false) where {T,N} =
    Histogram{T,N,typeof(edges)}(edges,weights,_check_closed_arg(closed,:Histogram),isdensity)
Histogram(edges::NTuple{N,AbstractVector}, ::Type{T}, closed::Symbol=:default_left,
          isdensity::Bool=false) where {T,N} =
    Histogram(edges,zeros(T,_edges_nbins(edges)...),_check_closed_arg(closed,:Histogram),isdensity)
Histogram(edges::NTuple{N,AbstractVector}, closed::Symbol=:default_left,
          isdensity::Bool=false) where {N} =
    Histogram(edges,Int,_check_closed_arg(closed,:Histogram),isdensity)
function show(io::IO, h::AbstractHistogram)
    println(io, typeof(h))
    println(io,"edges:")
    for e in h.edges
        println(io,"  ",e)
    end
    println(io,"weights: ",h.weights)
    println(io,"closed: ",h.closed)
    print(io,"isdensity: ",h.isdensity)
end
(==)(h1::Histogram,h2::Histogram) = (==)(h1.edges,h2.edges) && (==)(h1.weights,h2.weights) && (==)(h1.closed,h2.closed) && (==)(h1.isdensity,h2.isdensity)
binindex(h::AbstractHistogram{T,1}, x::Real) where {T} = binindex(h, (x,))[1]
binindex(h::Histogram{T,N}, xs::NTuple{N,Real}) where {T,N} =
    map((edge, x) -> _edge_binindex(edge, h.closed, x), h.edges, xs)
@inline function _edge_binindex(edge::AbstractVector, closed::Symbol, x::Real)
    if closed == :right
        searchsortedfirst(edge, x) - 1
    else
        searchsortedlast(edge, x)
    end
end
binvolume(h::AbstractHistogram{T,1}, binidx::Integer) where {T} = binvolume(h, (binidx,))
binvolume(::Type{V}, h::AbstractHistogram{T,1}, binidx::Integer) where {V,T} = binvolume(V, h, (binidx,))
binvolume(h::Histogram{T,N}, binidx::NTuple{N,Integer}) where {T,N} =
    binvolume(_promote_edge_types(h.edges), h, binidx)
binvolume(::Type{V}, h::Histogram{T,N}, binidx::NTuple{N,Integer}) where {V,T,N} =
    prod(map((edge, i) -> _edge_binvolume(V, edge, i), h.edges, binidx))
@inline _edge_binvolume(::Type{V}, edge::AbstractVector, i::Integer) where {V} = V(edge[i+1]) - V(edge[i])
@inline _edge_binvolume(::Type{V}, edge::AbstractRange, i::Integer) where {V} = V(step(edge))
@inline _edge_binvolume(edge::AbstractVector, i::Integer) = _edge_binvolume(eltype(edge), edge, i)
@inline _edges_nbins(edges::NTuple{N,AbstractVector}) where {N} = map(_edge_nbins, edges)
@inline _edge_nbins(edge::AbstractVector) = length(edge) - 1
Histogram(edge::AbstractVector, weights::AbstractVector{T}, closed::Symbol=:default_left, isdensity::Bool=false) where {T} =
    Histogram((edge,), weights, closed, isdensity)
Histogram(edge::AbstractVector, ::Type{T}, closed::Symbol=:default_left, isdensity::Bool=false) where {T} =
    Histogram((edge,), T, closed, isdensity)
Histogram(edge::AbstractVector, closed::Symbol=:default_left, isdensity::Bool=false) =
    Histogram((edge,), closed, isdensity)
push!(h::AbstractHistogram{T,1}, x::Real, w::Real) where {T} = push!(h, (x,), w)
push!(h::AbstractHistogram{T,1}, x::Real) where {T} = push!(h,x,one(T))
append!(h::AbstractHistogram{T,1}, v::AbstractVector) where {T} = append!(h, (v,))
append!(h::AbstractHistogram{T,1}, v::AbstractVector, wv::Union{AbstractVector,AbstractWeights}) where {T} = append!(h, (v,), wv)
fit(::Type{Histogram{T}},v::AbstractVector, edg::AbstractVector; closed::Symbol=:default_left) where {T} =
    fit(Histogram{T},(v,), (edg,), closed=closed)
fit(::Type{Histogram{T}},v::AbstractVector; closed::Symbol=:default_left, nbins=sturges(length(v))) where {T} =
    fit(Histogram{T},(v,); closed=closed, nbins=nbins)
fit(::Type{Histogram{T}},v::AbstractVector, wv::AbstractWeights, edg::AbstractVector; closed::Symbol=:default_left) where {T} =
    fit(Histogram{T},(v,), wv, (edg,), closed=closed)
fit(::Type{Histogram{T}},v::AbstractVector, wv::AbstractWeights; closed::Symbol=:default_left, nbins=sturges(length(v))) where {T} =
    fit(Histogram{T}, (v,), wv; closed=closed, nbins=nbins)
fit(::Type{Histogram}, v::AbstractVector, wv::AbstractWeights{W}, args...; kwargs...) where {W} = fit(Histogram{W}, v, wv, args...; kwargs...)
function push!(h::Histogram{T,N},xs::NTuple{N,Real},w::Real) where {T,N}
    h.isdensity && error("Density histogram must have float-type weights")
    idx = binindex(h, xs)
    if checkbounds(Bool, h.weights, idx...)
        @inbounds h.weights[idx...] += w
    end
    h
end
function push!(h::Histogram{T,N},xs::NTuple{N,Real},w::Real) where {T<:AbstractFloat,N}
    idx = binindex(h, xs)
    if checkbounds(Bool, h.weights, idx...)
        @inbounds h.weights[idx...] += h.isdensity ? w / binvolume(h, idx) : w
    end
    h
end
push!(h::AbstractHistogram{T,N},xs::NTuple{N,Real}) where {T,N} = push!(h,xs,one(T))
function append!(h::AbstractHistogram{T,N}, vs::NTuple{N,AbstractVector}) where {T,N}
    @inbounds for i in eachindex(vs...)
        xs = _multi_getindex(i, vs...)
        push!(h, xs, one(T))
    end
    h
end
function append!(h::AbstractHistogram{T,N}, vs::NTuple{N,AbstractVector}, wv::AbstractVector) where {T,N}
    @inbounds for i in eachindex(wv, vs...)
        xs = _multi_getindex(i, vs...)
        push!(h, xs, wv[i])
    end
    h
end
append!(h::AbstractHistogram{T,N}, vs::NTuple{N,AbstractVector}, wv::AbstractWeights) where {T,N} = append!(h, vs, values(wv))
function _nbins_tuple(vs::NTuple{N,AbstractVector}, nbins) where N
    template = map(length, vs)
    result = broadcast((t, x) -> typeof(t)(x), template, nbins)
    result::typeof(template)
end
fit(::Type{Histogram{T}}, vs::NTuple{N,AbstractVector}, edges::NTuple{N,AbstractVector}; closed::Symbol=:default_left) where {T,N} =
    append!(Histogram(edges, T, _check_closed_arg(closed,:fit), false), vs)
function fit(::Type{Histogram{T}}, vs::NTuple{N,AbstractVector}; closed::Symbol=:default_left, nbins=sturges(length(vs[1]))) where {T,N}
    closed = _check_closed_arg(closed,:fit)
    fit(Histogram{T}, vs, histrange(vs,_nbins_tuple(vs, nbins),closed); closed=closed)
end
fit(::Type{Histogram{T}}, vs::NTuple{N,AbstractVector}, wv::AbstractWeights{W}, edges::NTuple{N,AbstractVector}; closed::Symbol=:default_left) where {T,N,W} =
    append!(Histogram(edges, T, _check_closed_arg(closed,:fit), false), vs, wv)
function fit(::Type{Histogram{T}}, vs::NTuple{N,AbstractVector}, wv::AbstractWeights; closed::Symbol=:default_left, nbins=sturges(length(vs[1]))) where {T,N}
    closed = _check_closed_arg(closed,:fit)
    fit(Histogram{T}, vs, wv, histrange(vs,_nbins_tuple(vs, nbins),closed); closed=closed)
end
"""
    fit(Histogram, data[, weight][, edges]; closed=:right, nbins)
Fit a histogram to `data`.
* `data`: either a vector (for a 1-dimensional histogram), or a tuple of
  vectors of equal length (for an *n*-dimensional histogram).
* `weight`: an optional `AbstractWeights` (of the same length as the
  data vectors), denoting the weight each observation contributes to the
  bin. If no weight vector is supplied, each observation has weight 1.
* `edges`: a vector (typically an `AbstractRange` object), or tuple of vectors, that gives
  the edges of the bins along each dimension. If no edges are provided, these
  are determined from the data.
* `closed=:right`: if `:left`, the bin intervals are left-closed [a,b);
  if `:right` (the default), intervals are right-closed (a,b].
* `nbins`: if no `edges` argument is supplied, the approximate number of bins to use
  along each dimension (can be either a single integer, or a tuple of integers).
```julia
h = fit(Histogram, rand(100))
h = fit(Histogram, rand(100), 0:0.1:1.0)
h = fit(Histogram, rand(100), nbins=10)
h = fit(Histogram, rand(100), weights(rand(100)), 0:0.1:1.0)
h = fit(Histogram, [20], 0:20:100)
h = fit(Histogram, [20], 0:20:100, closed=:left)
h = fit(Histogram, (rand(100),rand(100)))
h = fit(Histogram, (rand(100),rand(100)),nbins=10)
```
"""
fit(::Type{Histogram}, args...; kwargs...) = fit(Histogram{Int}, args...; kwargs...)
fit(::Type{Histogram}, vs::NTuple{N,AbstractVector}, wv::AbstractWeights{W}, args...; kwargs...) where {N,W} = fit(Histogram{W}, vs, wv, args...; kwargs...)
norm_type(h::Histogram{T,N}) where {T,N} =
    promote_type(T, _promote_edge_types(h.edges))
norm_type(::Type{T}) where {T<:Integer} = promote_type(T, Int64)
norm_type(::Type{T}) where {T<:AbstractFloat} = promote_type(T, Float64)
"""
    norm(h::Histogram)
Calculate the norm of histogram `h` as the absolute value of its integral.
"""
@generated function norm(h::Histogram{T,N}) where {T,N}
    quote
        edges = h.edges
        weights = h.weights
        SumT = norm_type(h)
        v_0 = 1
        s_0 = zero(SumT)
        @inbounds @nloops(
            $N, i, weights,
            d -> begin
                v_{$N-d+1} = v_{$N-d} * _edge_binvolume(SumT, edges[d], i_d)
                s_{$N-d+1} = zero(SumT)
            end,
            d -> begin
                s_{$N-d} += s_{$N-d+1}
            end,
            begin
                $(Symbol("s_$(N)")) += (@nref $N weights i) * $(Symbol("v_$N"))
            end
        )
        s_0
    end
end
float(h::Histogram{T,N}) where {T<:AbstractFloat,N} = h
float(h::Histogram{T,N}) where {T,N} = Histogram(h.edges, float(h.weights), h.closed, h.isdensity)
"""
    normalize!(h::Histogram{T,N}, aux_weights::Array{T,N}...; mode::Symbol=:pdf) where {T<:AbstractFloat,N}
Normalize the histogram `h` and optionally scale one or more auxiliary weight
arrays appropriately. See description of `normalize` for details. Returns `h`.
"""
@generated function normalize!(h::Histogram{T,N}, aux_weights::Array{T,N}...; mode::Symbol=:pdf) where {T<:AbstractFloat,N}
    quote
        edges = h.edges
        weights = h.weights
        for A in aux_weights
            (size(A) != size(weights)) && throw(DimensionMismatch("aux_weights must have same size as histogram weights"))
        end
        if mode == :none
            # nothing to do
        elseif mode == :pdf || mode == :density || mode == :probability
            if h.isdensity
                if mode == :pdf || mode == :probability
                    # histogram already represents a density, just divide weights by norm
                    s = 1/norm(h)
                    weights .*= s
                    for A in aux_weights
                        A .*= s
                    end
                else
                    # :density - histogram already represents a density, nothing to do
                end
            else
                if mode == :pdf || mode == :density
                    # Divide weights by bin volume, for :pdf also divide by sum of weights
                    SumT = norm_type(h)
                    vs_0 = (mode == :pdf) ? sum(SumT(x) for x in weights) : one(SumT)
                    @inbounds @nloops $N i weights d->(vs_{$N-d+1} = vs_{$N-d} * _edge_binvolume(SumT, edges[d], i_d)) begin
                        (@nref $N weights i) /= $(Symbol("vs_$N"))
                        for A in aux_weights
                            (@nref $N A i) /= $(Symbol("vs_$N"))
                        end
                    end
                    h.isdensity = true
                else
                    # :probability - divide weights by sum of weights
                    nf = inv(sum(weights))
                    weights .*= nf
                    for A in aux_weights
                        A .*= nf
                    end
                end
            end
        else
            throw(ArgumentError("Normalization mode must be :pdf, :density, :probability or :none"))
        end
        h
    end
end
"""
    normalize(h::Histogram{T,N}; mode::Symbol=:pdf) where {T,N}
Normalize the histogram `h`.
Valid values for `mode` are:
*  `:pdf`: Normalize by sum of weights and bin sizes. Resulting histogram
   has norm 1 and represents a PDF.
* `:density`: Normalize by bin sizes only. Resulting histogram represents
   count density of input and does not have norm 1. Will not modify the
   histogram if it already represents a density (`h.isdensity == 1`).
* `:probability`: Normalize by sum of weights only. Resulting histogram
   represents the fraction of probability mass for each bin and does not have
   norm 1.
*  `:none`: Leaves histogram unchanged. Useful to simplify code that has to
   conditionally apply different modes of normalization.
Successive application of both `:probability` and `:density` normalization (in
any order) is equivalent to `:pdf` normalization.
"""
normalize(h::Histogram{T,N}; mode::Symbol=:pdf) where {T,N} =
    normalize!(deepcopy(float(h)), mode = mode)
"""
    normalize(h::Histogram{T,N}, aux_weights::Array{T,N}...; mode::Symbol=:pdf) where {T,N}
Normalize the histogram `h` and rescales one or more auxiliary weight arrays
at the same time (`aux_weights` may, e.g., contain estimated statistical
uncertainties). The values of the auxiliary arrays are scaled by the same
factor as the corresponding histogram weight values. Returns a tuple of the
normalized histogram and scaled auxiliary weights.
"""
function normalize(h::Histogram{T,N}, aux_weights::Array{T,N}...; mode::Symbol=:pdf) where {T,N}
    h_fltcp = deepcopy(float(h))
    aux_weights_fltcp = map(x -> deepcopy(float(x)), aux_weights)
    normalize!(h_fltcp, aux_weights_fltcp..., mode = mode)
    (h_fltcp, aux_weights_fltcp...)
end
"""
    zero(h::Histogram)
Create a new histogram with the same binning, type and shape of weights
and the same properties (`closed` and `isdensity`) as `h`, with all weights
set to zero.
"""
Base.zero(h::Histogram{T,N,E}) where {T,N,E} =
    Histogram{T,N,E}(deepcopy(h.edges), zero(h.weights), h.closed, h.isdensity)
"""
    merge!(target::Histogram, others::Histogram...)
Update histogram `target` by merging it with the histograms `others`. See
`merge(histogram::Histogram, others::Histogram...)` for details.
"""
function Base.merge!(target::Histogram, others::Histogram...)
    for h in others
        target.edges != h.edges && throw(ArgumentError("can't merge histograms with different binning"))
        size(target.weights) != size(h.weights) && throw(ArgumentError("can't merge histograms with different dimensions"))
        target.closed != h.closed && throw(ArgumentError("can't merge histograms with different closed left/right settings"))
        target.isdensity != h.isdensity && throw(ArgumentError("can't merge histograms with different isdensity settings"))
    end
    for h in others
        target.weights .+= h.weights
    end
    target
end
"""
    merge(h::Histogram, others::Histogram...)
Construct a new histogram by merging `h` with `others`. All histograms must
have the same binning, shape of weights and properties (`closed` and
`isdensity`). The weights of all histograms are summed up for each bin, the
weights of the resulting histogram will have the same type as those of `h`.
"""
Base.merge(h::Histogram, others::Histogram...) = merge!(zero(h), h, others...)
"""
    StatsBase.midpoints(v)
Calculate the midpoints (pairwise mean of consecutive elements).
"""
midpoints(v::AbstractVector) = [middle(v[i - 1], v[i]) for i in 2:length(v)]
midpoints(r::Range) = r[1:(end - 1)] + step(r) / 2
function norepeat(a::AbstractArray)
    sa = sort(a)
    for i = 2:length(a)
        if a[i] == a[i-1]
            return false
        end
    end
    return true
end
"""
    rle(v) -> (vals, lens)
Return the run-length encoding of a vector as a tuple. The
first element of the tuple is a vector of values of the input
and the second is the number of consecutive occurrences of each
element.
```jldoctest
julia> using StatsBase
julia> rle([1,1,1,2,2,3,3,3,3,2,2,2])
([1, 2, 3, 2], [3, 2, 4, 3])
```
"""
function rle(v::Vector{T}) where T
    n = length(v)
    vals = T[]
    lens = Int[]
    n>0 || return (vals,lens)
    cv = v[1]
    cl = 1
    i = 2
    @inbounds while i <= n
        vi = v[i]
        if vi == cv
            cl += 1
        else
            push!(vals, cv)
            push!(lens, cl)
            cv = vi
            cl = 1
        end
        i += 1
    end
    # the last section
    push!(vals, cv)
    push!(lens, cl)
    return (vals, lens)
end
"""
    inverse_rle(vals, lens)
Reconstruct a vector from its run-length encoding (see [`rle`](@ref)).
`vals` is a vector of the values and `lens` is a vector of the corresponding
run lengths.
"""
function inverse_rle(vals::AbstractVector{T}, lens::IntegerVector) where T
    m = length(vals)
    length(lens) == m || raise_dimerror()
    r = Vector{T}(uninitialized, sum(lens))
    p = 0
    @inbounds for i = 1 : m
        j = lens[i]
        v = vals[i]
        while j > 0
            r[p+=1] = v
            j -=1
        end
    end
    return r
end
"""
    indexmap(a)
Construct a dictionary that maps each unique value in `a` to
the index of its first occurrence in `a`.
"""
function indexmap(a::AbstractArray{T}) where T
    d = Dict{T,Int}()
    for i = 1 : length(a)
        @inbounds k = a[i]
        if !haskey(d, k)
            d[k] = i
        end
    end
    return d
end
"""
    levelsmap(a)
Construct a dictionary that maps each of the `n` unique values
in `a` to a number between 1 and `n`.
"""
function levelsmap(a::AbstractArray{T}) where T
    d = Dict{T,Int}()
    index = 1
    for i = 1 : length(a)
        @inbounds k = a[i]
        if !haskey(d, k)
            d[k] = index
            index += 1
        end
    end
    return d
end
"""
    indicatormat(x, k::Integer; sparse=false)
Construct a boolean matrix `I` of size `(k, length(x))` such that
`I[x[i], i] = true` and all other elements are set to `false`.
If `sparse` is `true`, the output will be a sparse matrix, otherwise
it will be dense (default).
```jldoctest
julia> using StatsBase
julia> indicatormat([1 2 2], 2)
2×3 Array{Bool,2}:
  true  false  false
 false   true   true
```
"""
function indicatormat(x::IntegerArray, k::Integer; sparse::Bool=false)
    sparse ? _indicatormat_sparse(x, k) : _indicatormat_dense(x, k)
end
"""
    indicatormat(x, c=sort(unique(x)); sparse=false)
Construct a boolean matrix `I` of size `(length(c), length(x))`.
Let `ci` be the index of `x[i]` in `c`. Then `I[ci, i] = true` and
all other elements are `false`.
"""
function indicatormat(x::AbstractArray, c::AbstractArray; sparse::Bool=false)
    sparse ? _indicatormat_sparse(x, c) : _indicatormat_dense(x, c)
end
indicatormat(x::AbstractArray; sparse::Bool=false) =
    indicatormat(x, sort!(unique(x)); sparse=sparse)
function _indicatormat_dense(x::IntegerArray, k::Integer)
    n = length(x)
    r = zeros(Bool, k, n)
    for i = 1 : n
        r[x[i], i] = true
    end
    return r
end
function _indicatormat_dense(x::AbstractArray{T}, c::AbstractArray{T}) where T
    d = indexmap(c)
    m = length(c)
    n = length(x)
    r = zeros(Bool, m, n)
    o = 0
    @inbounds for i = 1 : n
        xi = x[i]
        r[o + d[xi]] = true
        o += m
    end
    return r
end
_indicatormat_sparse(x::IntegerArray, k::Integer) = (n = length(x); sparse(x, 1:n, true, k, n))
function _indicatormat_sparse(x::AbstractArray{T}, c::AbstractArray{T}) where T
    d = indexmap(c)
    m = length(c)
    n = length(x)
    rinds = Vector{Int}(uninitialized, n)
    @inbounds for i = 1 : n
        rinds[i] = d[x[i]]
    end
    return sparse(rinds, 1:n, true, m, n)
end
using Base.Random: RangeGenerator
function direct_sample!(rng::AbstractRNG, a::UnitRange, x::AbstractArray)
    s = RangeGenerator(1:length(a))
    b = a[1] - 1
    if b == 0
        for i = 1:length(x)
            @inbounds x[i] = rand(rng, s)
        end
    else
        for i = 1:length(x)
            @inbounds x[i] = b + rand(rng, s)
        end
    end
    return x
end
direct_sample!(a::UnitRange, x::AbstractArray) = direct_sample!(Base.GLOBAL_RNG, a, x)
"""
    direct_sample!([rng], a::AbstractArray, x::AbstractArray)
Direct sampling: for each `j` in `1:k`, randomly pick `i` from `1:n`,
and set `x[j] = a[i]`, with `n=length(a)` and `k=length(x)`.
This algorithm consumes `k` random numbers.
"""
function direct_sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    s = RangeGenerator(1:length(a))
    for i = 1:length(x)
        @inbounds x[i] = a[rand(rng, s)]
    end
    return x
end
direct_sample!(a::AbstractArray, x::AbstractArray) = direct_sample!(Base.GLOBAL_RNG, a, x)
"""
    samplepair([rng], n)
Draw a pair of distinct integers between 1 and `n` without replacement.
Optionally specify a random number generator `rng` as the first argument
(defaults to `Base.GLOBAL_RNG`).
"""
function samplepair(rng::AbstractRNG, n::Int)
    i1 = rand(rng, 1:n)
    i2 = rand(rng, 1:n-1)
    return (i1, ifelse(i2 == i1, n, i2))
end
samplepair(n::Int) = samplepair(Base.GLOBAL_RNG, n)
"""
    samplepair([rng], a)
Draw a pair of distinct elements from the array `a` without replacement.
Optionally specify a random number generator `rng` as the first argument
(defaults to `Base.GLOBAL_RNG`).
"""
function samplepair(rng::AbstractRNG, a::AbstractArray)
    i1, i2 = samplepair(rng, length(a))
    return a[i1], a[i2]
end
samplepair(a::AbstractArray) = samplepair(Base.GLOBAL_RNG, a)
"""
    knuths_sample!([rng], a, x)
*Knuth's Algorithm S* for random sampling without replacement.
Reference: D. Knuth. *The Art of Computer Programming*. Vol 2, 3.4.2, p.142.
This algorithm consumes `length(a)` random numbers. It requires no additional
memory space. Suitable for the case where memory is tight.
"""
function knuths_sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray;
                        initshuffle::Bool=true)
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")
    # initialize
    for i = 1:k
        @inbounds x[i] = a[i]
    end
    if initshuffle
        @inbounds for j = 1:k
            l = rand(rng, j:k)
            if l != j
                t = x[j]
                x[j] = x[l]
                x[l] = t
            end
        end
    end
    # scan remaining
    s = RangeGenerator(1:k)
    for i = k+1:n
        if rand(rng) * i < k  # keep it with probability k / i
            @inbounds x[rand(rng, s)] = a[i]
        end
    end
    return x
end
knuths_sample!(a::AbstractArray, x::AbstractArray; initshuffle::Bool=true) =
    knuths_sample!(Base.GLOBAL_RNG, a, x; initshuffle=initshuffle)
"""
    fisher_yates_sample!([rng], a::AbstractArray, x::AbstractArray)
Fisher-Yates shuffling (with early termination).
Pseudo-code:
```
n = length(a)
k = length(x)
inds = collect(1:n)
for i = 1:k
    # swap element `i` with another random element in inds[i:n]
    # set element `i` in `x`
end
```
This algorithm consumes `k=length(x)` random numbers. It uses an integer array of
length `n=length(a)` internally to maintain the shuffled indices. It is considerably
faster than Knuth's algorithm especially when `n` is greater than `k`.
It is ``O(n)`` for initialization, plus ``O(k)`` for random shuffling
"""
function fisher_yates_sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")
    inds = Vector{Int}(uninitialized, n)
    for i = 1:n
        @inbounds inds[i] = i
    end
    @inbounds for i = 1:k
        j = rand(rng, i:n)
        t = inds[j]
        inds[j] = inds[i]
        inds[i] = t
        x[i] = a[t]
    end
    return x
end
fisher_yates_sample!(a::AbstractArray, x::AbstractArray) =
    fisher_yates_sample!(Base.GLOBAL_RNG, a, x)
"""
    self_avoid_sample!([rng], a::AbstractArray, x::AbstractArray)
Self-avoid sampling: use a set to maintain the index that has been sampled.
Each time draw a new index, if the index has already been sampled,
redraw until it draws an unsampled one.
This algorithm consumes about (or slightly more than) `k=length(x)` random numbers,
and requires ``O(k)`` memory to store the set of sampled indices.
Very fast when ``n >> k``, with `n=length(a)`.
However, if `k` is large and approaches ``n``, the rejection rate would increase
drastically, resulting in poorer performance.
"""
function self_avoid_sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")
    s = Set{Int}()
    sizehint!(s, k)
    rgen = RangeGenerator(1:n)
    # first one
    idx = rand(rng, rgen)
    x[1] = a[idx]
    push!(s, idx)
    # remaining
    for i = 2:k
        idx = rand(rng, rgen)
        while idx in s
            idx = rand(rng, rgen)
        end
        x[i] = a[idx]
        push!(s, idx)
    end
    return x
end
self_avoid_sample!(a::AbstractArray, x::AbstractArray) =
    self_avoid_sample!(Base.GLOBAL_RNG, a, x)
"""
    seqsample_a!([rng], a::AbstractArray, x::AbstractArray)
Random subsequence sampling using algorithm A described in the following paper (page 714):
Jeffrey Scott Vitter. "Faster Methods for Random Sampling". Communications of the ACM,
27 (7), July 1984.
This algorithm consumes ``O(n)`` random numbers, with `n=length(a)`.
The outputs are ordered.
"""
function seqsample_a!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")
    i = 0
    j = 0
    while k > 1
        u = rand(rng)
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        @inbounds x[j+=1] = a[i+=1]
        n -= 1
        k -= 1
    end
    if k > 0  # checking k > 0 is necessary: x can be empty
        s = trunc(Int, n * rand(rng))
        x[j+1] = a[i+(s+1)]
    end
    return x
end
seqsample_a!(a::AbstractArray, x::AbstractArray) = seqsample_a!(Base.GLOBAL_RNG, a, x)
"""
    seqsample_c!([rng], a::AbstractArray, x::AbstractArray)
Random subsequence sampling using algorithm C described in the following paper (page 715):
Jeffrey Scott Vitter. "Faster Methods for Random Sampling". Communications of the ACM,
27 (7), July 1984.
This algorithm consumes ``O(k^2)`` random numbers, with `k=length(x)`.
The outputs are ordered.
"""
function seqsample_c!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")
    i = 0
    j = 0
    while k > 1
        l = n - k + 1
        minv = l
        u = n
        while u >= l
            v = u * rand(rng)
            if v < minv
                minv = v
            end
            u -= 1
        end
        s = trunc(Int, minv) + 1
        x[j+=1] = a[i+=s]
        n -= s
        k -= 1
    end
    if k > 0
        s = trunc(Int, n * rand(rng))
        x[j+1] = a[i+(s+1)]
    end
    return x
end
seqsample_c!(a::AbstractArray, x::AbstractArray) = seqsample_c!(Base.GLOBAL_RNG, a, x)
"""
    sample([rng], a, [wv::AbstractWeights])
Select a single random element of `a`. Sampling probabilities are proportional to
the weights given in `wv`, if provided.
Optionally specify a random number generator `rng` as the first argument
(defaults to `Base.GLOBAL_RNG`).
"""
sample(rng::AbstractRNG, a::AbstractArray) = a[rand(rng, 1:length(a))]
sample(a::AbstractArray) = sample(Base.GLOBAL_RNG, a)
"""
    sample!([rng], a, [wv::AbstractWeights], x; replace=true, ordered=false)
Draw a random sample of `length(x)` elements from an array `a`
and store the result in `x`. A polyalgorithm is used for sampling.
Sampling probabilities are proportional to the weights given in `wv`,
if provided. `replace` dictates whether sampling is performed with
replacement and `order` dictates whether an ordered sample, also called
a sequential sample, should be taken.
Optionally specify a random number generator `rng` as the first argument
(defaults to `Base.GLOBAL_RNG`).
"""
function sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray;
                 replace::Bool=true, ordered::Bool=false)
    n = length(a)
    k = length(x)
    k == 0 && return x
    if replace  # with replacement
        if ordered
            sort!(direct_sample!(rng, a, x))
        else
            direct_sample!(rng, a, x)
        end
    else  # without replacement
        k <= n || error("Cannot draw more samples without replacement.")
        if ordered
            if n > 10 * k * k
                seqsample_c!(rng, a, x)
            else
                seqsample_a!(rng, a, x)
            end
        else
            if k == 1
                @inbounds x[1] = sample(rng, a)
            elseif k == 2
                @inbounds (x[1], x[2]) = samplepair(rng, a)
            elseif n < k * 24
                fisher_yates_sample!(rng, a, x)
            else
                self_avoid_sample!(rng, a, x)
            end
        end
    end
    return x
end
sample!(a::AbstractArray, x::AbstractArray; replace::Bool=true, ordered::Bool=false) =
    sample!(Base.GLOBAL_RNG, a, x; replace=replace, ordered=ordered)
"""
    sample([rng], a, [wv::AbstractWeights], n::Integer; replace=true, ordered=false)
Select a random, optionally weighted sample of size `n` from an array `a`
using a polyalgorithm. Sampling probabilities are proportional to the weights
given in `wv`, if provided. `replace` dictates whether sampling is performed
with replacement and `order` dictates whether an ordered sample, also called
a sequential sample, should be taken.
Optionally specify a random number generator `rng` as the first argument
(defaults to `Base.GLOBAL_RNG`).
"""
function sample(rng::AbstractRNG, a::AbstractArray{T}, n::Integer;
                replace::Bool=true, ordered::Bool=false) where T
    sample!(rng, a, Vector{T}(uninitialized, n); replace=replace, ordered=ordered)
end
sample(a::AbstractArray, n::Integer; replace::Bool=true, ordered::Bool=false) =
    sample(Base.GLOBAL_RNG, a, n; replace=replace, ordered=ordered)
"""
    sample([rng], a, [wv::AbstractWeights], dims::Dims; replace=true, ordered=false)
Select a random, optionally weighted sample from an array `a` specifying
the dimensions `dims` of the output array. Sampling probabilities are
proportional to the weights given in `wv`, if provided. `replace` dictates
whether sampling is performed with replacement and `order` dictates whether
an ordered sample, also called a sequential sample, should be taken.
Optionally specify a random number generator `rng` as the first argument
(defaults to `Base.GLOBAL_RNG`).
"""
function sample(rng::AbstractRNG, a::AbstractArray{T}, dims::Dims;
                replace::Bool=true, ordered::Bool=false) where T
    sample!(rng, a, Array{T}(uninitialized, dims), rng; replace=replace, ordered=ordered)
end
sample(a::AbstractArray, dims::Dims; replace::Bool=true, ordered::Bool=false) =
    sample(Base.GLOBAL_RNG, a, dims; replace=replace, ordered=ordered)
"""
    sample([rng], wv::AbstractWeights)
Select a single random integer in `1:length(wv)` with probabilities
proportional to the weights given in `wv`.
Optionally specify a random number generator `rng` as the first argument
(defaults to `Base.GLOBAL_RNG`).
"""
function sample(rng::AbstractRNG, wv::AbstractWeights)
    t = rand(rng) * sum(wv)
    w = values(wv)
    n = length(w)
    i = 1
    cw = w[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end
sample(wv::AbstractWeights) = sample(Base.GLOBAL_RNG, wv)
sample(rng::AbstractRNG, a::AbstractArray, wv::AbstractWeights) = a[sample(rng, wv)]
sample(a::AbstractArray, wv::AbstractWeights) = sample(Base.GLOBAL_RNG, a, wv)
"""
    direct_sample!([rng], a::AbstractArray, wv::AbstractWeights, x::AbstractArray)
Direct sampling.
Draw each sample by scanning the weight vector.
Noting `k=length(x)` and `n=length(a)`, this algorithm:
* consumes `k` random numbers
* has time complexity ``O(n k)``, as scanning the weight vector each time takes ``O(n)``
* requires no additional memory space.
"""
function direct_sample!(rng::AbstractRNG, a::AbstractArray,
                        wv::AbstractWeights, x::AbstractArray)
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("Inconsistent lengths."))
    for i = 1:length(x)
        x[i] = a[sample(rng, wv)]
    end
    return x
end
direct_sample!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    direct_sample!(Base.GLOBAL_RNG, a, wv, x)
function make_alias_table!(w::AbstractVector{Float64}, wsum::Float64,
                           a::AbstractVector{Float64},
                           alias::AbstractVector{Int})
    # Arguments:
    #
    #   w [in]:         input weights
    #   wsum [in]:      pre-computed sum(w)
    #
    #   a [out]:        acceptance probabilities
    #   alias [out]:    alias table
    #
    # Note: a and w can be the same way, then that away will be
    #       overriden inplace by acceptance probabilities
    #
    # Returns nothing
    #
    n = length(w)
    length(a) == length(alias) == n ||
        throw(DimensionMismatch("Inconsistent array lengths."))
    ac = n / wsum
    for i = 1:n
        @inbounds a[i] = w[i] * ac
    end
    larges = Vector{Int}(uninitialized, n)
    smalls = Vector{Int}(uninitialized, n)
    kl = 0  # actual number of larges
    ks = 0  # actual number of smalls
    for i = 1:n
        @inbounds ai = a[i]
        if ai > 1.0
            larges[kl+=1] = i  # push to larges
        elseif ai < 1.0
            smalls[ks+=1] = i  # push to smalls
        end
    end
    while kl > 0 && ks > 0
        s = smalls[ks]; ks -= 1  # pop from smalls
        l = larges[kl]; kl -= 1  # pop from larges
        @inbounds alias[s] = l
        @inbounds al = a[l] = (a[l] - 1.0) + a[s]
        if al > 1.0
            larges[kl+=1] = l  # push to larges
        else
            smalls[ks+=1] = l  # push to smalls
        end
    end
    # this loop should be redundant, except for rounding
    for i = 1:ks
        @inbounds a[smalls[i]] = 1.0
    end
    nothing
end
"""
    alias_sample!([rng], a::AbstractArray, wv::AbstractWeights, x::AbstractArray)
Alias method.
Build an alias table, and sample therefrom.
Reference: Walker, A. J. "An Efficient Method for Generating Discrete Random Variables
with General Distributions." *ACM Transactions on Mathematical Software* 3 (3): 253, 1977.
Noting `k=length(x)` and `n=length(a)`, this algorithm takes ``O(n \\log n)`` time
for building the alias table, and then ``O(1)`` to draw each sample. It consumes ``2 k`` random numbers.
"""
function alias_sample!(rng::AbstractRNG, a::AbstractArray, wv::AbstractWeights, x::AbstractArray)
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("Inconsistent lengths."))
    # create alias table
    ap = Vector{Float64}(uninitialized, n)
    alias = Vector{Int}(uninitialized, n)
    make_alias_table!(values(wv), sum(wv), ap, alias)
    # sampling
    s = RangeGenerator(1:n)
    for i = 1:length(x)
        j = rand(rng, s)
        x[i] = rand(rng) < ap[j] ? a[j] : a[alias[j]]
    end
    return x
end
alias_sample!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    alias_sample!(Base.GLOBAL_RNG, a, wv, x)
"""
    naive_wsample_norep!([rng], a::AbstractArray, wv::AbstractWeights, x::AbstractArray)
Naive implementation of weighted sampling without replacement.
It makes a copy of the weight vector at initialization, and sets the weight to zero
when the corresponding sample is picked.
Noting `k=length(x)` and `n=length(a)`, this algorithm consumes ``O(k)`` random numbers,
and has overall time complexity ``O(n k)``.
"""
function naive_wsample_norep!(rng::AbstractRNG, a::AbstractArray,
                              wv::AbstractWeights, x::AbstractArray)
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("Inconsistent lengths."))
    k = length(x)
    w = Vector{Float64}(uninitialized, n)
    copy!(w, values(wv))
    wsum = sum(wv)
    for i = 1:k
        u = rand(rng) * wsum
        j = 1
        c = w[1]
        while c < u && j < n
            @inbounds c += w[j+=1]
        end
        @inbounds x[i] = a[j]
        @inbounds wsum -= w[j]
        @inbounds w[j] = 0.0
    end
    return x
end
naive_wsample_norep!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    naive_wsample_norep!(Base.GLOBAL_RNG, a, wv, x)
"""
    efraimidis_a_wsample_norep!([rng], a::AbstractArray, wv::AbstractWeights, x::AbstractArray)
Weighted sampling without replacement using Efraimidis-Spirakis A algorithm.
Reference: Efraimidis, P. S., Spirakis, P. G. "Weighted random sampling with a reservoir."
*Information Processing Letters*, 97 (5), 181-185, 2006. doi:10.1016/j.ipl.2005.11.003.
Noting `k=length(x)` and `n=length(a)`, this algorithm takes ``O(n + k \\log k)``
processing time to draw ``k`` elements. It consumes ``n`` random numbers.
"""
function efraimidis_a_wsample_norep!(rng::AbstractRNG, a::AbstractArray,
                                     wv::AbstractWeights, x::AbstractArray)
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("a and wv must be of same length (got $n and $(length(wv)))."))
    k = length(x)
    # calculate keys for all items
    keys = randexp(rng, n)
    for i in 1:n
        @inbounds keys[i] = wv.values[i]/keys[i]
    end
    # return items with largest keys
    index = sortperm(keys; alg = PartialQuickSort(k), rev = true)
    for i in 1:k
        @inbounds x[i] = a[index[i]]
    end
    return x
end
efraimidis_a_wsample_norep!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    efraimidis_a_wsample_norep!(Base.GLOBAL_RNG, a, wv, x)
"""
    efraimidis_ares_wsample_norep!([rng], a::AbstractArray, wv::AbstractWeights, x::AbstractArray)
Implementation of weighted sampling without replacement using Efraimidis-Spirakis A-Res algorithm.
Reference: Efraimidis, P. S., Spirakis, P. G. "Weighted random sampling with a reservoir."
*Information Processing Letters*, 97 (5), 181-185, 2006. doi:10.1016/j.ipl.2005.11.003.
Noting `k=length(x)` and `n=length(a)`, this algorithm takes ``O(k \\log(k) \\log(n / k))``
processing time to draw ``k`` elements. It consumes ``n`` random numbers.
"""
function efraimidis_ares_wsample_norep!(rng::AbstractRNG, a::AbstractArray,
                                        wv::AbstractWeights, x::AbstractArray)
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("a and wv must be of same length (got $n and $(length(wv)))."))
    k = length(x)
    k > 0 || return x
    # initialize priority queue
    pq = Vector{Pair{Float64,Int}}(k)
    i = 0
    s = 0
    @inbounds for _s in 1:n
        s = _s
        w = wv.values[s]
        w < 0 && error("Negative weight found in weight vector at index $s")
        if w > 0
            i += 1
            pq[i] = (w/randexp(rng) => s)
        end
        i >= k && break
    end
    i < k && throw(DimensionMismatch("wv must have at least $k strictly positive entries (got $i)"))
    heapify!(pq)
    # set threshold
    @inbounds threshold = pq[1].first
    @inbounds for i in s+1:n
        w = wv.values[i]
        w < 0 && error("Negative weight found in weight vector at index $i")
        w > 0 || continue
        key = w/randexp(rng)
        # if key is larger than the threshold
        if key > threshold
            # update priority queue
            pq[1] = (key => i)
            percolate_down!(pq, 1)
            # update threshold
            threshold = pq[1].first
        end
    end
    # fill output array with items in descending order
    @inbounds for i in k:-1:1
        x[i] = a[heappop!(pq).second]
    end
    return x
end
efraimidis_ares_wsample_norep!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    efraimidis_ares_wsample_norep!(Base.GLOBAL_RNG, a, wv, x)
"""
    efraimidis_aexpj_wsample_norep!([rng], a::AbstractArray, wv::AbstractWeights, x::AbstractArray)
Implementation of weighted sampling without replacement using Efraimidis-Spirakis A-ExpJ algorithm.
Reference: Efraimidis, P. S., Spirakis, P. G. "Weighted random sampling with a reservoir."
*Information Processing Letters*, 97 (5), 181-185, 2006. doi:10.1016/j.ipl.2005.11.003.
Noting `k=length(x)` and `n=length(a)`, this algorithm takes ``O(k \\log(k) \\log(n / k))``
processing time to draw ``k`` elements. It consumes ``O(k \\log(n / k))`` random numbers.
"""
function efraimidis_aexpj_wsample_norep!(rng::AbstractRNG, a::AbstractArray,
                                         wv::AbstractWeights, x::AbstractArray)
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("a and wv must be of same length (got $n and $(length(wv)))."))
    k = length(x)
    k > 0 || return x
    # initialize priority queue
    pq = Vector{Pair{Float64,Int}}(k)
    i = 0
    s = 0
    @inbounds for _s in 1:n
        s = _s
        w = wv.values[s]
        w < 0 && error("Negative weight found in weight vector at index $s")
        if w > 0
            i += 1
            pq[i] = (w/randexp(rng) => s)
        end
        i >= k && break
    end
    i < k && throw(DimensionMismatch("wv must have at least $k strictly positive entries (got $i)"))
    heapify!(pq)
    # set threshold
    @inbounds threshold = pq[1].first
    X = threshold*randexp(rng)
    @inbounds for i in s+1:n
        w = wv.values[i]
        w < 0 && error("Negative weight found in weight vector at index $i")
        w > 0 || continue
        X -= w
        X <= 0 || continue
        # update priority queue
        t = exp(-w/threshold)
        pq[1] = (-w/log(t+rand(rng)*(1-t)) => i)
        percolate_down!(pq, 1)
        # update threshold
        threshold = pq[1].first
        X = threshold * randexp(rng)
    end
    # fill output array with items in descending order
    @inbounds for i in k:-1:1
        x[i] = a[heappop!(pq).second]
    end
    return x
end
efraimidis_aexpj_wsample_norep!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    efraimidis_aexpj_wsample_norep!(Base.GLOBAL_RNG, a, wv, x)
function sample!(rng::AbstractRNG, a::AbstractArray, wv::AbstractWeights, x::AbstractArray;
                 replace::Bool=true, ordered::Bool=false)
    n = length(a)
    k = length(x)
    if replace
        if ordered
            sort!(direct_sample!(rng, a, wv, x))
        else
            if n < 40
                direct_sample!(rng, a, wv, x)
            else
                t = ifelse(n < 500, 64, 32)
                if k < t
                    direct_sample!(rng, a, wv, x)
                else
                    alias_sample!(rng, a, wv, x)
                end
            end
        end
    else
        k <= n || error("Cannot draw $n samples from $k samples without replacement.")
        efraimidis_aexpj_wsample_norep!(rng, a, wv, x)
        if ordered
            sort!(x)
        end
    end
    return x
end
sample!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    sample!(Base.GLOBAL_RNG, a, wv, x)
sample(rng::AbstractRNG, a::AbstractArray{T}, wv::AbstractWeights, n::Integer;
       replace::Bool=true, ordered::Bool=false) where {T} =
    sample!(rng, a, wv, Vector{T}(uninitialized, n); replace=replace, ordered=ordered)
sample(a::AbstractArray, wv::AbstractWeights, n::Integer;
       replace::Bool=true, ordered::Bool=false) =
    sample(Base.GLOBAL_RNG, a, wv, n; replace=replace, ordered=ordered)
sample(rng::AbstractRNG, a::AbstractArray{T}, wv::AbstractWeights, dims::Dims;
       replace::Bool=true, ordered::Bool=false) where {T} =
    sample!(rng, a, wv, Array{T}(uninitialized, dims); replace=replace, ordered=ordered)
sample(a::AbstractArray, wv::AbstractWeights, dims::Dims;
       replace::Bool=true, ordered::Bool=false) =
    sample(Base.GLOBAL_RNG, a, wv, dims; replace=replace, ordered=ordered)
"""
    wsample!([rng], a, w, x; replace=true, ordered=false)
Select a weighted sample from an array `a` and store the result in `x`. Sampling
probabilities are proportional to the weights given in `w`. `replace` dictates
whether sampling is performed with replacement and `order` dictates whether an
ordered sample, also called a sequential sample, should be taken.
Optionally specify a random number generator `rng` as the first argument
(defaults to `Base.GLOBAL_RNG`).
"""
wsample!(rng::AbstractRNG, a::AbstractArray, w::RealVector, x::AbstractArray;
         replace::Bool=true, ordered::Bool=false) =
    sample!(rng, a, weights(w), x; replace=replace, ordered=ordered)
wsample!(a::AbstractArray, w::RealVector, x::AbstractArray;
         replace::Bool=true, ordered::Bool=false) =
    sample!(Base.GLOBAL_RNG, a, weights(w), x; replace=replace, ordered=ordered)
"""
    wsample([rng], [a], w)
Select a weighted random sample of size 1 from `a` with probabilities proportional
to the weights given in `w`. If `a` is not present, select a random weight from `w`.
Optionally specify a random number generator `rng` as the first argument
(defaults to `Base.GLOBAL_RNG`).
"""
wsample(rng::AbstractRNG, w::RealVector) = sample(rng, weights(w))
wsample(w::RealVector) = wsample(Base.GLOBAL_RNG, w)
wsample(rng::AbstractRNG, a::AbstractArray, w::RealVector) = sample(rng, a, weights(w))
wsample(a::AbstractArray, w::RealVector) = wsample(Base.GLOBAL_RNG, a, w)
"""
    wsample([rng], [a], w, n::Integer; replace=true, ordered=false)
Select a weighted random sample of size `n` from `a` with probabilities proportional
to the weights given in `w` if `a` is present, otherwise select a random sample of size
`n` of the weights given in `w`. `replace` dictates whether sampling is performed with
replacement and `order` dictates whether an ordered sample, also called a sequential
sample, should be taken.
Optionally specify a random number generator `rng` as the first argument
(defaults to `Base.GLOBAL_RNG`).
"""
wsample(rng::AbstractRNG, a::AbstractArray{T}, w::RealVector, n::Integer;
        replace::Bool=true, ordered::Bool=false) where {T} =
    wsample!(rng, a, w, Vector{T}(uninitialized, n); replace=replace, ordered=ordered)
wsample(a::AbstractArray, w::RealVector, n::Integer;
        replace::Bool=true, ordered::Bool=false) =
    wsample(Base.GLOBAL_RNG, a, w, n; replace=replace, ordered=ordered)
"""
    wsample([rng], [a], w, dims::Dims; replace=true, ordered=false)
Select a weighted random sample from `a` with probabilities proportional to the
weights given in `w` if `a` is present, otherwise select a random sample of size
`n` of the weights given in `w`. The dimensions of the output are given by `dims`.
Optionally specify a random number generator `rng` as the first argument
(defaults to `Base.GLOBAL_RNG`).
"""
wsample(rng::AbstractRNG, a::AbstractArray{T}, w::RealVector, dims::Dims;
        replace::Bool=true, ordered::Bool=false) where {T} =
    wsample!(rng, a, w, Array{T}(uninitialized, dims); replace=replace, ordered=ordered)
wsample(a::AbstractArray, w::RealVector, dims::Dims;
        replace::Bool=true, ordered::Bool=false) =
    wsample(Base.GLOBAL_RNG, a, w, dims; replace=replace, ordered=ordered)
abstract type StatisticalModel end
"""
    coef(obj::StatisticalModel)
Return the coefficients of the model.
"""
coef(obj::StatisticalModel) = error("coef is not defined for $(typeof(obj)).")
"""
    coefnames(obj::StatisticalModel)
Return the names of the coefficients.
"""
coefnames(obj::StatisticalModel) = error("coefnames is not defined for $(typeof(obj)).")
"""
    coeftable(obj::StatisticalModel)
Return a table of class `CoefTable` with coefficients and related statistics.
"""
coeftable(obj::StatisticalModel) = error("coeftable is not defined for $(typeof(obj)).")
"""
    confint(obj::StatisticalModel)
Compute confidence intervals for coefficients.
"""
confint(obj::StatisticalModel) = error("coefint is not defined for $(typeof(obj)).")
"""
    deviance(obj::StatisticalModel)
Return the deviance of the model relative to a reference, which is usually when applicable
the saturated model. It is equal, *up to a constant*, to ``-2 \\log L``, with ``L``
the likelihood of the model.
"""
deviance(obj::StatisticalModel) = error("deviance is not defined for $(typeof(obj)).")
"""
    nulldeviance(obj::StatisticalModel)
Return the deviance of the null model, that is the one including only the intercept.
"""
nulldeviance(obj::StatisticalModel) = error("nulldeviance is not defined for $(typeof(obj)).")
"""
    loglikelihood(obj::StatisticalModel)
Return the log-likelihood of the model.
"""
loglikelihood(obj::StatisticalModel) = error("loglikelihood is not defined for $(typeof(obj)).")
"""
    loglikelihood(obj::StatisticalModel)
Return the log-likelihood of the null model corresponding to model `obj`.
This is usually the model containing only the intercept.
"""
nullloglikelihood(obj::StatisticalModel) = error("nullloglikelihood is not defined for $(typeof(obj)).")
"""
    nobs(obj::StatisticalModel)
Return the number of independent observations on which the model was fitted. Be careful
when using this information, as the definition of an independent observation may vary
depending on the model, on the format used to pass the data, on the sampling plan
(if specified), etc.
"""
nobs(obj::StatisticalModel) = error("nobs is not defined for $(typeof(obj)).")
"""
    dof(obj::StatisticalModel)
Return the number of degrees of freedom consumed in the model, including
when applicable the intercept and the distribution's dispersion parameter.
"""
dof(obj::StatisticalModel) = error("dof is not defined for $(typeof(obj)).")
"""
    stderr(obj::StatisticalModel)
Return the standard errors for the coefficients of the model.
"""
stderr(obj::StatisticalModel) = sqrt.(diag(vcov(obj)))
"""
    vcov(obj::StatisticalModel)
Return the variance-covariance matrix for the coefficients of the model.
"""
vcov(obj::StatisticalModel) = error("vcov is not defined for $(typeof(obj)).")
"""
Fit a statistical model.
"""
fit(obj::StatisticalModel, args...) = error("fit is not defined for $(typeof(obj)).")
"""
Fit a statistical model in-place.
"""
fit!(obj::StatisticalModel, args...) = error("fit! is not defined for $(typeof(obj)).")
"""
    aic(obj::StatisticalModel)
Akaike's Information Criterion, defined as ``-2 \\log L + 2k``, with ``L`` the likelihood
of the model, and `k` its number of consumed degrees of freedom
(as returned by [`dof`](@ref)).
"""
aic(obj::StatisticalModel) = -2loglikelihood(obj) + 2dof(obj)
"""
    aicc(obj::StatisticalModel)
Corrected Akaike's Information Criterion for small sample sizes (Hurvich and Tsai 1989),
defined as ``-2 \\log L + 2k + 2k(k-1)/(n-k-1)``, with ``L`` the likelihood of the model,
``k`` its number of consumed degrees of freedom (as returned by [`dof`](@ref)),
and ``n`` the number of observations (as returned by [`nobs`](@ref)).
"""
function aicc(obj::StatisticalModel)
    k = dof(obj)
    n = nobs(obj)
    -2loglikelihood(obj) + 2k + 2k*(k+1)/(n-k-1)
end
"""
    bic(obj::StatisticalModel)
Bayesian Information Criterion, defined as ``-2 \\log L + k \\log n``, with ``L``
the likelihood of the model,  ``k`` its number of consumed degrees of freedom
(as returned by [`dof`](@ref)), and ``n`` the number of observations
(as returned by [`nobs`](@ref)).
"""
bic(obj::StatisticalModel) = -2loglikelihood(obj) + dof(obj)*log(nobs(obj))
"""
    r2(obj::StatisticalModel, variant::Symbol)
    r²(obj::StatisticalModel, variant::Symbol)
Coefficient of determination (R-squared).
For a linear model, the R² is defined as ``ESS/TSS``, with ``ESS`` the explained sum of squares
and ``TSS`` the total sum of squares, and `variant` can be omitted.
For other models, one of several pseudo R² definitions must be chosen via `variant`.
Supported variants are:
- `:MacFadden` (a.k.a. likelihood ratio index), defined as ``1 - \\log L/\\log L0``.
- `:CoxSnell`, defined as ``1 - (L0/L)^{2/n}``
- `:Nagelkerke`, defined as ``(1 - (L0/L)^{2/n})/(1 - L0^{2/n})``, with ``n`` the number
of observations (as returned by [`nobs`](@ref)).
In the above formulas, ``L`` is the likelihood of the model, ``L0`` that of the null model
(the model including only the intercept). These two quantities are taken to be minus half
`deviance` of the corresponding models.
"""
function r2(obj::StatisticalModel, variant::Symbol)
    ll = -deviance(obj)/2
    ll0 = -nulldeviance(obj)/2
    if variant == :McFadden
        1 - ll/ll0
    elseif variant == :CoxSnell
        1 - exp(2/nobs(obj) * (ll0 - ll))
    elseif variant == :Nagelkerke
        (1 - exp(2/nobs(obj) * (ll0 - ll)))/(1 - exp(2/nobs(obj) * ll0))
    else
        error("variant must be one of :McFadden, :CoxSnell or :Nagelkerke")
    end
end
const r² = r2
"""
    adjr2(obj::StatisticalModel, variant::Symbol)
    adjr²(obj::StatisticalModel, variant::Symbol)
Adjusted coefficient of determination (adjusted R-squared).
For linear models, the adjusted R² is defined as ``1 - (1 - (1-R^2)(n-1)/(n-p))``, with ``R^2``
the coefficient of determination, ``n`` the number of observations, and ``p`` the number of
coefficients (including the intercept). This definition is generally known as the Wherry Formula I.
For other models, one of the several pseudo R² definitions must be chosen via `variant`.
The only currently supported variant is `:MacFadden`, defined as ``1 - (\\log L - k)/\\log L0``.
In this formula, ``L`` is the likelihood of the model, ``L0`` that of the null model
(the model including only the intercept). These two quantities are taken to be minus half
`deviance` of the corresponding models. ``k`` is the number of consumed degrees of freedom
of the model (as returned by [`dof`](@ref)).
"""
function adjr2(obj::StatisticalModel, variant::Symbol)
    ll = -deviance(obj)/2
    ll0 = -nulldeviance(obj)/2
    k = dof(obj)
    if variant == :McFadden
        1 - (ll - k)/ll0
    else
        error(":McFadden is the only currently supported variant")
    end
end
const adjr² = adjr2
abstract type RegressionModel <: StatisticalModel end
"""
    fitted(obj::RegressionModel)
Return the fitted values of the model.
"""
fitted(obj::RegressionModel) = error("fitted is not defined for $(typeof(obj)).")
"""
    model_response(obj::RegressionModel)
Return the model response (a.k.a. the dependent variable).
"""
model_response(obj::RegressionModel) = error("model_response is not defined for $(typeof(obj)).")
"""
    modelmatrix(obj::RegressionModel)
Return the model matrix (a.k.a. the design matrix).
"""
modelmatrix(obj::RegressionModel) = error("modelmatrix is not defined for $(typeof(obj)).")
"""
    residuals(obj::RegressionModel)
Return the residuals of the model.
"""
residuals(obj::RegressionModel) = error("residuals is not defined for $(typeof(obj)).")
"""
    predict(obj::RegressionModel, [newX])
Form the predicted response of model `obj`. An object with new covariate values `newX` can be supplied,
which should have the same type and structure as that used to fit `obj`; e.g. for a GLM
it would generally be a `DataFrame` with the same variable names as the original predictors.
"""
function predict end
predict(obj::RegressionModel) = error("predict is not defined for $(typeof(obj)).")
"""
    predict!
In-place version of [`predict`](@ref).
"""
function predict! end
predict!(obj::RegressionModel) = error("predict! is not defined for $(typeof(obj)).")
"""
    dof_residual(obj::RegressionModel)
Return the residual degrees of freedom of the model.
"""
dof_residual(obj::RegressionModel) = error("dof_residual is not defined for $(typeof(obj)).")
"""
    params(obj)
Return all parameters of a model.
"""
params(obj) = error("params is not defined for $(typeof(obj))")
function params! end
mutable struct CoefTable
    cols::Vector
    colnms::Vector
    rownms::Vector
    function CoefTable(cols::Vector,colnms::Vector,rownms::Vector)
        nc = length(cols)
        nrs = map(length,cols)
        nr = nrs[1]
        length(colnms) in [0,nc] || error("colnms should have length 0 or $nc")
        length(rownms) in [0,nr] || error("rownms should have length 0 or $nr")
        all(nrs .== nr) || error("Elements of cols should have equal lengths, but got $nrs")
        new(cols,colnms,rownms)
    end
    function CoefTable(mat::Matrix,colnms::Vector,rownms::Vector,pvalcol::Int=0)
        nc = size(mat,2)
        cols = Any[mat[:, i] for i in 1:nc]
        if pvalcol != 0                         # format the p-values column
            cols[pvalcol] = [PValue(cols[pvalcol][j])
                            for j in eachindex(cols[pvalcol])]
        end
        CoefTable(cols,colnms,rownms)
    end
end
mutable struct PValue
    v::Number
    function PValue(v::Number)
        0. <= v <= 1. || isnan(v) || error("p-values must be in [0.,1.]")
        new(v)
    end
end
function show(io::IO, pv::PValue)
    v = pv.v
    if isnan(v)
        @printf(io,"%d", v)
    elseif v >= 1e-4
        @printf(io,"%.4f", v)
    else
        @printf(io,"<1e%2.2d", ceil(Integer, max(nextfloat(log10(v)), -99)))
    end
end
function show(io::IO, ct::CoefTable)
    cols = ct.cols; rownms = ct.rownms; colnms = ct.colnms;
    nc = length(cols)
    nr = length(cols[1])
    if length(rownms) == 0
        rownms = [lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    rnwidth = max(4,maximum([length(nm) for nm in rownms]) + 1)
    rownms = [rpad(nm,rnwidth) for nm in rownms]
    widths = [length(cn)::Int for cn in colnms]
    str = String[isa(cols[j][i], AbstractString) ? cols[j][i] :
        sprint(showcompact,cols[j][i]) for i in 1:nr, j in 1:nc]
    for j in 1:nc
        for i in 1:nr
            lij = length(str[i,j])
            if lij > widths[j]
                widths[j] = lij
            end
        end
    end
    widths .+= 1
    println(io," " ^ rnwidth *
            join([lpad(string(colnms[i]), widths[i]) for i = 1:nc], ""))
    for i = 1:nr
        print(io, rownms[i])
        for j in 1:nc
            print(io, lpad(str[i,j],widths[j]))
        end
        println(io)
    end
end
"""
    ConvergenceException(iters::Int, lastchange::Real=NaN, tol::Real=NaN)
The fitting procedure failed to converge in `iters` number of iterations,
i.e. the `lastchange` between the cost of the final and penultimate iteration was greater than
specified tolerance `tol`.
"""
struct ConvergenceException{T<:Real} <: Exception
    iters::Int
    lastchange::T
    tol::T
    function ConvergenceException{T}(iters, lastchange::T, tol::T) where T<:Real
        if tol > lastchange
            throw(ArgumentError("Change must be greater than tol."))
        else
            new(iters, lastchange, tol)
        end
    end
end
ConvergenceException(iters, lastchange::T=NaN, tol::T=NaN) where {T<:Real} =
    ConvergenceException{T}(iters, lastchange, tol)
function Base.showerror(io::IO, ce::ConvergenceException)
    print(io, "failure to converge after $(ce.iters) iterations.")
    if !isnan(ce.lastchange)
        print(io, " Last change ($(ce.lastchange)) was greater than tolerance ($(ce.tol)).")
    end
end
import Base.@deprecate
import Base.depwarn
import Base.@deprecate_binding
import Base.varm, Base.stdm
@deprecate varm(v::RealArray, m::Real, wv::AbstractWeights) varm(v, wv, m)
@deprecate varm(A::RealArray, M::RealArray, wv::AbstractWeights, dim::Int) varm(v, wv, m, dim)
@deprecate stdm(v::RealArray, m::Real, wv::AbstractWeights) stdm(v, wv, m)
@deprecate stdm(v::RealArray, m::RealArray, wv::AbstractWeights, dim::Int) stdm(v, wv, m, dim)
@deprecate trimmean(x::RealArray, p::Real) mean(trim(x, p/2))
@deprecate _moment2(v::RealArray, m::Real, wv::AbstractWeights) _moment2(v, wv, m)
@deprecate _moment3(v::RealArray, m::Real, wv::AbstractWeights) _moment3(v, wv, m)
@deprecate _moment4(v::RealArray, m::Real, wv::AbstractWeights) _moment4(v, wv, m)
@deprecate _momentk(v::RealArray, k::Int, m::Real, wv::AbstractWeights) _momentk(v, k, wv, m)
@deprecate moment(v::RealArray, k::Int, m::Real, wv::AbstractWeights) moment(v, k, wv, m)
@deprecate AIC(obj::StatisticalModel) aic(obj)
@deprecate AICc(obj::StatisticalModel) aicc(obj)
@deprecate BIC(obj::StatisticalModel) bic(obj)
@deprecate R2(obj::StatisticalModel, variant::Symbol) r2(obj, variant)
@deprecate R²(obj::StatisticalModel, variant::Symbol) r²(obj, variant)
@deprecate adjR2(obj::StatisticalModel, variant::Symbol) adjr2(obj, variant)
@deprecate adjR²(obj::StatisticalModel, variant::Symbol) adjr²(obj, variant)
function findat!(r::IntegerArray, a::AbstractArray{T}, b::AbstractArray{T}) where T
    Base.depwarn("findat! is deprecated, use indexin instead", :findat!)
    length(r) == length(b) || raise_dimerror()
    d = indexmap(a)
    @inbounds for i = 1 : length(b)
        r[i] = get(d, b[i], 0)
    end
    return r
end
"""
    findat(a, b)
For each element in `b`, find its first index in `a`. If the value does
not occur in `a`, the corresponding index is 0.
"""
findat(a::AbstractArray, b::AbstractArray) = findat!(Array{Int}(uninitialized, size(b)), a, b)
@deprecate df(obj::StatisticalModel) dof(obj)
@deprecate df_residual(obj::StatisticalModel) dof_residual(obj)
@deprecate_binding WeightVec Weights
struct RandIntSampler  # for generating Int samples in [0, K-1]
    a::Int
    Ku::UInt
    U::UInt
    function RandIntSampler(K::Int)
        Base.depwarn("RandIntSampler is deprecated, use Base.Random.RangeGenerator instead",
                     :RandIntSampler)
        Ku = UInt(K)
        new(1, Ku, div(typemax(UInt), Ku) * Ku)
    end
    function RandIntSampler(a::Int, b::Int)
        Base.depwarn("RandIntSampler is deprecated, use Base.Random.RangeGenerator instead",
                     :RandIntSampler)
        Ku = UInt(b-a+1)
        new(a, Ku, div(typemax(UInt), Ku) * Ku)
    end
end
function rand(rng::AbstractRNG, s::RandIntSampler)
    x = rand(rng, UInt)
    while x >= s.U
        x = rand(rng, UInt)
    end
    s.a + Int(rem(x, s.Ku))
end
rand(s::RandIntSampler) = rand(Base.GLOBAL_RNG, s)
@deprecate randi(rng::AbstractRNG, K::Int) rand(rng, 1:K)
@deprecate randi(K::Int) rand(1:K)
@deprecate randi(rng::AbstractRNG, a::Int, b::Int) rand(rng, a:b)
@deprecate randi(a::Int, b::Int) rand(a:b)
@deprecate(mad!(v::AbstractArray{T}, center;
                constant::Real = 1 / (-sqrt(2 * one(T)) * erfcinv(3 * one(T) / 2))) where T<:Real,
           mad!(v, center=center, constant=constant))
end # module
