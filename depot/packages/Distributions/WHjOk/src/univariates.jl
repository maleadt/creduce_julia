struct RealInterval
    lb::Float64
    ub::Float64
    RealInterval(lb::Real, ub::Real) = new(Float64(lb), Float64(ub))
end
minimum(r::RealInterval) = r.lb
maximum(r::RealInterval) = r.ub
in(x::Real, r::RealInterval) = (r.lb <= Float64(x) <= r.ub)
isbounded(d::Union{D,Type{D}}) where {D<:UnivariateDistribution} = isupperbounded(d) && islowerbounded(d)
islowerbounded(d::Union{D,Type{D}}) where {D<:UnivariateDistribution} = minimum(d) > -Inf
isupperbounded(d::Union{D,Type{D}}) where {D<:UnivariateDistribution} = maximum(d) < +Inf
hasfinitesupport(d::Union{D,Type{D}}) where {D<:DiscreteUnivariateDistribution} = isbounded(d)
hasfinitesupport(d::Union{D,Type{D}}) where {D<:ContinuousUnivariateDistribution} = false
""" """ params(d::UnivariateDistribution)
""" """ scale(d::UnivariateDistribution)
""" """ location(d::UnivariateDistribution)
""" """ shape(d::UnivariateDistribution)
""" """ rate(d::UnivariateDistribution)
""" """ ncategories(d::UnivariateDistribution)
""" """ ntrials(d::UnivariateDistribution)
""" """ dof(d::UnivariateDistribution)
""" """ minimum(d::UnivariateDistribution)
""" """ maximum(d::UnivariateDistribution)
""" """ insupport{D<:UnivariateDistribution}(d::Union{D, Type{D}}, x::Any)
function insupport!(r::AbstractArray, d::Union{D,Type{D}}, X::AbstractArray) where D<:UnivariateDistribution
    length(r) == length(X) ||
        throw(DimensionMismatch("Inconsistent array dimensions."))
    for i in 1 : length(X)
        @inbounds r[i] = insupport(d, X[i])
    end
    return r
end
insupport(d::Union{D,Type{D}}, X::AbstractArray) where {D<:UnivariateDistribution} =
     insupport!(BitArray(undef, size(X)), d, X)
insupport(d::Union{D,Type{D}},x::Real) where {D<:ContinuousUnivariateDistribution} = minimum(d) <= x <= maximum(d)
insupport(d::Union{D,Type{D}},x::Real) where {D<:DiscreteUnivariateDistribution} = isinteger(x) && minimum(d) <= x <= maximum(d)
support(d::Union{D,Type{D}}) where {D<:ContinuousUnivariateDistribution} = RealInterval(minimum(d), maximum(d))
support(d::Union{D,Type{D}}) where {D<:DiscreteUnivariateDistribution} = round(Int, minimum(d)):round(Int, maximum(d))
struct FiniteSupport{T} end
macro distr_support(D, lb, ub)
    D_has_constantbounds = (isa(ub, Number) || ub == :Inf) &&
                           (isa(lb, Number) || lb == :(-Inf))
    paramdecl = D_has_constantbounds ? :(d::Union{$D, Type{$D}}) : :(d::$D)
    esc(quote
        minimum($(paramdecl)) = $lb
        maximum($(paramdecl)) = $ub
    end)
end
""" """ rand(d::UnivariateDistribution) = quantile(d, rand())
""" """ rand!(d::UnivariateDistribution, A::AbstractArray) = _rand!(sampler(d), A)
rand(d::UnivariateDistribution, n::Int) = _rand!(sampler(d), Vector{eltype(d)}(undef, n))
rand(d::UnivariateDistribution, shp::Dims) = _rand!(sampler(d), Vector{eltype(d)}(undef, shp))
sampler(d::UnivariateDistribution) = d
""" """ mean(d::UnivariateDistribution)
""" """ var(d::UnivariateDistribution)
""" """ std(d::UnivariateDistribution) = sqrt(var(d))
""" """ median(d::UnivariateDistribution) = quantile(d, 0.5)
""" """ modes(d::UnivariateDistribution) = [mode(d)]
""" """ mode(d::UnivariateDistribution)
""" """ skewness(d::UnivariateDistribution)
""" """ entropy(d::UnivariateDistribution)
""" """ entropy(d::UnivariateDistribution, b::Real) = entropy(d) / log(b)
""" """ isplatykurtic(d::UnivariateDistribution) = kurtosis(d) > 0.0
""" """ isleptokurtic(d::UnivariateDistribution) = kurtosis(d) < 0.0
""" """ ismesokurtic(d::UnivariateDistribution) = kurtosis(d) ≈ 0.0
""" """ kurtosis(d::UnivariateDistribution)
""" """ function kurtosis(d::Distribution, correction::Bool)
    if correction
        return kurtosis(d)
    else
        return kurtosis(d) + 3.0
    end
end
excess(d::Distribution) = kurtosis(d)
excess_kurtosis(d::Distribution) = kurtosis(d)
proper_kurtosis(d::Distribution) = kurtosis(d, false)
""" """ mgf(d::UnivariateDistribution, t)
""" """ cf(d::UnivariateDistribution, t)
""" """ pdf(d::UnivariateDistribution, x::Real)
pdf(d::DiscreteUnivariateDistribution, x::Int) = throw(MethodError(pdf, (d, x)))
pdf(d::DiscreteUnivariateDistribution, x::Integer) = pdf(d, round(Int, x))
pdf(d::DiscreteUnivariateDistribution, x::Real) = isinteger(x) ? pdf(d, round(Int, x)) : 0.0
""" """ logpdf(d::UnivariateDistribution, x::Real) = log(pdf(d, x))
logpdf(d::DiscreteUnivariateDistribution, x::Int) = log(pdf(d, x))
logpdf(d::DiscreteUnivariateDistribution, x::Integer) = logpdf(d, round(Int, x))
logpdf(d::DiscreteUnivariateDistribution, x::Real) = isinteger(x) ? logpdf(d, round(Int, x)) : -Inf
""" """ cdf(d::UnivariateDistribution, x::Real)
cdf(d::DiscreteUnivariateDistribution, x::Int) = cdf(d, x, FiniteSupport{hasfinitesupport(d)})
function cdf(d::DiscreteUnivariateDistribution, x::Int, ::Type{FiniteSupport{false}})
    c = 0.0
    for y = minimum(d):x
        c += pdf(d, y)
    end
    return c
end
function cdf(d::DiscreteUnivariateDistribution, x::Int, ::Type{FiniteSupport{true}})
    x <= div(minimum(d) + maximum(d),2) && return cdf(d, x, FiniteSupport{false})
    c = 1.0
    for y = x+1:maximum(d)
        c -= pdf(d, y)
    end
    return c
end
cdf(d::DiscreteUnivariateDistribution, x::Real) = cdf(d, floor(Int,x))
cdf(d::ContinuousUnivariateDistribution, x::Real) = throw(MethodError(cdf, (d, x)))
""" """ ccdf(d::UnivariateDistribution, x::Real) = 1.0 - cdf(d, x)
ccdf(d::DiscreteUnivariateDistribution, x::Int) = 1.0 - cdf(d, x)
ccdf(d::DiscreteUnivariateDistribution, x::Real) = ccdf(d, floor(Int,x))
""" """ logcdf(d::UnivariateDistribution, x::Real) = log(cdf(d, x))
logcdf(d::DiscreteUnivariateDistribution, x::Int) = log(cdf(d, x))
logcdf(d::DiscreteUnivariateDistribution, x::Real) = logcdf(d, floor(Int,x))
""" """ logccdf(d::UnivariateDistribution, x::Real) = log(ccdf(d, x))
logccdf(d::DiscreteUnivariateDistribution, x::Int) = log(ccdf(d, x))
logccdf(d::DiscreteUnivariateDistribution, x::Real) = logccdf(d, floor(Int,x))
""" """ quantile(d::UnivariateDistribution, p::Real)
""" """ cquantile(d::UnivariateDistribution, p::Real) = quantile(d, 1.0 - p)
""" """ invlogcdf(d::UnivariateDistribution, lp::Real) = quantile(d, exp(lp))
""" """ invlogccdf(d::UnivariateDistribution, lp::Real) = quantile(d, -expm1(lp))
gradlogpdf(d::ContinuousUnivariateDistribution, x::Real) = throw(MethodError(gradlogpdf, (d, x)))
function _pdf_fill_outside!(r::AbstractArray, d::DiscreteUnivariateDistribution, X::UnitRange)
    vl = vfirst = first(X)
    vr = vlast = last(X)
    n = vlast - vfirst + 1
    if islowerbounded(d)
        lb = minimum(d)
        if vl < lb
            vl = lb
        end
    end
    if isupperbounded(d)
        ub = maximum(d)
        if vr > ub
            vr = ub
        end
    end
    if vl > vfirst
        for i = 1:(vl - vfirst)
            r[i] = 0.0
        end
    end
    fm1 = vfirst - 1
    for v = vl:vr
        r[v - fm1] = pdf(d, v)
    end
    if vr < vlast
        for i = (vr-vfirst+2):n
            r[i] = 0.0
        end
    end
    return vl, vr, vfirst, vlast
end
function _pdf!(r::AbstractArray, d::DiscreteUnivariateDistribution, X::UnitRange)
    vl,vr, vfirst, vlast = _pdf_fill_outside!(r, d, X)
    fm1 = vfirst - 1
    for v = vl:vr
        r[v - fm1] = pdf(d, v)
    end
    return r
end
abstract type RecursiveProbabilityEvaluator end
function _pdf!(r::AbstractArray, d::DiscreteUnivariateDistribution, X::UnitRange, rpe::RecursiveProbabilityEvaluator)
    vl,vr, vfirst, vlast = _pdf_fill_outside!(r, d, X)
    if vl <= vr
        fm1 = vfirst - 1
        r[vl - fm1] = pv = pdf(d, vl)
        for v = (vl+1):vr
            r[v - fm1] = pv = nextpdf(rpe, pv, v)
        end
    end
    return r
end
""" """ loglikelihood(d::UnivariateDistribution, X::AbstractArray) = sum(x -> logpdf(d, x), X)
macro _delegate_statsfuns(D, fpre, psyms...)
    dt = eval(D)
    T = dt <: DiscreteUnivariateDistribution ? :Int : :Real
    fpdf = Symbol(fpre, "pdf")
    flogpdf = Symbol(fpre, "logpdf")
    fcdf = Symbol(fpre, "cdf")
    fccdf = Symbol(fpre, "ccdf")
    flogcdf = Symbol(fpre, "logcdf")
    flogccdf = Symbol(fpre, "logccdf")
    finvcdf = Symbol(fpre, "invcdf")
    finvccdf = Symbol(fpre, "invccdf")
    finvlogcdf = Symbol(fpre, "invlogcdf")
    finvlogccdf = Symbol(fpre, "invlogccdf")
    pargs = [Expr(:(.), :d, Expr(:quote, s)) for s in psyms]
    esc(quote
        pdf(d::$D, x::$T) = $(fpdf)($(pargs...), x)
        logpdf(d::$D, x::$T) = $(flogpdf)($(pargs...), x)
        cdf(d::$D, x::$T) = $(fcdf)($(pargs...), x)
        ccdf(d::$D, x::$T) = $(fccdf)($(pargs...), x)
        logcdf(d::$D, x::$T) = $(flogcdf)($(pargs...), x)
        logccdf(d::$D, x::$T) = $(flogccdf)($(pargs...), x)
        quantile(d::$D, q::Real) = convert($T, $(finvcdf)($(pargs...), q))
        cquantile(d::$D, q::Real) = convert($T, $(finvccdf)($(pargs...), q))
        invlogcdf(d::$D, lq::Real) = convert($T, $(finvlogcdf)($(pargs...), lq))
        invlogccdf(d::$D, lq::Real) = convert($T, $(finvlogccdf)($(pargs...), lq))
    end)
end
const discrete_distributions = [
    "bernoulli",
    "betabinomial",
    "binomial",
    "categorical",
    "discreteuniform",
    "geometric",
    "hypergeometric",
    "negativebinomial",
    "noncentralhypergeometric",
    "poisson",
    "skellam",
    "poissonbinomial"
]
const continuous_distributions = [
    "arcsine",
    "beta",
    "betaprime",
    "biweight",
    "cauchy",
    "chisq",    # Chi depends on Chisq
    "chi",
    "cosine",
    "epanechnikov",
    "exponential",
    "fdist",
    "frechet",
    "gamma", "erlang",
    "generalizedpareto",
    "generalizedextremevalue",
    "gumbel",
    "inversegamma",
    "inversegaussian",
    "kolmogorov",
    "ksdist",
    "ksonesided",
    "laplace",
    "levy",
    "locationscale",
    "logistic",
    "noncentralbeta",
    "noncentralchisq",
    "noncentralf",
    "noncentralt",
    "normal",
    "normalcanon",
    "normalinversegaussian",
    "lognormal",    # LogNormal depends on Normal
    "pareto",
    "rayleigh",
    "semicircle",
    "symtriangular",
    "tdist",
    "triangular",
    "triweight",
    "uniform",
    "vonmises",
    "weibull"
]
for dname in discrete_distributions
    include(joinpath("univariate", "discrete", "$(dname).jl"))
end
for dname in continuous_distributions
    include(joinpath("univariate", "continuous", "$(dname).jl"))
end
