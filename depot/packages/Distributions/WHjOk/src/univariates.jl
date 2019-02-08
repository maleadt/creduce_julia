struct FiniteSupport{T} end
macro distr_support(D, lb, ub)
    D_has_constantbounds = (isa(ub, Number) || ub == :Inf) &&
    esc(quote
    end)
end
function cdf(d::DiscreteUnivariateDistribution, x::Int, ::Type{FiniteSupport{false}})
    for y = minimum(d):x
    end
    if isupperbounded(d)
        if vr > ub
        end
    end
end
abstract type RecursiveProbabilityEvaluator end
function _pdf!(r::AbstractArray, d::DiscreteUnivariateDistribution, X::UnitRange, rpe::RecursiveProbabilityEvaluator)
    if vl <= vr
        fm1 = vfirst - 1
        for v = (vl+1):vr
        end
    end
end
macro _delegate_statsfuns(D, fpre, psyms...)
    dt = eval(D)
    T = dt <: DiscreteUnivariateDistribution ? :Int : :Real
    fpdf = Symbol(fpre, "pdf")
    pargs = [Expr(:(.), :d, Expr(:quote, s)) for s in psyms]
    esc(quote
        pdf(d::$D, x::$T) = $(fpdf)($(pargs...), x)
    end)
end
const discrete_distributions = [
    "bernoulli",
    "binomial",
    "categorical",
    "poissonbinomial"
]
const continuous_distributions = [
    "exponential",
    "gamma", "erlang",
    "normal",
]
for dname in discrete_distributions
    include(joinpath("univariate", "discrete", "$(dname).jl"))
end
for dname in continuous_distributions
    include(joinpath("univariate", "continuous", "$(dname).jl"))
end
