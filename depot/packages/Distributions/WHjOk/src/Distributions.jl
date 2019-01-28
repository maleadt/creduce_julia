module Distributions
using StatsBase, PDMats, StatsFuns, Statistics
import QuadGK: quadgk
import Base: size, eltype, length, convert, show, getindex, rand
import Base: sum, maximum, minimum, +, -
import Base.Math: @horner
using LinearAlgebra, Printf
using Random
import Random: GLOBAL_RNG, RangeGenerator, rand!, SamplerRangeInt
import Statistics: mean, median, quantile, std, var, cov, cor
import StatsBase: kurtosis, skewness, entropy, mode, modes,
                  fit, kldivergence, loglikelihood, dof, span,
                  params, params!
import PDMats: dim, PDMat, invquad
using SpecialFunctions
export
    mean, median, quantile, std, var, cov, cor,
    VariateForm,
    ValueSupport,
    Univariate,
    Multivariate,
    Matrixvariate,
    Discrete,
    Continuous,
    Sampleable,
    Distribution,
    UnivariateDistribution,
    MultivariateDistribution,
    MatrixDistribution,
    NoncentralHypergeometric,
    NonMatrixDistribution,
    DiscreteDistribution,
    ContinuousDistribution,
    DiscreteUnivariateDistribution,
    DiscreteMultivariateDistribution,
    DiscreteMatrixDistribution,
    ContinuousUnivariateDistribution,
    ContinuousMultivariateDistribution,
    ContinuousMatrixDistribution,
    SufficientStats,
    AbstractMvNormal,
    AbstractMixtureModel,
    UnivariateMixture,
    MultivariateMixture,
    Arcsine,
    Bernoulli,
    Beta,
    BetaBinomial,
    BetaPrime,
    Binomial,
    Biweight,
    Categorical,
    Cauchy,
    Chi,
    Chisq,
    Cosine,
    DiagNormal,
    DiagNormalCanon,
    Dirichlet,
    DirichletMultinomial,
    DiscreteUniform,
    DoubleExponential,
    EdgeworthMean,
    EdgeworthSum,
    EdgeworthZ,
    EmpiricalUnivariateDistribution,
    Erlang,
    Epanechnikov,
    Exponential,
    FDist,
    FisherNoncentralHypergeometric,
    Frechet,
    FullNormal,
    FullNormalCanon,
    Gamma,
    GeneralizedPareto,
    GeneralizedExtremeValue,
    Geometric,
    Gumbel,
    Hypergeometric,
    InverseWishart,
    InverseGamma,
    InverseGaussian,
    IsoNormal,
    IsoNormalCanon,
    Kolmogorov,
    KSDist,
    KSOneSided,
    Laplace,
    Levy,
    LocationScale,
    Logistic,
    LogNormal,
    MixtureModel,
    Multinomial,
    MultivariateNormal,
    MvLogNormal,
    MvNormal,
    MvNormalCanon,
    MvNormalKnownCov,
    MvTDist,
    NegativeBinomial,
    NoncentralBeta,
    NoncentralChisq,
    NoncentralF,
    NoncentralHypergeometric,
    NoncentralT,
    Normal,
    NormalCanon,
    NormalInverseGaussian,
    Pareto,
    Poisson,
    PoissonBinomial,
    QQPair,
    Rayleigh,
    Semicircle,
    Skellam,
    SymTriangularDist,
    TDist,
    TriangularDist,
    Triweight,
    Truncated,
    TruncatedNormal,
    Uniform,
    UnivariateGMM,
    VonMises,
    VonMisesFisher,
    WalleniusNoncentralHypergeometric,
    Weibull,
    Wishart,
    ZeroMeanIsoNormal,
    ZeroMeanIsoNormalCanon,
    ZeroMeanDiagNormal,
    ZeroMeanDiagNormalCanon,
    ZeroMeanFullNormal,
    ZeroMeanFullNormalCanon,
    RealInterval,
    binaryentropy,      # entropy of distribution in bits
    canonform,          # get canonical form of a distribution
    ccdf,               # complementary cdf, i.e. 1 - cdf
    cdf,                # cumulative distribution function
    cf,                 # characteristic function
    cgf,                # cumulant generating function
    cquantile,          # complementary quantile (i.e. using prob in right hand tail)
    cumulant,           # cumulants of distribution
    component,          # get the k-th component of a mixture model
    components,         # get components from a mixture model
    componentwise_pdf,      # component-wise pdf for mixture models
    componentwise_logpdf,   # component-wise logpdf for mixture models
    concentration,      # the concentration parameter
    dim,                # sample dimension of multivariate distribution
    dof,                # get the degree of freedom
    entropy,            # entropy of distribution in nats
    failprob,           # failing probability
    fit,                # fit a distribution to data (using default method)
    fit_mle,            # fit a distribution to data using MLE
    fit_mle!,           # fit a distribution to data using MLE (inplace update to initial guess)
    fit_map,            # fit a distribution to data using MAP
    fit_map!,           # fit a distribution to data using MAP (inplace update to initial guess)
    freecumulant,       # free cumulants of distribution
    insupport,          # predicate, is x in the support of the distribution?
    invcov,             # get the inversed covariance
    invlogccdf,         # complementary quantile based on log probability
    invlogcdf,          # quantile based on log probability
    isplatykurtic,      # Is excess kurtosis > 0.0?
    isleptokurtic,      # Is excess kurtosis < 0.0?
    ismesokurtic,       # Is excess kurtosis = 0.0?
    isprobvec,          # Is a probability vector?
    isupperbounded,
    islowerbounded,
    isbounded,
    hasfinitesupport,
    kde,                # Kernel density estimator (from Stats.jl)
    kurtosis,           # kurtosis of the distribution
    logccdf,            # ccdf returning log-probability
    logcdf,             # cdf returning log-probability
    logdetcov,          # log-determinant of covariance
    loglikelihood,      # log probability of array of IID draws
    logpdf,             # log probability density
    logpdf!,            # evaluate log pdf to provided storage
    invscale,           # Inverse scale parameter
    sqmahal,            # squared Mahalanobis distance to Gaussian center
    sqmahal!,           # inplace evaluation of sqmahal
    location,           # get the location parameter
    location!,          # provide storage for the location parameter (used in multivariate distribution mvlognormal)
    mean,               # mean of distribution
    meandir,            # mean direction (of a spherical distribution)
    meanform,           # convert a normal distribution from canonical form to mean form
    meanlogx,           # the mean of log(x)
    median,             # median of distribution
    mgf,                # moment generating function
    mode,               # the mode of a unimodal distribution
    modes,              # mode(s) of distribution as vector
    moment,             # moments of distribution
    nsamples,           # get the number of samples contained in an array
    ncategories,        # the number of categories in a Categorical distribution
    ncomponents,        # the number of components in a mixture model
    ntrials,            # the number of trials being performed in the experiment
    params,             # get the tuple of parameters
    params!,            # provide storage space to calculate the tuple of parameters for a multivariate distribution like mvlognormal
    partype,            # returns a type large enough to hold all of a distribution's parameters' element types
    pdf,                # probability density function (ContinuousDistribution)
    probs,              # Get the vector of probabilities
    probval,            # The pdf/pmf value for a uniform distribution
    quantile,           # inverse of cdf (defined for p in (0,1))
    qqbuild,            # build a paired quantiles data structure for qqplots
    rate,               # get the rate parameter
    sampler,            # create a Sampler object for efficient samples
    scale,              # get the scale parameter
    scale!,             # provide storage for the scale parameter (used in multivariate distribution mvlognormal)
    shape,              # get the shape parameter
    skewness,           # skewness of the distribution
    span,               # the span of the support, e.g. maximum(d) - minimum(d)
    std,                # standard deviation of distribution
    stdlogx,            # standard deviation of log(x)
    suffstats,          # compute sufficient statistics
    succprob,           # the success probability
    support,            # the support of a distribution (or a distribution type)
    test_samples,       # test a sampler
    test_distr,         # test a distribution
    var,                # variance of distribution
    varlogx,            # variance of log(x)
    expected_logdet,    # expected logarithm of random matrix determinant
    gradlogpdf,         # gradient (or derivative) of logpdf(d,x) wrt x
    sample, sample!,        # sample from a source array
    wsample, wsample!      # weighted sampling from a source array
include("common.jl")
include("utils.jl")
include("show.jl")
include("quantilealgs.jl")
include("genericrand.jl")
include("functionals.jl")
include("genericfit.jl")
include("univariates.jl")
include("empirical.jl")
include("edgeworth.jl")
include("multivariates.jl")
include("matrixvariates.jl")
include("samplers.jl")
include("truncate.jl")
include("conversion.jl")
include("qq.jl")
include("estimators.jl")
include("testutils.jl")
include("mixtures/mixturemodel.jl")
include("mixtures/unigmm.jl")
include("deprecates.jl")
""" """ Distributions
end # module
