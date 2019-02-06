""" """ struct PoissonBinomial{T<:Real} <: DiscreteUnivariateDistribution
    p::Vector{T}
    pmf::Vector{T}
    function PoissonBinomial{T}(p::AbstractArray) where T
        for i=1:length(p)
            if !(0 <= p[i] <= 1)
                error("Each element of p must be in [0, 1].")
            end
        end
        pb = poissonbinomial_pdf_fft(p)
        @assert isprobvec(pb)
        new{T}(p, pb)
    end
end
PoissonBinomial(p::AbstractArray{T}) where {T<:Real} = PoissonBinomial{T}(p)
@distr_support PoissonBinomial 0 length(d.p)
function PoissonBinomial(::Type{PoissonBinomial{T}}, p::Vector{S}) where {T <: Real, S <: Real}
    PoissonBinomial(Vector{T}(p))
end
function PoissonBinomial(::Type{PoissonBinomial{T}}, d::PoissonBinomial{S}) where {T <: Real, S <: Real}
    PoissonBinomial(Vector{T}(d.p))
end
ntrials(d::PoissonBinomial) = length(d.p)
succprob(d::PoissonBinomial) = d.p
failprob(d::PoissonBinomial) = 1 .- d.p
params(d::PoissonBinomial) = (d.p, )
@inline partype(d::PoissonBinomial{T}) where {T<:Real} = T
mean(d::PoissonBinomial) = sum(succprob(d))
var(d::PoissonBinomial) = sum(succprob(d) .* failprob(d))
function skewness(d::PoissonBinomial{T}) where T<:Real
    v = zero(T)
    s = zero(T)
    p,  = params(d)
    for i=1:length(p)
        v += p[i] * (1 - p[i])
        s += p[i] * (1 - p[i]) * (1 - 2 * p[i])
    end
    s / sqrt(v) / v
end
function kurtosis(d::PoissonBinomial{T}) where T<:Real
    v = zero(T)
    s = zero(T)
    p,  = params(d)
    for i=1:length(p)
        v += p[i] * (1 - p[i])
        s += p[i] * (1 - p[i]) * (1 - 6 * (1 - p[i] ) * p[i])
    end
    s / v / v
end
entropy(d::PoissonBinomial) = entropy(Categorical(d.pmf))
median(d::PoissonBinomial) = median(Categorical(d.pmf)) - 1
mode(d::PoissonBinomial) = argmax(d.pmf) - 1
modes(d::PoissonBinomial) = [x  - 1 for x in modes(Categorical(d.pmf))]
quantile(d::PoissonBinomial, x::Float64) = quantile(Categorical(d.pmf), x) - 1
function mgf(d::PoissonBinomial, t::Real)
    p,  = params(d)
    prod(1 .- p .+ p .* exp(t))
end
function cf(d::PoissonBinomial, t::Real)
    p,  = params(d)
    prod(1 .- p .+ p .* cis(t))
end
pdf(d::PoissonBinomial, k::Int) = insupport(d, k) ? d.pmf[k+1] : 0
function logpdf(d::PoissonBinomial{T}, k::Int) where T<:Real
    insupport(d, k) ? log(d.pmf[k + 1]) : -T(Inf)
end
function poissonbinomial_pdf_fft(p::AbstractArray)
    n = length(p)
    ω = 2 / (n + 1)
    x = Vector{Complex{Float64}}(undef, n+1)
    lmax = ceil(Int, n/2)
    x[1] = 1/(n + 1)
    for l=1:lmax
        logz = 0.
        argz = 0.
        for j=1:n
            zjl = 1 - p[j] + p[j] * cospi(ω*l) + im * p[j] * sinpi(ω * l)
            logz += log(abs(zjl))
            argz += atan(imag(zjl), real(zjl))
        end
        dl = exp(logz)
        x[l + 1] = dl * cos(argz) / (n + 1) + dl * sin(argz) * im / (n + 1)
        if n + 1 - l > l
            x[n + 1 - l + 1] = conj(x[l + 1])
        end
    end
    [max(0, real(xi)) for xi in _dft(x)]
end
function _dft(x::Vector{T}) where T
    n = length(x)
    y = zeros(complex(float(T)), n)
    @inbounds for j = 0:n-1, k = 0:n-1
        y[k+1] += x[j+1] * cis(-π * float(T)(2 * mod(j * k, n)) / n)
    end
    return y
end
sampler(d::PoissonBinomial) = PoissBinAliasSampler(d)