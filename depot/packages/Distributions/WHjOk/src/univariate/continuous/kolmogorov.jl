"""
    Kolmogorov()
Kolmogorov distribution defined as
```math
\\sup_{t \\in [0,1]} |B(t)|
```
where ``B(t)`` is a Brownian bridge used in the Kolmogorov--Smirnov
test for large n.
"""
struct Kolmogorov <: ContinuousUnivariateDistribution
end
@distr_support Kolmogorov 0.0 Inf
params(d::Kolmogorov) = ()
mean(d::Kolmogorov) = sqrt2π*log(2)/2
var(d::Kolmogorov) = pi^2/12 - pi*log(2)^2/2
mode(d::Kolmogorov) = 0.735467907916572
median(d::Kolmogorov) = 0.8275735551899077
function cdf_raw(d::Kolmogorov, x::Real)
    a = -(pi*pi)/(x*x)
    f = exp(a)
    f2 = f*f
    u = (1 + f*(1 + f2))
    sqrt2π*exp(a/8)*u/x
end
function ccdf_raw(d::Kolmogorov, x::Real)
    f = exp(-2*x*x)
    f2 = f*f
    f3 = f2*f
    f5 = f2*f3
    f7 = f2*f5
    u = (1 - f3*(1 - f5*(1 - f7)))
    2f*u
end
function cdf(d::Kolmogorov,x::Real)
    if x <= 0
        0
    elseif x <= 1
        cdf_raw(d,x)
    else
        1-ccdf_raw(d,x)
    end
end
function ccdf(d::Kolmogorov,x::Real)
    if x <= 0
        1
    elseif x <= 1
        1-cdf_raw(d,x)
    else
        ccdf_raw(d,x)
    end
end
function pdf(d::Kolmogorov,x::Real)
    if x <= 0
        return 0.0
    elseif x <= 1
        c = π/(2*x)
        s = 0.0
        for i = 1:20
            k = ((2i - 1)*c)^2
            s += (k - 1)*exp(-k/2)
        end
        return sqrt2π*s/x^2
    else
        s = 0.0
        for i = 1:20
            s += (iseven(i) ? -1 : 1)*i^2*exp(-2(i*x)^2)
        end
        return 8*x*s
    end
end
@quantile_newton Kolmogorov
rand(d::Kolmogorov) = rand(GLOBAL_RNG, d)
function rand(rng::AbstractRNG, d::Kolmogorov)
    t = 0.75
    if rand(rng) < 0.3728329582237386 # cdf(d,t)
        while true
            g = rand_trunc_gamma(rng)
            x = pi/sqrt(8g)
            w = 0.0
            z = 1/(2g)
            p = exp(-g)
            n = 1
            q = 1.0
            u = rand(rng)
            while u >= w
                w += z*q
                if u >= w
                    return x
                end
                n += 2
                nsq = n*n
                q = p^(nsq-1)
                w -= nsq*q
            end
        end
    else
        while true
            e = randexp(rng)
            u = rand(rng)
            x = sqrt(t*t+e/2)
            w = 0.0
            n = 1
            z = exp(-2*x*x)
            while u > w
                n += 1
                w += n*n*z^(n*n-1)
                if u >= w
                    return x
                end
                n += 1
                w -= n*n*z^(n*n-1)
            end
        end
    end
end
function rand_trunc_gamma(rng::AbstractRNG)
    tp = 2.193245422464302 #pi^2/(8*t^2)
    while true
        e0 = rand(rng, Exponential(1.2952909208355123))
        e1 = rand(rng, Exponential(2))
        g = tp + e0
        if (e0*e0 <= tp*e1*(g+tp)) || (g/tp - 1 - log(g/tp) <= e1)
            return g
        end
    end
end
