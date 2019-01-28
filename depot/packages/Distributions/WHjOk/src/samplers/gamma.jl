struct GammaRmathSampler <: Sampleable{Univariate,Continuous}
    d::Gamma
end
rand(s::GammaRmathSampler) = StatsFuns.RFunctions.gammarand(shape(s.d), scale(s.d))
struct GammaGDSampler <: Sampleable{Univariate,Continuous}
    a::Float64
    s2::Float64
    s::Float64
    i2s::Float64
    d::Float64
    q0::Float64
    b::Float64
    σ::Float64
    c::Float64
    scale::Float64
end
function GammaGDSampler(g::Gamma)
    a = shape(g)
    s2 = a-0.5
    s = sqrt(s2)
    i2s = 0.5/s
    d = 5.656854249492381 - 12.0s # 4*sqrt(2) - 12s
    ia = 1.0/a
    q0 = ia*@horner(ia,
                    0.0416666664,
                    0.0208333723,
                    0.0079849875,
                    0.0015746717,
                    -0.0003349403,
                    0.0003340332,
                    0.0006053049,
                    -0.0004701849,
                    0.0001710320)
    if a <= 3.686
        b = 0.463 + s + 0.178s2
        σ = 1.235
        c = 0.195/s - 0.079 + 0.16s
    elseif a <= 13.022
        b = 1.654 + 0.0076s2
        σ = 1.68/s + 0.275
        c = 0.062/s + 0.024
    else
        b = 1.77
        σ = 0.75
        c = 0.1515/s
    end
    GammaGDSampler(a,s2,s,i2s,d,q0,b,σ,c,scale(g))
end
function rand(s::GammaGDSampler)
    t = randn()
    x = s.s + 0.5t
    t >= 0.0 && return x*x*s.scale
    u = rand()
    s.d*u <= t*t*t && return x*x*s.scale
    if x > 0.0
        v = t*s.i2s
        if abs(v) > 0.25
            q = s.q0 - s.s*t + 0.25*t*t + 2.0*s.s2*log1p(v)
        else
            q = s.q0 + 0.5*t*t*(v*@horner(v,
                                         0.333333333,
                                         -0.249999949,
                                         0.199999867,
                                         -0.1666774828,
                                         0.142873973,
                                         -0.124385581,
                                         0.110368310,
                                         -0.112750886,
                                         0.10408986))
        end
        log1p(-u) <= q && return x*x*s.scale
    end
    @label step8
    e = randexp()
    u = 2.0rand() - 1.0
    t = s.b + e*s.σ*sign(u)
    t < -0.718_744_837_717_19 && @goto step8
    v = t*s.i2s
    if abs(v) > 0.25
        q = s.q0 - s.s*t + 0.25*t*t + 2.0*s.s2*log1p(v)
    else
        q = s.q0 + 0.5*t*t*(v*@horner(v,
                                      0.333333333,
                                      -0.249999949,
                                      0.199999867,
                                      -0.1666774828,
                                      0.142873973,
                                      -0.124385581,
                                      0.110368310,
                                      -0.112750886,
                                      0.10408986))
    end
    (q <= 0.0 || s.c*abs(u) > expm1(q)*exp(e-0.5t*t)) && @goto step8
    x = s.s+0.5t
    return x*x*s.scale
end
struct GammaGSSampler <: Sampleable{Univariate,Continuous}
    a::Float64
    ia::Float64
    b::Float64
    scale::Float64
end
function GammaGSSampler(d::Gamma)
    a = shape(d)
    ia = 1.0 / a
    b = 1.0+0.36787944117144233 * a
    GammaGSSampler(a, ia, b, scale(d))
end
function rand(s::GammaGSSampler)
    while true
        p = s.b*rand()
        e = randexp()
        if p <= 1.0
            x = exp(log(p)*s.ia)
            e < x || return s.scale*x
        else
            x = -log(s.ia*(s.b-p))
            e < log(x)*(1.0-s.a) || return s.scale*x
        end
    end
end
struct GammaMTSampler <: Sampleable{Univariate,Continuous}
    d::Float64
    c::Float64
    κ::Float64
end
function GammaMTSampler(g::Gamma)
    d = shape(g) - 1/3
    c = 1.0 / sqrt(9.0 * d)
    κ = d * scale(g)
    GammaMTSampler(d, c, κ)
end
function rand(s::GammaMTSampler)
    while true
        x = randn()
        v = 1.0 + s.c * x
        while v <= 0.0
            x = randn()
            v = 1.0 + s.c * x
        end
        v *= (v * v)
        u = rand()
        x2 = x * x
        if u < 1.0 - 0.331 * abs2(x2) || log(u) < 0.5 * x2 + s.d * (1.0 - v + log(v))
            return v*s.κ
        end
    end
end
struct GammaIPSampler{S<:Sampleable{Univariate,Continuous}} <: Sampleable{Univariate,Continuous}
    s::S #sampler for Gamma(1+shape,scale)
    nia::Float64 #-1/scale
end
function GammaIPSampler(d::Gamma,::Type{S}) where S<:Sampleable
    GammaIPSampler(Gamma(1.0 + shape(d), scale(d)), -1.0 / shape(d))
end
GammaIPSampler(d::Gamma) = GammaIPSampler(d,GammaMTSampler)
function rand(s::GammaIPSampler)
    x = rand(s.s)
    e = randexp()
    x*exp(s.nia*e)
end
