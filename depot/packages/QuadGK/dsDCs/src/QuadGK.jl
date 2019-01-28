VERSION < v"0.7.0-beta2.199" && __precompile__()
module QuadGK
export quadgk, gauss, kronrod
using DataStructures, LinearAlgebra
import Base.Order.Reverse
const rulecache = Dict{Any,Any}( (Float64,7) => # precomputed in 100-bit arith.
  ([-9.9145537112081263920685469752598e-01,
    -9.4910791234275852452618968404809e-01,
    -8.6486442335976907278971278864098e-01,
    -7.415311855993944398638647732811e-01,
    -5.8608723546769113029414483825842e-01,
    -4.0584515137739716690660641207707e-01,
    -2.0778495500789846760068940377309e-01,
    0.0],
   [2.2935322010529224963732008059913e-02,
    6.3092092629978553290700663189093e-02,
    1.0479001032225018383987632254189e-01,
    1.4065325971552591874518959051021e-01,
    1.6900472663926790282658342659795e-01,
    1.9035057806478540991325640242055e-01,
    2.0443294007529889241416199923466e-01,
    2.0948214108472782801299917489173e-01],
    [1.2948496616886969327061143267787e-01,
     2.797053914892766679014677714229e-01,
     3.8183005050511894495036977548818e-01,
     4.1795918367346938775510204081658e-01]) )
struct Segment
    a::Number
    b::Number
    I
    E
end
Base.isless(i::Segment, j::Segment) = isless(i.E, j.E)
function evalrule(f, a,b, x,w,gw, nrm)
    s = convert(eltype(x), 0.5) * (b-a)
    n1 = 1 - (length(x) & 1) # 0 if even order, 1 if odd order
    fg = f(a + (1+x[2])*s) + f(a + (1-x[2])*s)
    fk = f(a + (1+x[1])*s) + f(a + (1-x[1])*s)
    Ig = fg * gw[1]
    Ik = fg * w[2] + fk * w[1]
    for i = 2:length(gw)-n1
        fg = f(a + (1+x[2i])*s) + f(a + (1-x[2i])*s)
        fk = f(a + (1+x[2i-1])*s) + f(a + (1-x[2i-1])*s)
        Ig += fg * gw[i]
        Ik += fg * w[2i] + fk * w[2i-1]
    end
    if n1 == 0 # even: Gauss rule does not include x == 0
        Ik += f(a + s) * w[end]
    else # odd: don't count x==0 twice in Gauss rule
        f0 = f(a + s)
        Ig += f0 * gw[end]
        Ik += f0 * w[end] +
              (f(a + (1+x[end-1])*s) + f(a + (1-x[end-1])*s)) * w[end-1]
    end
    Ik *= s
    Ig *= s
    E = nrm(Ik - Ig)
    if isnan(E) || isinf(E)
        throw(DomainError(a+s, "integrand produced $E in the interval ($a, $b)"))
    end
    return Segment(a, b, Ik, E)
end
rulekey(::Type{BigFloat}, n) = (BigFloat, precision(BigFloat), n)
rulekey(T,n) = (T,n)
function do_quadgk(f, s, n, ::Type{Tw}, atol, rtol, maxevals, nrm) where Tw
    if eltype(s) <: Real # check for infinite or semi-infinite intervals
        s1 = s[1]; s2 = s[end]; inf1 = isinf(s1); inf2 = isinf(s2)
        if inf1 || inf2
            if inf1 && inf2 # x = t/(1-t^2) coordinate transformation
                return do_quadgk(t -> begin t2 = t*t; den = 1 / (1 - t2);
                                            f(t*den) * (1+t2)*den*den; end,
                                 map(x -> isinf(x) ? copysign(one(x), x) : 2x / (1+hypot(1,2x)), s),
                                 n, Tw, atol, rtol, maxevals, nrm)
            end
            s0,si = inf1 ? (s2,s1) : (s1,s2)
            if si < 0 # x = s0 - t/(1-t)
                return do_quadgk(t -> begin den = 1 / (1 - t);
                                            f(s0 - t*den) * den*den; end,
                                 reverse!(map(x -> 1 / (1 + 1 / (s0 - x)), s)),
                                 n, Tw, atol, rtol, maxevals, nrm)
            else # x = s0 + t/(1-t)
                return do_quadgk(t -> begin den = 1 / (1 - t);
                                            f(s0 + t*den) * den*den; end,
                                 map(x -> 1 / (1 + 1 / (x - s0)), s),
                                 n, Tw, atol, rtol, maxevals, nrm)
            end
        end
    end
    key = rulekey(Tw,n)
    x,w,gw = haskey(rulecache, key) ? rulecache[key] :
       (rulecache[key] = kronrod(Tw, n))
    segs = Segment[]
    for i in 1:length(s) - 1
        heappush!(segs, evalrule(f, s[i],s[i+1], x,w,gw, nrm), Reverse)
    end
    numevals = (2n+1) * length(segs)
    I = segs[1].I
    E = segs[1].E
    for i in 2:length(segs)
        I += segs[i].I
        E += segs[i].E
    end
    while E > atol && E > rtol * nrm(I) && numevals < maxevals
        s = heappop!(segs, Reverse)
        mid = (s.a + s.b) / 2
        s1 = evalrule(f, s.a, mid, x,w,gw, nrm)
        s2 = evalrule(f, mid, s.b, x,w,gw, nrm)
        heappush!(segs, s1, Reverse)
        heappush!(segs, s2, Reverse)
        I = (I - s.I) + s1.I + s2.I
        E = (E - s.E) + s1.E + s2.E
        numevals += 4n+2
    end
    I = segs[1].I
    E = segs[1].E
    for i in 2:length(segs)
        I += segs[i].I
        E += segs[i].E
    end
    return (I, E)
end
function tols(atol,rtol,abstol,reltol)
    if !ismissing(abstol) || !ismissing(reltol)
        Base.depwarn("abstol and reltol keywords are now atol and rtol, respectively", :quadgk)
    end
    return coalesce(abstol,atol), coalesce(reltol,rtol)
end
function quadgk(f, a::T,b::T,c::T...;
                atol=zero(T), rtol=sqrt(eps(T)), abstol=missing, reltol=missing,
                maxevals=10^7, order=7, norm=norm) where T<:AbstractFloat
    atol_,rtol_ = tols(atol,rtol,abstol,reltol)
    do_quadgk(f, [a, b, c...], order, T, atol_, rtol_, maxevals, norm)
end
function quadgk(f, a::Complex{T},
                b::Complex{T},c::Complex{T}...;
                atol=zero(T), rtol=sqrt(eps(T)), abstol=missing, reltol=missing,
                maxevals=10^7, order=7, norm=norm) where T<:AbstractFloat
    atol_,rtol_ = tols(atol,rtol,abstol,reltol)
    do_quadgk(f, [a, b, c...], order, T, atol_, rtol_, maxevals, norm)
end
""" """ function quadgk(f, a, b, c...; kws...)
    T = promote_type(typeof(float(a)), typeof(b))
    for x in c
        T = promote_type(T, typeof(x))
    end
    cT = map(T, c)
    quadgk(f, convert(T, a), convert(T, b), cT...; kws...)
end
function eigpoly(b,z,m=length(b)+1)
    d1 = z
    d1deriv = d2 = one(z)
    d2deriv = zero(z)
    for i = 2:m
        b2 = b[i-1]^2
        d = z * d1 - b2 * d2
        dderiv = d1 + z * d1deriv - b2 * d2deriv
        d2 = d1
        d1 = d
        d2deriv = d1deriv
        d1deriv = dderiv
    end
    return (d1, d1deriv)
end
function eignewt(b,m,n)
    H = SymTridiagonal(zeros(m), Float64[ b[i] for i in 1:m-1 ])
    lambda0 = sort(eigvals(H))
    lambda = Array{eltype(b)}(undef, n)
    for i = 1:n
        lambda[i] = lambda0[i]
        for k = 1:1000
            (p,pderiv) = eigpoly(b,lambda[i],m)
            lambda[i] = (lamold = lambda[i]) - p / pderiv
            if abs(lambda[i] - lamold) < 10 * eps(lambda[i]) * abs(lambda[i])
                break
            end
        end
        (p,pderiv) = eigpoly(b,lambda[i],m)
        lambda[i] = lambda[i] - p / pderiv
    end
    return lambda
end
function eigvec1(b,z::Number,m=length(b)+1)
    v = Array{eltype(b)}(undef, m)
    v[1] = 1
    if m > 1
        s = v[1]
        v[2] = z * v[1] / b[1]
        s += v[2]^2
        for i = 3:m
            v[i] = - (b[i-2]*v[i-2] - z*v[i-1]) / b[i-1]
            s += v[i]^2
        end
        rmul!(v, 1 / sqrt(s))
    end
    return v
end
""" """ function gauss(::Type{T}, N::Integer) where T<:AbstractFloat
    if N < 1
        throw(ArgumentError("Gauss rules require positive order"))
    end
    o = one(T)
    b = T[ n / sqrt(4n^2 - o) for n = 1:N-1 ]
    x = eignewt(b,N,N)
    w = T[ 2*eigvec1(b,x[i])[1]^2 for i = 1:N ]
    return (x, w)
end
gauss(N::Integer) = gauss(Float64, N)
""" """ function kronrod(::Type{T}, n::Integer) where T<:AbstractFloat
    if n < 1
        throw(ArgumentError("Kronrod rules require positive order"))
    end
    o = one(T)
    b = zeros(T, 2n+1)
    b[1] = 2*o
    for j = 1:div(3n+1,2)
        b[j+1] = j^2 / (4j^2 - o)
    end
    s = zeros(T, div(n,2) + 2)
    t = zeros(T, div(n,2) + 2)
    t[2] = b[n+2]
    for m = 0:n-2
        u = zero(T)
        for k = div(m+1,2):-1:0
            l = m - k + 1
            k1 = k + n + 2
            u += b[k1]*s[k+1] - b[l]*s[k+2]
            s[k+2] = u
        end
        s,t = t,s
    end
    for j = div(n,2):-1:0
        s[j+2] = s[j+1]
    end
    for m = n-1:2n-3
        u = zero(T)
        for k = m+1-n:div(m-1,2)
            l = m - k + 1
            j = n - l
            k1 = k + n + 2
            u -= b[k1]*s[j+2] - b[l]*s[j+3]
            s[j+2] = u
        end
        k = div(m+1,2)
        if 2k != m
            j = n - (m - k + 2)
            b[k+n+2] = s[j+2] / s[j+3]
        end
        s,t = t,s
    end
    for j = 1:2n
        b[j] = sqrt(b[j+1])
    end
    x = eignewt(b,2n+1,n+1) # x <= 0
    w = T[ 2*eigvec1(b,x[i],2n+1)[1]^2 for i in 1:n+1 ]
    for j = 1:n-1
        b[j] = j / sqrt(4j^2 - o)
    end
    gw = T[ 2*eigvec1(b,x[i],n)[1]^2 for i = 2:2:n+1 ]
    return (x, w, gw)
end
kronrod(N::Integer) = kronrod(Float64, N)
end # module QuadGK
