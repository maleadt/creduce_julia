using Base.MPFR: ROUNDING_MODE, big_ln2
const ComplexOrReal{T} = Union{T,Complex{T}}
""" """ function digamma(z::ComplexOrReal{Float64})
    x = real(z)
    if x <= 0 # reflection formula
        ψ = -π * cot(π*z)
        z = 1 - z
        x = real(z)
    else
        ψ = zero(z)
    end
    if x < 7
        n = 7 - floor(Int,x)
        for ν = 1:n-1
            ψ -= inv(z + ν)
        end
        ψ -= inv(z)
        z += n
    end
    t = inv(z)
    ψ += log(z) - 0.5*t
    t *= t # 1/z^2
    ψ -= t * @evalpoly(t,0.08333333333333333,-0.008333333333333333,0.003968253968253968,-0.004166666666666667,0.007575757575757576,-0.021092796092796094,0.08333333333333333,-0.4432598039215686)
end
function digamma(x::BigFloat)
    z = BigFloat()
    ccall((:mpfr_digamma, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Int32), z, x, ROUNDING_MODE[])
    return z
end
""" """ function trigamma(z::ComplexOrReal{Float64})
    x = real(z)
    if x <= 0 # reflection formula
        return (π * csc(π*z))^2 - trigamma(1 - z)
    end
    ψ = zero(z)
    if x < 8
        n = 8 - floor(Int,x)
        ψ += inv(z)^2
        for ν = 1:n-1
            ψ += inv(z + ν)^2
        end
        z += n
    end
    t = inv(z)
    w = t * t # 1/z^2
    ψ += t + 0.5*w
    ψ += t*w * @evalpoly(w,0.16666666666666666,-0.03333333333333333,0.023809523809523808,-0.03333333333333333,0.07575757575757576,-0.2531135531135531,1.1666666666666667,-7.092156862745098)
end
signflip(m::Number, z) = (-1+0im)^m * z
signflip(m::Integer, z) = iseven(m) ? z : -z
function cotderiv_q(m::Int)
    m < 0 && throw(DomainError(m, "`m` must be nonnegative."))
    m == 0 && return [1.0]
    m == 1 && return [1.0, 1.0]
    q₋ = cotderiv_q(m-1)
    d = length(q₋) - 1 # degree of q₋
    if isodd(m-1)
        q = Vector{Float64}(undef, length(q₋))
        q[end] = d * q₋[end] * 2/m
        for i = 1:length(q)-1
            q[i] = ((i-1)*q₋[i] + i*q₋[i+1]) * 2/m
        end
    else # iseven(m-1)
        q = Vector{Float64}(undef, length(q₋) + 1)
        q[1] = q₋[1] / m
        q[end] = (1 + 2d) * q₋[end] / m
        for i = 2:length(q)-1
            q[i] = ((1 + 2(i-1))*q₋[i] + (1 + 2(i-2))*q₋[i-1]) / m
        end
    end
    return q
end
const cotderiv_Q = [cotderiv_q(m) for m in 1:100]
function cotderiv(m::Integer, z)
    isinf(imag(z)) && return zero(z)
    if m <= 0
        m == 0 && return π * cot(π*z)
        throw(DomainError(m, "`m` must be nonnegative."))
    end
    if m <= length(cotderiv_Q)
        q = cotderiv_Q[m]
        x = cot(π*z)
        y = x*x
        s = q[1] + q[2] * y
        t = y
        for i = 3:length(q)
            t *= y
            s += q[i] * t
        end
        return π^(m+1) * (isodd(m) ? s : x*s)
    else # m is large, series derivative should converge quickly
        p = m+1
        z -= round(real(z))
        s = inv(z^p)
        n = 1
        sₒ = zero(s)
        while s != sₒ
            sₒ = s
            a = (z+n)^p
            b = (z-n)^p
            s += (a + b) / (a * b)
            n += 1
        end
        return s
    end
end
macro pg_horner(x, m, p...)
    k = length(p)
    me = esc(m)
    xe = esc(x)
    ex = :(($me + $(2k-1)) * ($me + $(2k-2)) * $(p[end]/((2k-1)*(2k-2))))
    for k = length(p)-1:-1:2
        cdiv = 1 / ((2k-1)*(2k-2))
        ex = :(($cdiv * ($me + $(2k-1)) * ($me + $(2k-2))) *
               ($(p[k]) + $xe * $ex))
    end
    :(($me + 1) * ($(p[1]) + $xe * $ex))
end
pow_oftype(x, y, p) = oftype(x, y)^p
pow_oftype(x::Complex, y::Real, p::Complex) = oftype(x, y^p)
function pow_oftype(x::Complex, y::Real, p::Real)
    if p >= 0
        return oftype(x, y^p)
    else
        yp = y^-p # use real power for efficiency
        return oftype(x, Complex(yp, -zero(yp))) # get correct sign of zero!
    end
end
""" """ function zeta(s::ComplexOrReal{Float64}, z::ComplexOrReal{Float64})
    ζ = zero(promote_type(typeof(s), typeof(z)))
    (z == 1 || z == 0) && return oftype(ζ, zeta(s))
    s == 2 && return oftype(ζ, trigamma(z))
    x = real(z)
    if !isfinite(s)
        (isnan(s) || isnan(z)) && return (s*z)^2 # type-stable NaN+Nan*im
        if real(s) == Inf
            z == 1 && return one(ζ)
            if x > 1 || (x >= 0.5 ? abs(z) > 1 : abs(z - round(x)) > 1)
                return zero(ζ) # distance to poles is > 1
            end
            x > 0 && imag(z) == 0 && imag(s) == 0 && return oftype(ζ, Inf)
        end
        throw(DomainError(s, "`s` must be finite."))  # nothing clever to return
    end
    if isnan(x)
        if imag(z) == 0 && imag(s) == 0
            return oftype(ζ, x)
        else
            return oftype(ζ, Complex(x,x))
        end
    end
    m = s - 1
    cutoff = 7 + real(m) + abs(imag(m)) # TODO: this cutoff is too conservative?
    if x < cutoff
        xf = floor(x)
        nx = Int(xf)
        n = ceil(Int, cutoff - nx)
        minus_s = -s
        if nx < 0 # x < 0
            minus_z = -z
            ζ += pow_oftype(ζ, minus_z, minus_s) # ν = 0 term
            if xf != z
                ζ += pow_oftype(ζ, z - nx, minus_s)
            end
            if real(s) > 0
                for ν in -nx-1:-1:1
                    ζₒ= ζ
                    ζ += pow_oftype(ζ, minus_z - ν, minus_s)
                    ζ == ζₒ && break # prevent long loop for large -x > 0
                end
            else
                for ν in 1:-nx-1
                    ζₒ= ζ
                    ζ += pow_oftype(ζ, minus_z - ν, minus_s)
                    ζ == ζₒ && break # prevent long loop for large -x > 0
                end
            end
        else # x ≥ 0 && z != 0
            ζ += pow_oftype(ζ, z, minus_s)
        end
        if real(s) > 0
            for ν in max(1,1-nx):n-1
                ζₒ= ζ
                ζ += pow_oftype(ζ, z + ν, minus_s)
                ζ == ζₒ && break # prevent long loop for large m
            end
        else
            for ν in n-1:-1:max(1,1-nx)
                ζₒ= ζ
                ζ += pow_oftype(ζ, z + ν, minus_s)
                ζ == ζₒ && break # prevent long loop for large m
            end
        end
        z += n
    end
    t = inv(z)
    w = isa(t, Real) ? conj(oftype(ζ, t))^m : oftype(ζ, t)^m
    ζ += w * (inv(m) + 0.5*t)
    t *= t # 1/z^2
    ζ += w*t * @pg_horner(t,m,0.08333333333333333,-0.008333333333333333,0.003968253968253968,-0.004166666666666667,0.007575757575757576,-0.021092796092796094,0.08333333333333333,-0.4432598039215686,3.0539543302701198)
    return ζ
end
""" """ function polygamma(m::Integer, z::ComplexOrReal{Float64})
    m == 0 && return digamma(z)
    m == 1 && return trigamma(z)
    real(m) < 0 && throw(DomainError(m, "`real(m)` must be nonnegative, since the definition in this case is ambiguous."))
    s = Float64(m+1)
    if real(z) <= 0 # reflection formula
        (zeta(s, 1-z) + signflip(m, cotderiv(m,z))) * (-gamma(s))
    else
        signflip(m, zeta(s,z) * (-gamma(s)))
    end
end
""" """ function invdigamma(y::Float64)
    if y >= -2.22
        x_old = exp(y) + 0.5
        x_new = x_old
    else
        x_old = -1.0 / (y - digamma(1.0))
        x_new = x_old
    end
    delta = Inf
    iteration = 0
    while delta > 1e-12 && iteration < 25
        iteration += 1
        x_new = x_old - (digamma(x_old) - y) / trigamma(x_old)
        delta = abs(x_new - x_old)
        x_old = x_new
    end
    return x_new
end
""" """ function zeta(s::ComplexOrReal{Float64})
    s == 1 && return NaN + zero(s) * imag(s)
    if !isfinite(s) # annoying NaN and Inf cases
        isnan(s) && return imag(s) == 0 ? s : s*s
        if isfinite(imag(s))
            real(s) > 0 && return 1.0 - zero(s)*imag(s)
            imag(s) == 0 && return NaN + zero(s)
        end
        return NaN*zero(s) # NaN + NaN*im
    elseif real(s) < 0.5
        absim = abs(imag(s))
        if abs(real(s)) + absim < 1e-3 # Taylor series for small |s|
            return @evalpoly(s, -0.5,
                             -0.918938533204672741780329736405617639861,
                             -1.0031782279542924256050500133649802190,
                             -1.00078519447704240796017680222772921424,
                             -0.9998792995005711649578008136558752359121)
        end
        if absim > 12 # amplitude of sinpi(s/2) ≈ exp(imag(s)*π/2)
            lg = lgamma(1 - s)
            ln2pi = 1.83787706640934548356 # log(2pi) to double precision
            rehalf = real(s)*0.5
            return zeta(1 - s) * exp(lg + absim*(pi/2) + s*ln2pi) * (0.5/π) *
                Complex(sinpi(rehalf), copysign(cospi(rehalf), imag(s)))
        else
            return zeta(1 - s) * gamma(1 - s) * sinpi(s*0.5) * (2π)^s / π
        end
    end
    m = s - 1
    n = ceil(Int,6 + 0.7*abs(imag(s-1))^inv(1 + real(m)*0.05))
    ζ = one(s)
    for ν = 2:n
        ζₒ= ζ
        ζ += inv(ν)^s
        ζ == ζₒ && break # prevent long loop for large m
    end
    z = 1 + n
    t = inv(z)
    w = t^m
    ζ += w * (inv(m) + 0.5*t)
    t *= t # 1/z^2
    ζ += w*t * @pg_horner(t,m,0.08333333333333333,-0.008333333333333333,0.003968253968253968,-0.004166666666666667,0.007575757575757576,-0.021092796092796094,0.08333333333333333,-0.4432598039215686,3.0539543302701198)
    return ζ
end
function zeta(x::BigFloat)
    z = BigFloat()
    ccall((:mpfr_zeta, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Int32), z, x, ROUNDING_MODE[])
    return z
end
""" """ function eta(z::ComplexOrReal{Float64})
    δz = 1 - z
    if abs(real(δz)) + abs(imag(δz)) < 7e-3 # Taylor expand around z==1
        return 0.6931471805599453094172321214581765 *
               @evalpoly(δz,
                         1.0,
                         -0.23064207462156020589789602935331414700440,
                         -0.047156357547388879740146103148112380421254,
                         -0.002263576552598880778433550956278702759143568,
                         0.001081837223249910136105931217561387128141157)
    else
        return -zeta(z) * expm1(0.6931471805599453094172321214581765*δz)
    end
end
function eta(x::BigFloat)
    x == 1 && return big_ln2()
    return -zeta(x) * expm1(big_ln2()*(1-x))
end
for T in (Float16, Float32, Float64)
    @eval f64(x::Complex{$T}) = Complex{Float64}(x)
    @eval f64(x::$T) = Float64(x)
end
for f in (:digamma, :trigamma, :zeta, :eta, :invdigamma)
    @eval begin
        function $f(z::Union{ComplexOrReal{Float16}, ComplexOrReal{Float32}})
            oftype(z, $f(f64(z)))
        end
        function $f(z::Number)
            x = float(z)
            typeof(x) === typeof(z) && throw(MethodError($f, (z,)))
            $f(x)
        end
    end
end
for T1 in (Float16, Float32, Float64), T2 in (Float16, Float32, Float64)
    (T1 == T2 == Float64) && continue # Avoid redefining base definition
    @eval function zeta(s::ComplexOrReal{$T1}, z::ComplexOrReal{$T2})
        ζ = zeta(f64(s), f64(z))
        convert(promote_type(typeof(s), typeof(z)),  ζ)
    end
end
function zeta(s::Number, z::Number)
    t = float(s)
    x = float(z)
    if typeof(t) === typeof(s) && typeof(x) === typeof(z)
        throw(MethodError(zeta,(s,z)))
    end
    zeta(t, x)
end
function polygamma(m::Integer, z::Union{ComplexOrReal{Float16}, ComplexOrReal{Float32}})
    oftype(z, polygamma(m, f64(z)))
end
function polygamma(m::Integer, z::Number)
    x = float(z)
    typeof(x) === typeof(z) && throw(MethodError(polygamma, (m,z)))
    polygamma(m, x)
end
export gamma, lgamma, beta, lbeta, lfactorial
gamma(x::Float64) = nan_dom_err(ccall((:tgamma,libm),  Float64, (Float64,), x), x)
gamma(x::Float32) = nan_dom_err(ccall((:tgammaf,libm),  Float32, (Float32,), x), x)
""" """ gamma(x::Real) = gamma(float(x))
function lgamma_r(x::Float64)
    signp = Ref{Int32}()
    y = ccall((:lgamma_r,libm),  Float64, (Float64, Ptr{Int32}), x, signp)
    return y, signp[]
end
function lgamma_r(x::Float32)
    signp = Ref{Int32}()
    y = ccall((:lgammaf_r,libm),  Float32, (Float32, Ptr{Int32}), x, signp)
    return y, signp[]
end
lgamma_r(x::Real) = lgamma_r(float(x))
lgamma_r(x::Number) = lgamma(x), 1 # lgamma does not take abs for non-real x
""" """ lgamma_r
""" """ lfactorial(x::Integer) = x < 0 ? throw(DomainError(x, "`x` must be non-negative.")) : lgamma(x + oneunit(x))
Base.@deprecate lfact lfactorial
""" """ function lgamma end
@inline function lgamma_asymptotic(z::Complex{Float64})
    zinv = inv(z)
    t = zinv*zinv
    return (z-0.5)*log(z) - z + 9.1893853320467274178032927e-01 + # <-- log(2pi)/2
       zinv*@evalpoly(t, 8.3333333333333333333333368e-02,-2.7777777777777777777777776e-03,
                         7.9365079365079365079365075e-04,-5.9523809523809523809523806e-04,
                         8.4175084175084175084175104e-04,-1.9175269175269175269175262e-03,
                         6.4102564102564102564102561e-03,-2.9550653594771241830065352e-02)
end
function lgamma(z::Complex{Float64})
    x = real(z)
    y = imag(z)
    yabs = abs(y)
    if !isfinite(x) || !isfinite(y) # Inf or NaN
        if isinf(x) && isfinite(y)
            return Complex(x, x > 0 ? (y == 0 ? y : copysign(Inf, y)) : copysign(Inf, -y))
        elseif isfinite(x) && isinf(y)
            return Complex(-Inf, y)
        else
            return Complex(NaN, NaN)
        end
    elseif x > 7 || yabs > 7 # use the Stirling asymptotic series for sufficiently large x or |y|
        return lgamma_asymptotic(z)
    elseif x < 0.1 # use reflection formula to transform to x > 0
        if x == 0 && y == 0 # return Inf with the correct imaginary part for z == 0
            return Complex(Inf, signbit(x) ? copysign(oftype(x, pi), -y) : -y)
        end
        return Complex(1.1447298858494001741434262, # log(pi)
                       copysign(6.2831853071795864769252842, y) # 2pi
                       * floor(0.5*x+0.25)) -
               log(sinpi(z)) - lgamma(1-z)
    elseif abs(x - 1) + yabs < 0.1
        w = Complex(x - 1, y)
        return w * @evalpoly(w, -5.7721566490153286060651188e-01,8.2246703342411321823620794e-01,
                                -4.0068563438653142846657956e-01,2.705808084277845478790009e-01,
                                -2.0738555102867398526627303e-01,1.6955717699740818995241986e-01,
                                -1.4404989676884611811997107e-01,1.2550966952474304242233559e-01,
                                -1.1133426586956469049087244e-01,1.000994575127818085337147e-01,
                                -9.0954017145829042232609344e-02,8.3353840546109004024886499e-02,
                                -7.6932516411352191472827157e-02,7.1432946295361336059232779e-02,
                                -6.6668705882420468032903454e-02)
    elseif abs(x - 2) + yabs < 0.1
        w = Complex(x - 2, y)
        return w * @evalpoly(w, 4.2278433509846713939348812e-01,3.2246703342411321823620794e-01,
                               -6.7352301053198095133246196e-02,2.0580808427784547879000897e-02,
                               -7.3855510286739852662729527e-03,2.8905103307415232857531201e-03,
                               -1.1927539117032609771139825e-03,5.0966952474304242233558822e-04,
                               -2.2315475845357937976132853e-04,9.9457512781808533714662972e-05,
                               -4.4926236738133141700224489e-05,2.0507212775670691553131246e-05)
    end
    shiftprod = Complex(x,yabs)
    x += 1
    sb = false # == signbit(imag(shiftprod)) == signbit(yabs)
    signflips = 0
    while x <= 7
        shiftprod *= Complex(x,yabs)
        sb′ = signbit(imag(shiftprod))
        signflips += sb′ & (sb′ != sb)
        sb = sb′
        x += 1
    end
    shift = log(shiftprod)
    if signbit(y) # if y is negative, conjugate the shift
        shift = Complex(real(shift), signflips*-6.2831853071795864769252842 - imag(shift))
    else
        shift = Complex(real(shift), imag(shift) + signflips*6.2831853071795864769252842)
    end
    return lgamma_asymptotic(Complex(x,y)) - shift
end
lgamma(z::Complex{T}) where {T<:Union{Integer,Rational}} = lgamma(float(z))
lgamma(z::Complex{T}) where {T<:Union{Float32,Float16}} = Complex{T}(lgamma(Complex{Float64}(z)))
gamma(z::Complex) = exp(lgamma(z))
""" """ function beta(x::Number, w::Number)
    yx, sx = lgamma_r(x)
    yw, sw = lgamma_r(w)
    yxw, sxw = lgamma_r(x+w)
    return exp(yx + yw - yxw) * (sx*sw*sxw)
end
""" """ lbeta(x::Number, w::Number) = lgamma(x)+lgamma(w)-lgamma(x+w)
function gamma(x::BigFloat)
    isnan(x) && return x
    z = BigFloat()
    ccall((:mpfr_gamma, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Int32), z, x, ROUNDING_MODE[])
    isnan(z) && throw(DomainError(x, "NaN result for non-NaN input."))
    return z
end
function lgamma_r(x::BigFloat)
    z = BigFloat()
    lgamma_signp = Ref{Cint}()
    ccall((:mpfr_lgamma,:libmpfr), Cint, (Ref{BigFloat}, Ref{Cint}, Ref{BigFloat}, Int32), z, lgamma_signp, x, ROUNDING_MODE[])
    return z, lgamma_signp[]
end
lgamma(x::BigFloat) = lgamma_r(x)[1]
if Base.MPFR.version() >= v"4.0.0"
    function beta(y::BigFloat, x::BigFloat)
        z = BigFloat()
        ccall((:mpfr_beta, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Int32), z, y, x, ROUNDING_MODE[])
        return z
    end
end
function gamma(n::Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64})
    n < 0 && throw(DomainError(n, "`n` must not be negative."))
    n == 0 && return Inf
    n <= 2 && return 1.0
    n > 20 && return gamma(Float64(n))
    @inbounds return Float64(Base._fact_table64[n-1])
end
@inline lgamma(x::Float64) = nan_dom_err(ccall((:lgamma, libm), Float64, (Float64,), x), x)
@inline lgamma(x::Float32) = nan_dom_err(ccall((:lgammaf, libm), Float32, (Float32,), x), x)
@inline lgamma(x::Real) = lgamma(float(x))
@static if !hasmethod(Base.factorial, Tuple{Number})
    import Base: factorial
end
factorial(x) = Base.factorial(x) # to make SpecialFunctions.factorial work unconditionally
factorial(x::Number) = gamma(x + 1) # fallback for x not Integer
""" """ function lbinomial(n::T, k::T) where {T<:Integer}
    S = float(T)
    (k < 0) && return typemin(S)
    if n < 0
        n = -n + k - 1
    end
    k > n && return typemin(S)
    (k == 0 || k == n) && return zero(S)
    (k == 1) && return log(abs(n))
    if k > (n>>1)
        k = n - k
    end
    -log1p(n) - lbeta(n - k + one(T), k + one(T))
end
lbinomial(n::Integer, k::Integer) = lbinomial(promote(n, k)...)
