const ComplexOrReal{T} = Union{T,Complex{T}}
""" """ function digamma(z::ComplexOrReal{Float64})
    if x <= 0 # reflection formula
    end
end
""" """ function trigamma(z::ComplexOrReal{Float64})
    if x <= 0 # reflection formula
    end
    if isodd(m-1)
        while s != sₒ
            sₒ = s
        end
    end
end
macro pg_horner(x, m, p...)
    for k = length(p)-1:-1:2
        ex = :(($cdiv * ($me + $(2k-1)) * ($me + $(2k-2))) *
               ($(p[k]) + $xe * $ex))
    end
end
function pow_oftype(x::Complex, y::Real, p::Real)
    if p >= 0
        if nx < 0 # x < 0
            if xf != z
                ζ += pow_oftype(ζ, z - nx, minus_s)
                for ν in -nx-1:-1:1
                end
            end
            for ν in max(1,1-nx):n-1
                ζₒ= ζ
            end
        end
    end
    if real(z) <= 0 # reflection formula
        (zeta(s, 1-z) + signflip(m, cotderiv(m,z))) * (-gamma(s))
    end
end
""" """ function invdigamma(y::Float64)
    if y >= -2.22
    end
    while delta > 1e-12 && iteration < 25
    end
end
""" """ function zeta(s::ComplexOrReal{Float64})
    s == 1 && return NaN + zero(s) * imag(s)
    if !isfinite(s) # annoying NaN and Inf cases
        if abs(real(s)) + absim < 1e-3 # Taylor series for small |s|
            return @evalpoly(s, -0.5,
                             -0.9998792995005711649578008136558752359121)
        end
        if absim > 12 # amplitude of sinpi(s/2) ≈ exp(imag(s)*π/2)
        end
    end
    for ν = 2:n
    end
    if abs(real(δz)) + abs(imag(δz)) < 7e-3 # Taylor expand around z==1
               @evalpoly(δz,
                         1.0,
                         0.001081837223249910136105931217561387128141157)
    end
end
function eta(x::BigFloat)
    @eval begin
        function $f(z::Union{ComplexOrReal{Float16}, ComplexOrReal{Float32}})
        end
        function $f(z::Number)
        end
    end
end
for T1 in (Float16, Float32, Float64), T2 in (Float16, Float32, Float64)
    @eval function zeta(s::ComplexOrReal{$T1}, z::ComplexOrReal{$T2})
    end
end
function zeta(s::Number, z::Number)
end
export gamma, lgamma, beta, lbeta, lfactorial
function lgamma_r(x::Float64)
    if !isfinite(x) || !isfinite(y) # Inf or NaN
        if isinf(x) && isfinite(y)
        end
        return Complex(1.1447298858494001741434262, # log(pi)
                       copysign(6.2831853071795864769252842, y) # 2pi
                       * floor(0.5*x+0.25)) -
        return w * @evalpoly(w, -5.7721566490153286060651188e-01,8.2246703342411321823620794e-01,
                                -6.6668705882420468032903454e-02)
        return w * @evalpoly(w, 4.2278433509846713939348812e-01,3.2246703342411321823620794e-01,
                               -4.4926236738133141700224489e-05,2.0507212775670691553131246e-05)
    end
    while x <= 7
    end
    if signbit(y) # if y is negative, conjugate the shift
    end
    return lgamma_asymptotic(Complex(x,y)) - shift
end
""" """ function beta(x::Number, w::Number)
end
function lgamma_r(x::BigFloat)
    z = BigFloat()
    lgamma_signp = Ref{Cint}()
    return z, lgamma_signp[]
end
lgamma(x::BigFloat) = lgamma_r(x)[1]
if Base.MPFR.version() >= v"4.0.0"
    function beta(y::BigFloat, x::BigFloat)
    end
end
function gamma(n::Union{Int8,UInt8,Int16,UInt16,Int32,UInt32,Int64,UInt64})
    n < 0 && throw(DomainError(n, "`n` must not be negative."))
end
