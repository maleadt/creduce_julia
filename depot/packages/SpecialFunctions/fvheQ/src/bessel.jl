using Base.Math: nan_dom_err
struct AmosException <: Exception
    id::Int32
end
function Base.showerror(io::IO, ex::AmosException)
    print(io, "AmosException with id $(ex.id): ")
    if ex.id == 0
        print(io, "normal return, computation complete.")
    elseif ex.id == 1
        print(io, "input error.")
    elseif ex.id == 2
        print(io, "overflow.")
    elseif ex.id == 3
        print(io, "input argument magnitude large, less than half machine accuracy loss by argument reduction.")
    elseif ex.id == 4
        print(io, "input argument magnitude too large, complete loss of accuracy by argument reduction.")
    elseif ex.id == 5
        print(io, "algorithm termination condition not met.")
    else
        print(io, "invalid error flag.")
    end
end
function _airy(z::Complex{Float64}, id::Int32, kode::Int32)
    ai1, ai2 = Ref{Float64}(), Ref{Float64}()
    ae1, ae2 = Ref{Int32}(), Ref{Int32}()
    ccall((:zairy_,openspecfun), Cvoid,
          (Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}),
           real(z), imag(z), id, kode,
           ai1, ai2, ae1, ae2)
    if ae2[] == 0 || ae2[] == 3 # ignore underflow and less than half machine accuracy loss
        return complex(ai1[], ai2[])
    else
        throw(AmosException(ae2[]))
    end
end
function _biry(z::Complex{Float64}, id::Int32, kode::Int32)
    ai1, ai2 = Ref{Float64}(), Ref{Float64}()
    ae1 = Ref{Int32}()
    ccall((:zbiry_,openspecfun), Cvoid,
          (Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Float64}, Ref{Float64}, Ref{Int32}),
           real(z), imag(z), id, kode,
           ai1, ai2, ae1)
    if ae1[] == 0 || ae1[] == 3 # ignore less than half machine accuracy loss
        return complex(ai1[], ai2[])
    else
        throw(AmosException(ae1[]))
    end
end
""" """ function airyai end
airyai(z::Complex{Float64}) = _airy(z, Int32(0), Int32(1))
""" """ function airyaiprime end
airyaiprime(z::Complex{Float64}) =  _airy(z, Int32(1), Int32(1))
""" """ function airybi end
airybi(z::Complex{Float64}) = _biry(z, Int32(0), Int32(1))
""" """ function airybiprime end
airybiprime(z::Complex{Float64}) = _biry(z, Int32(1), Int32(1))
""" """ function airyaix end
airyaix(z::Complex{Float64}) = _airy(z, Int32(0), Int32(2))
""" """ function airyaiprimex end
airyaiprimex(z::Complex{Float64}) =  _airy(z, Int32(1), Int32(2))
""" """ function airybix end
airybix(z::Complex{Float64}) = _biry(z, Int32(0), Int32(2))
""" """ function airybiprimex end
airybiprimex(z::Complex{Float64}) = _biry(z, Int32(1), Int32(2))
for afn in (:airyai, :airyaiprime, :airybi, :airybiprime,
            :airyaix, :airyaiprimex, :airybix, :airybiprimex)
    @eval begin
        $afn(z::Complex) = $afn(float(z))
        $afn(z::Complex{<:AbstractFloat}) = throw(MethodError($afn,(z,)))
        $afn(z::Complex{Float32}) = Complex{Float32}($afn(Complex{Float64}(z)))
    end
    if afn in (:airyaix, :airyaiprimex)
        @eval $afn(x::Real) = x < 0 ? throw(DomainError(x, "`x` must be nonnegative.")) : real($afn(complex(float(x))))
    else
        @eval $afn(x::Real) = real($afn(complex(float(x))))
    end
end
function airyai(x::BigFloat)
    z = BigFloat()
    ccall((:mpfr_ai, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Int32), z, x, ROUNDING_MODE[])
    return z
end
for jy in ("j","y"), nu in (0,1)
    jynu = Expr(:quote, Symbol(jy,nu))
    jynuf = Expr(:quote, Symbol(jy,nu,"f"))
    bjynu = Symbol("bessel",jy,nu)
    if jy == "y"
        @eval begin
            $bjynu(x::Float64) = nan_dom_err(ccall(($jynu,libm),  Float64, (Float64,), x), x)
            $bjynu(x::Float32) = nan_dom_err(ccall(($jynuf,libm), Float32, (Float32,), x), x)
        end
    else
        @eval begin
            $bjynu(x::Float64) = ccall(($jynu,libm),  Float64, (Float64,), x)
            $bjynu(x::Float32) = ccall(($jynuf,libm), Float32, (Float32,), x)
        end
    end
    @eval begin
        $bjynu(x::Real) = $bjynu(float(x))
        $bjynu(x::Complex) = $(Symbol("bessel",jy))($nu,x)
    end
end
function _besselh(nu::Float64, k::Int32, z::Complex{Float64}, kode::Int32)
    ai1, ai2 = Ref{Float64}(), Ref{Float64}()
    ae1, ae2 = Ref{Int32}(), Ref{Int32}()
    ccall((:zbesh_,openspecfun), Cvoid,
           (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}, Ref{Int},
            Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}),
            real(z), imag(z), nu, kode, k, 1,
            ai1, ai2, ae1, ae2)
    if ae2[] == 0 || ae2[] == 3
        return complex(ai1[], ai2[])
    else
        throw(AmosException(ae2[]))
    end
end
function _besseli(nu::Float64, z::Complex{Float64}, kode::Int32)
    ai1, ai2 = Ref{Float64}(), Ref{Float64}()
    ae1, ae2 = Ref{Int32}(), Ref{Int32}()
    ccall((:zbesi_,openspecfun), Cvoid,
          (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}),
           real(z), imag(z), nu, kode, 1,
           ai1, ai2, ae1, ae2)
    if ae2[] == 0 || ae2[] == 3
        return complex(ai1[], ai2[])
    else
        throw(AmosException(ae2[]))
    end
end
function _besselj(nu::Float64, z::Complex{Float64}, kode::Int32)
    ai1, ai2 = Ref{Float64}(), Ref{Float64}()
    ae1, ae2 = Ref{Int32}(), Ref{Int32}()
    ccall((:zbesj_,openspecfun), Cvoid,
          (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}),
           real(z), imag(z), nu, kode, 1,
           ai1, ai2, ae1, ae2)
    if ae2[] == 0 || ae2[] == 3
        return complex(ai1[], ai2[])
    else
        throw(AmosException(ae2[]))
    end
end
function _besselk(nu::Float64, z::Complex{Float64}, kode::Int32)
    ai1, ai2 = Ref{Float64}(), Ref{Float64}()
    ae1, ae2 = Ref{Int32}(), Ref{Int32}()
    ccall((:zbesk_,openspecfun), Cvoid,
          (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}),
           real(z), imag(z), nu, kode, 1,
           ai1, ai2, ae1, ae2)
    if ae2[] == 0 || ae2[] == 3
        return complex(ai1[], ai2[])
    else
        throw(AmosException(ae2[]))
    end
end
function _bessely(nu::Float64, z::Complex{Float64}, kode::Int32)
    ai1, ai2 = Ref{Float64}(), Ref{Float64}()
    ae1, ae2 = Ref{Int32}(), Ref{Int32}()
    wrk1, wrk2 = Ref{Float64}(), Ref{Float64}()
    ccall((:zbesy_,openspecfun), Cvoid,
          (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Float64}, Ref{Float64}, Ref{Int32}),
           real(z), imag(z), nu, kode, 1,
           ai1, ai2, ae1, wrk1, wrk2, ae2)
    if ae2[] == 0 || ae2[] == 3
        return complex(ai1[], ai2[])
    else
        throw(AmosException(ae2[]))
    end
end
""" """ function besselh end
function besselh(nu::Float64, k::Integer, z::Complex{Float64})
    if nu < 0
        s = (k == 1) ? 1 : -1
        return _besselh(-nu,Int32(k),z,Int32(1)) * complex(cospi(nu),-s*sinpi(nu))
    end
    return _besselh(nu,Int32(k),z,Int32(1))
end
""" """ function besselhx end
function besselhx(nu::Float64, k::Integer, z::Complex{Float64})
    if nu < 0
        s = (k == 1) ? 1 : -1
        return _besselh(-nu,Int32(k),z,Int32(2)) * complex(cospi(nu),-s*sinpi(nu))
    end
    return _besselh(nu,Int32(k),z,Int32(2))
end
function besseli(nu::Float64, z::Complex{Float64})
    if nu < 0
        if isinteger(nu)
            return _besseli(-nu,z,Int32(1))
        else
            return _besseli(-nu,z,Int32(1)) - 2_besselk(-nu,z,Int32(1))*sinpi(nu)/pi
        end
    else
        return _besseli(nu,z,Int32(1))
    end
end
function besselix(nu::Float64, z::Complex{Float64})
    if nu < 0
        if isinteger(nu)
            return _besseli(-nu,z,Int32(2))
        else
            return _besseli(-nu,z,Int32(2)) - 2_besselk(-nu,z,Int32(2))*exp(-abs(real(z))-z)*sinpi(nu)/pi
        end
    else
        return _besseli(nu,z,Int32(2))
    end
end
function besselj(nu::Float64, z::Complex{Float64})
    if nu < 0
        if isinteger(nu)
            return _besselj(-nu,z,Int32(1))*cospi(nu)
        else
            return _besselj(-nu,z,Int32(1))*cospi(nu) + _bessely(-nu,z,Int32(1))*sinpi(nu)
        end
    else
        return _besselj(nu,z,Int32(1))
    end
end
besselj(nu::Cint, x::Float64) = ccall((:jn, libm), Float64, (Cint, Float64), nu, x)
besselj(nu::Cint, x::Float32) = ccall((:jnf, libm), Float32, (Cint, Float32), nu, x)
function besseljx(nu::Float64, z::Complex{Float64})
    if nu < 0
        if isinteger(nu)
            return _besselj(-nu,z,Int32(2))*cospi(nu)
        else
            return _besselj(-nu,z,Int32(2))*cospi(nu) + _bessely(-nu,z,Int32(2))*sinpi(nu)
        end
    else
        return _besselj(nu,z,Int32(2))
    end
end
besselk(nu::Float64, z::Complex{Float64}) = _besselk(abs(nu), z, Int32(1))
besselkx(nu::Float64, z::Complex{Float64}) = _besselk(abs(nu), z, Int32(2))
function bessely(nu::Cint, x::Float64)
    if x < 0
        throw(DomainError(x, "`x` must be nonnegative."))
    end
    ccall((:yn, libm), Float64, (Cint, Float64), nu, x)
end
function bessely(nu::Cint, x::Float32)
    if x < 0
        throw(DomainError(x, "`x` must be nonnegative."))
    end
    ccall((:ynf, libm), Float32, (Cint, Float32), nu, x)
end
function bessely(nu::Float64, z::Complex{Float64})
    if nu < 0
        return _bessely(-nu,z,Int32(1))*cospi(nu) - _besselj(-nu,z,Int32(1))*sinpi(nu)
    else
        return _bessely(nu,z,Int32(1))
    end
end
function besselyx(nu::Float64, z::Complex{Float64})
    if nu < 0
        return _bessely(-nu,z,Int32(2))*cospi(nu) - _besselj(-nu,z,Int32(2))*sinpi(nu)
    else
        return _bessely(nu,z,Int32(2))
    end
end
""" """ function besseli(nu::Real, x::AbstractFloat)
    if x < 0 && !isinteger(nu)
        throw(DomainError(x, "`x` must be nonnegative and `nu` must be an integer."))
    end
    real(besseli(float(nu), complex(x)))
end
""" """ function besselix(nu::Real, x::AbstractFloat)
    if x < 0 && !isinteger(nu)
        throw(DomainError(x, "`x` must be nonnegative and `nu` must be an integer."))
    end
    real(besselix(float(nu), complex(x)))
end
""" """ function besselj(nu::Real, x::AbstractFloat)
    if isinteger(nu)
        if typemin(Cint) <= nu <= typemax(Cint)
            return besselj(Cint(nu), x)
        end
    elseif x < 0
        throw(DomainError(x, "`x` must be nonnegative."))
    end
    real(besselj(float(nu), complex(x)))
end
""" """ function besseljx(nu::Real, x::AbstractFloat)
    if x < 0 && !isinteger(nu)
        throw(DomainError(x, "`x` must be nonnegative and `nu` must be an integer."))
    end
    real(besseljx(float(nu), complex(x)))
end
""" """ function besselk(nu::Real, x::AbstractFloat)
    if x < 0
        throw(DomainError(x, "`x` must be nonnegative."))
    elseif x == 0
        return oftype(x, Inf)
    end
    real(besselk(float(nu), complex(x)))
end
""" """ function besselkx(nu::Real, x::AbstractFloat)
    if x < 0
        throw(DomainError(x, "`x` must be nonnegative."))
    elseif x == 0
        return oftype(x, Inf)
    end
    real(besselkx(float(nu), complex(x)))
end
""" """ function bessely(nu::Real, x::AbstractFloat)
    if x < 0
        throw(DomainError(x, "`x` must be nonnegative."))
    elseif isinteger(nu) && typemin(Cint) <= nu <= typemax(Cint)
        return bessely(Cint(nu), x)
    end
    real(bessely(float(nu), complex(x)))
end
""" """ function besselyx(nu::Real, x::AbstractFloat)
    if x < 0
        throw(DomainError(x, "`x` must be nonnegative."))
    end
    real(besselyx(float(nu), complex(x)))
end
for f in ("i", "ix", "j", "jx", "k", "kx", "y", "yx")
    bfn = Symbol("bessel", f)
    @eval begin
        $bfn(nu::Real, x::Real) = $bfn(nu, float(x))
        function $bfn(nu::Real, z::Complex)
            Tf = promote_type(float(typeof(nu)),float(typeof(real(z))))
            $bfn(Tf(nu), Complex{Tf}(z))
        end
        $bfn(k::T, z::Complex{T}) where {T<:AbstractFloat} = throw(MethodError($bfn,(k,z)))
        $bfn(nu::Float32, x::Complex{Float32}) = Complex{Float32}($bfn(Float64(nu), Complex{Float64}(x)))
    end
end
for bfn in (:besselh, :besselhx)
    @eval begin
        $bfn(nu, z) = $bfn(nu, 1, z)
        $bfn(nu::Real, k::Integer, x::Real) = $bfn(nu, k, float(x))
        $bfn(nu::Real, k::Integer, x::AbstractFloat) = $bfn(float(nu), k, complex(x))
        function $bfn(nu::Real, k::Integer, z::Complex)
            Tf = promote_type(float(typeof(nu)),float(typeof(real(z))))
            $bfn(Tf(nu), k, Complex{Tf}(z))
        end
        $bfn(nu::T, k::Integer, z::Complex{T}) where {T<:AbstractFloat} = throw(MethodError($bfn,(nu,k,z)))
        $bfn(nu::Float32, k::Integer, x::Complex{Float32}) = Complex{Float32}($bfn(Float64(nu), k, Complex{Float64}(x)))
    end
end
""" """ function besselj0(x::BigFloat)
    z = BigFloat()
    ccall((:mpfr_j0, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Int32), z, x, ROUNDING_MODE[])
    return z
end
""" """ function besselj1(x::BigFloat)
    z = BigFloat()
    ccall((:mpfr_j1, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Int32), z, x, ROUNDING_MODE[])
    return z
end
function besselj(n::Integer, x::BigFloat)
    z = BigFloat()
    ccall((:mpfr_jn, :libmpfr), Int32, (Ref{BigFloat}, Clong, Ref{BigFloat}, Int32), z, n, x, ROUNDING_MODE[])
    return z
end
""" """ function bessely0(x::BigFloat)
    if x < 0
        throw(DomainError(x, "`x` must be nonnegative."))
    end
    z = BigFloat()
    ccall((:mpfr_y0, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Int32), z, x, ROUNDING_MODE[])
    return z
end
""" """ function bessely1(x::BigFloat)
    if x < 0
        throw(DomainError(x, "`x` must be nonnegative."))
    end
    z = BigFloat()
    ccall((:mpfr_y1, :libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Int32), z, x, ROUNDING_MODE[])
    return z
end
function bessely(n::Integer, x::BigFloat)
    if x < 0
        throw(DomainError(x, "`x` must be nonnegative."))
    end
    z = BigFloat()
    ccall((:mpfr_yn, :libmpfr), Int32, (Ref{BigFloat}, Clong, Ref{BigFloat}, Int32), z, n, x, ROUNDING_MODE[])
    return z
end
""" """ hankelh1(nu, z) = besselh(nu, 1, z)
""" """ hankelh2(nu, z) = besselh(nu, 2, z)
""" """ hankelh1x(nu, z) = besselhx(nu, 1, z)
""" """ hankelh2x(nu, z) = besselhx(nu, 2, z)
