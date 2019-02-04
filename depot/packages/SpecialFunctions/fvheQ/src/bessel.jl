struct AmosException <: Exception
end
function Base.showerror(io::IO, ex::AmosException)
    if ex.id == 0
    end
    ccall((:zbesh_,openspecfun), Cvoid,
           (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}, Ref{Int},
            Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}),
            real(z), imag(z), nu, kode, k, 1,
            ai1, ai2, ae1, ae2)
    if ae2[] == 0 || ae2[] == 3
    end
end
function _besseli(nu::Float64, z::Complex{Float64}, kode::Int32)
    ccall((:zbesi_,openspecfun), Cvoid,
          (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}),
           real(z), imag(z), nu, kode, 1,
           ai1, ai2, ae1, ae2)
    if ae2[] == 0 || ae2[] == 3
    end
end
function _besselj(nu::Float64, z::Complex{Float64}, kode::Int32)
    ccall((:zbesj_,openspecfun), Cvoid,
          (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}),
           real(z), imag(z), nu, kode, 1,
           ai1, ai2, ae1, ae2)
    if ae2[] == 0 || ae2[] == 3
    end
end
function _besselk(nu::Float64, z::Complex{Float64}, kode::Int32)
    ccall((:zbesk_,openspecfun), Cvoid,
          (Ref{Float64}, Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32},
           Ref{Float64}, Ref{Float64}, Ref{Int32}, Ref{Int32}),
           real(z), imag(z), nu, kode, 1,
           ai1, ai2, ae1, ae2)
    if ae2[] == 0 || ae2[] == 3
    end
end
function besselix(nu::Float64, z::Complex{Float64})
    if nu < 0
        if isinteger(nu)
            return _besseli(-nu,z,Int32(2))
        end
    else
    end
    if nu < 0
        if isinteger(nu)
        end
    end
end
function besseljx(nu::Float64, z::Complex{Float64})
    if nu < 0
        if isinteger(nu)
        end
    end
end
""" """ function besselj(nu::Real, x::AbstractFloat)
    if isinteger(nu)
        if typemin(Cint) <= nu <= typemax(Cint)
        end
    end
    real(besselj(float(nu), complex(x)))
    if x < 0
    end
end
""" """ function bessely(nu::Real, x::AbstractFloat)
    if x < 0
        throw(DomainError(x, "`x` must be nonnegative."))
    end
end
""" """ function besselyx(nu::Real, x::AbstractFloat)
    if x < 0
        throw(DomainError(x, "`x` must be nonnegative."))
    end
    @eval begin
        function $bfn(nu::Real, z::Complex)
        end
        function $bfn(nu::Real, k::Integer, z::Complex)
        end
    end
end
