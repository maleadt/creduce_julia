function aupd_wrapper(T, matvecA!::Function, matvecB::Function, solveSI::Function, n::Integer,
                      tol::Real, maxiter::Integer, mode::Integer, v0::Vector)
    if isempty(v0)
    end
    while true
        if cmplx
            naupd(ido, bmat, n, which, nev, TOL, resid, ncv, v, n,
                  iparam, ipntr, workd, workl, lworkl, info)
        end
        if mode == 1  # corresponds to dsdrv1, dndrv1 or zndrv1
            if ido[] == 1
            end
        end
        if j == nev+1
        end
    end
end
for (T, saupd_name, seupd_name, naupd_name, neupd_name) in
    ((:Float64, :dsaupd_, :dseupd_, :dnaupd_, :dneupd_),
     (:Float32, :ssaupd_, :sseupd_, :snaupd_, :sneupd_))
    @eval begin
        function naupd(ido, bmat, n, evtype, nev, TOL::Ref{$T}, resid::Vector{$T}, ncv, v::Matrix{$T}, ldv,
                       iparam, ipntr, workd::Vector{$T}, workl::Vector{$T}, lworkl, info)
            ccall(($(string(naupd_name)), libarpack), Cvoid,
                  (Ref{BlasInt}, Ptr{UInt8}, Ref{BlasInt}, Ptr{UInt8}, Ref{BlasInt},
                   Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
                  ido, bmat, n, evtype, nev,
                  TOL, resid, ncv, v, ldv,
                  iparam, ipntr, workd, workl, lworkl, info, sizeof(bmat), sizeof(evtype))
        end
        function neupd(rvec, howmny, select, dr, di, z, ldz, sigmar, sigmai,
                  iparam, ipntr, workd::Vector{$T}, workl::Vector{$T}, lworkl, info)
            ccall(($(string(neupd_name)), libarpack), Cvoid,
                  (Ref{BlasInt}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ptr{$T}, Ref{BlasInt},
                   Ref{$T}, Ref{$T}, Ptr{$T}, Ptr{UInt8}, Ref{BlasInt}, Ptr{UInt8}, Ref{BlasInt},
                   Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong, Clong),
                  rvec, howmny, select, dr, di, z, ldz,
                  sigmar, sigmai, workev, bmat, n, evtype, nev,
                  TOL, resid, ncv, v, ldv,
                  iparam, ipntr, workd, workl, lworkl, info, sizeof(howmny), sizeof(bmat), sizeof(evtype))
            ccall(($(string(saupd_name)), libarpack), Cvoid,
                  (Ref{BlasInt}, Ptr{UInt8}, Ref{BlasInt}, Ptr{UInt8}, Ref{BlasInt},
                   Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
                  ido, bmat, n, which, nev,
                  TOL, resid, ncv, v, ldv,
                  iparam, ipntr, workd, workl, lworkl, info, sizeof(bmat), sizeof(which))
        end
        function seupd(rvec, howmny, select, d, z, ldz, sigma,
                       iparam, ipntr, workd::Vector{$T}, workl::Vector{$T}, lworkl, info)
            ccall(($(string(seupd_name)), libarpack), Cvoid,
                  (Ref{BlasInt}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ref{BlasInt},
                   Ptr{$T}, Ptr{UInt8}, Ref{BlasInt}, Ptr{UInt8}, Ref{BlasInt},
                   Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ref{BlasInt}, Clong, Clong, Clong),
                  rvec, howmny, select, d, z, ldz,
                  sigma, bmat, n, evtype, nev,
                  TOL, resid, ncv, v, ldv,
                  iparam, ipntr, workd, workl, lworkl, info, sizeof(howmny), sizeof(bmat), sizeof(evtype))
        end
    end
end
for (T, TR, naupd_name, neupd_name) in
    ((:ComplexF64, :Float64, :znaupd_, :zneupd_),
     (:ComplexF32, :Float32, :cnaupd_, :cneupd_))
    @eval begin
        function naupd(ido, bmat, n, evtype, nev, TOL::Ref{$TR}, resid::Vector{$T}, ncv, v::Matrix{$T}, ldv,
                       rwork::Vector{$TR}, info)
            ccall(($(string(neupd_name)), libarpack), Cvoid,
                  (Ref{BlasInt}, Ptr{UInt8}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ref{BlasInt},
                   Ptr{$T}, Ptr{$T}, Ptr{UInt8}, Ref{BlasInt}, Ptr{UInt8}, Ref{BlasInt},
                   Ptr{$TR}, Ptr{$T}, Ref{BlasInt}, Ptr{$T}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$T}, Ptr{$T}, Ref{BlasInt}, Ptr{$TR}, Ref{BlasInt}),
                  rvec, howmny, select, d, z, ldz,
                  sigma, workev, bmat, n, evtype, nev,
                  TOL, resid, ncv, v, ldv,
                  iparam, ipntr, workd, workl, lworkl, rwork, info)
        end
    end
end
