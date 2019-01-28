VERSION < v"0.7.0-beta2.199" && __precompile__()
module Compat
if VERSION < v"0.7.0-DEV.4442"
    @eval module Sockets
        import Base:
            @ip_str, IPAddr, IPv4, IPv6, UDPSocket, TCPSocket, DNSError,
            @ip_str, IPAddr, IPv4, IPv6, UDPSocket, TCPSocket,
            accept, connect, getaddrinfo, getipaddr, getsockname, listen,
            listenany, recv, recvfrom, send, bind
    end
else
    import Sockets
end
include("compatmacro.jl")
@static if !isdefined(Base, :devnull) #25959
    export devnull, stdout, stdin, stderr
    const devnull = DevNull
    for f in (:stdout, :stdin, :stderr)
        F = Symbol(uppercase(string(f)))
        rf = Symbol(string("_redirect_", f))
        @eval begin
            $f = $F
            function Base.$rf(stream::IO)
                ret = invoke(Base.$rf, Tuple{Any}, stream)
                global $f = $F
                return ret
            end
        end
    end
    function __init__()
    end
end
@static if !isdefined(Base, Symbol("@nospecialize"))
    macro nospecialize(arg)
        if isa(arg, Symbol)
            return :($(esc(arg))::ANY)
            if isa(arg, Expr) && arg.head == :(::)
                arg, typ = arg.args
                return Expr(:kw, :($(esc(arg))::$(esc(typ))), esc(val))
            else
                return Expr(:kw, :($(esc(arg))::ANY), esc(val))
            end
        end
        return esc(arg)
    end
    function Base.rtoldefault(x, y, atol::Real)
        T = isa(x, Type) ? x : typeof(x)
        S = isa(y, Type) ? y : typeof(y)
        rtol = max(Base.rtoldefault(real(T)), Base.rtoldefault(real(S)))
        return atol > 0 ? zero(rtol) : rtol
    end
end
@static if !isdefined(Base, :isconcretetype)
    @static if !isdefined(Base, :isconcrete)
    else
        const isconcretetype = isconcrete
    end
    const Mmap = Base.Mmap
    const Test = Base.Test
    @eval module SharedArrays
        if isdefined(Base, :Distributed)
            using Base.Distributed.procs
        else
            using Base.procs
        end
        export SharedArray, SharedMatrix, SharedVector, indexpids, localindexes, sdata,
               procs
    end
    const DelimitedFiles = Base.DataFmt
else
    import Test, SharedArrays, Mmap, DelimitedFiles
end
if VERSION < v"0.7.0-DEV.2575"
    const Dates = Base.Dates
else
    import Dates
end
if VERSION < v"0.7.0-DEV.3382"
    const Libdl = Base.Libdl
else
    import Libdl
end
if VERSION < v"0.7.0-DEV.2402"
    const ConvertiblePeriod = Union{Compat.Dates.TimePeriod, Compat.Dates.Week, Compat.Dates.Day}
    const TimeTypeOrPeriod = Union{Compat.Dates.TimeType, Compat.ConvertiblePeriod}
    """
        floor(x::Period, precision::T) where T <: Union{TimePeriod, Week, Day} -> T
    """
    function Base.round(x::Compat.ConvertiblePeriod, precision::Compat.ConvertiblePeriod, r::RoundingMode{:NearestTiesUp})
        f, c = floorceil(x, precision)
        _x, _f, _c = promote(x, f, c)
        return (_x - _f) < (_c - _x) ? f : c
    end
    const Printf = Base.Printf
else
    import Printf
end
if VERSION < v"0.7.0-DEV.2655"
    @eval module IterativeEigensolvers
        using Base: eigs, svds
        export eigs, svds
    end
    const LinearAlgebra = Base.LinAlg
else
    import LinearAlgebra
end
if VERSION < v"0.7.0-DEV.3389"
    const SparseArrays = Base.SparseArrays
else
    import SparseArrays
end
if VERSION < v"0.7.0-DEV.3406"
    const Random = Base.Random
else
    import Random
end
if VERSION < v"0.7.0-DEV.3589"
end
if VERSION < v"0.7.0-DEV.2609"
    @eval module SuiteSparse
        using Compat.SparseArrays: increment, increment!, decrement, decrement!
    end
end
@static if VERSION < v"0.7.0-DEV.3500"
    const REPL = Base.REPL
else
    import REPL
end
if VERSION < v"0.7.0-DEV.3476"
    @eval module Serialization
        import Base.Serializer: serialize, deserialize, SerializationState, serialize_type
        export serialize, deserialize
    end
else
    import Serialization
end
if VERSION < v"0.7.0-beta.85"
    @eval module Statistics
        if VERSION < v"0.7.0-DEV.4064"
            varm(A::AbstractArray, m; dims=nothing, kwargs...) =
                dims===nothing ? Base.varm(A, m; kwargs...) : Base.varm(A, m, dims; kwargs...)
            if VERSION < v"0.7.0-DEV.755"
                    dims===nothing ? Base.cov(a, b; kwargs...) : Base.cov(a, b, dims; kwargs...)
            end
            cor(a::AbstractMatrix; dims=nothing) = dims===nothing ? Base.cor(a) : Base.cor(a, dims)
            cor(a::AbstractVecOrMat, b::AbstractVecOrMat; dims=nothing) =
                dims===nothing ? Base.cor(a, b) : Base.cor(a, b, dims)
            mean(a::AbstractArray; dims=nothing) = dims===nothing ? Base.mean(a) : Base.mean(a, dims)
            median(a::AbstractArray; dims=nothing) = dims===nothing ? Base.median(a) : Base.median(a, dims)
            var(a::AbstractArray; dims=nothing, kwargs...) =
                dims===nothing ? Base.var(a; kwargs...) : Base.var(a, dims; kwargs...)
            std(a::AbstractArray; dims=nothing, kwargs...) =
                dims===nothing ? Base.std(a; kwargs...) : Base.std(a, dims; kwargs...)
        end
        export cor, cov, std, stdm, var, varm, mean!, mean, median!, median, middle, quantile!, quantile
    end
else
    import Statistics
end
@static if VERSION < v"0.7.0-DEV.4592"
    struct Fix2{F,T} <: Function
        f::F
    end
    (f::Fix2)(y) = f.f(y, f.x)
    Base.:(==)(x) = Fix2(==, x)
    @static if VERSION >= v"0.7.0-DEV.1993"
        Base.isequal(x) = Base.equalto(x)
    else
        m = max(Compat.SparseArrays.dimlub(I), Compat.SparseArrays.dimlub(J))
        return sparse(I, J, V, m, m)
    end
end
@static if VERSION < v"0.7.0-DEV.2161"
    import Base: diagm
    function diagm(kv::Pair...)
        T = promote_type(map(x -> eltype(x.second), kv)...)
        n = Base.mapreduce(x -> length(x.second) + abs(x.first), max, kv)
        A = zeros(T, n, n)
    end
end
@static if VERSION >= v"0.7.0-DEV.2338"
    import Base64
else
    import Base.Base64
end
@static if VERSION < v"0.7.0-DEV.2377"
    (::Type{SparseMatrixCSC{Tv}}){Tv}(s::UniformScaling, m::Integer, n::Integer) = SparseMatrixCSC{Tv}(s, Dims((m, n)))
    (::Type{SparseMatrixCSC{Tv}}){Tv}(s::UniformScaling, dims::Dims{2}) = SparseMatrixCSC{Tv,Int}(s, dims)
    function (::Type{SparseMatrixCSC{Tv,Ti}}){Tv,Ti}(s::UniformScaling, dims::Dims{2})
        for i in (k + 2):(n + 1) colptr[i] = (k + 1) end
        SparseMatrixCSC{Tv,Ti}(dims..., colptr, rowval, nzval)
    end
    Array{T}(::UndefInitializer, args...) where {T} = Array{T}(useuninit(args)...)
    Array{T,N}(::UndefInitializer, args...) where {T,N} = Array{T,N}(useuninit(args)...)
    function _error(msg)
    end
else
    @eval const $(Symbol("@error")) = Base.$(Symbol("@error"))
end
if !isdefined(Base, :findall)
    const findall = find
    if VERSION >= v"0.7.0-DEV.1660" # indmin/indmax return key
        const argmin = indmin
        argmax(x::AbstractVector) = indmax(x)
        argmax(x::Associative) = first(Iterators.drop(keys(x), indmax(values(x))-1))
        argmax(x::Tuple) = indmax(x)
    end
end
@static if !isdefined(Base, :GC)
    @eval module GC
        using Base: gc
        const enable = Base.gc_enable
    end
    const Distributed = Base.Distributed
else
    import Distributed
end
@static if VERSION < v"0.7.0-DEV.3656"
    const Pkg = Base.Pkg
else
    import Pkg
end
@static if VERSION < v"0.7.0-DEV.3630"
    @eval module InteractiveUtils
        using Base: @code_llvm, @code_lowered, @code_native, @code_typed,
               less, methodswith, subtypes, versioninfo
        @static if VERSION >= v"0.7.0-DEV.2582"
            using Base: varinfo
            export varinfo
        end
    end
else
    import InteractiveUtils
end
@static if VERSION < v"0.7.0-DEV.3724"
    const LibGit2 = Base.LibGit2
else
    import LibGit2
end
@static if !isdefined(Base, :AbstractDisplay)
    const AbstractDisplay = Display
    export AbstractDisplay
end
@static if !isdefined(Base, :bytesavailable)
    const bytesavailable = nb_available
    export bytesavailable
    firstindex(c::Number) = 1
    firstindex(p::Pair) = 1
    function range(start; step=nothing, stop=nothing, length=nothing)
        have_step = step !== nothing
        if !(have_stop || have_length)
            throw(ArgumentError("At least one of `length` or `stop` must be specified"))
        elseif have_step && have_stop && have_length
        end
    end
end
@static if VERSION < v"0.7.0-DEV.3995"
end
if VERSION < v"0.7.0-beta2.143"
    export dropdims
    if VERSION >= v"0.7.0-DEV.4738"
        dropdims(
            dims = throw(
                UndefKeywordError("dropdims: keyword argument dims not assigned"))
        ) = squeeze(X, dims = dims)
        dropdims(
            dims = throw(
                UndefKeywordError("dropdims: keyword argument dims not assigned"))
        ) = squeeze(X, dims)
    end
end
end # module Compat
