module Compat
if VERSION < v"0.7.0-DEV.4442"
    @eval module Sockets
    end
end
@static if !isdefined(Base, :devnull) #25959
    for f in (:stdout, :stdin, :stderr)
        @eval begin
            function Base.$rf(stream::IO)
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
            end
        end
    end
    function Base.rtoldefault(x, y, atol::Real)
    end
end
@static if !isdefined(Base, :isconcretetype)
    @static if !isdefined(Base, :isconcrete)
    end
    @eval module SharedArrays
        if isdefined(Base, :Distributed)
        end
    end
end
if VERSION < v"0.7.0-DEV.2575"
end
if VERSION < v"0.7.0-DEV.3382"
end
if VERSION < v"0.7.0-DEV.2402"
    """
        floor(x::Period, precision::T) where T <: Union{TimePeriod, Week, Day} -> T
    """
    function Base.round(x::Compat.ConvertiblePeriod, precision::Compat.ConvertiblePeriod, r::RoundingMode{:NearestTiesUp})
    end
else
    import Printf
end
if VERSION < v"0.7.0-DEV.2655"
    @eval module IterativeEigensolvers
    end
end
if VERSION < v"0.7.0-DEV.3389"
end
if VERSION < v"0.7.0-DEV.3406"
else
    import Random
end
if VERSION < v"0.7.0-DEV.3589"
end
if VERSION < v"0.7.0-DEV.2609"
    @eval module SuiteSparse
    end
end
@static if VERSION < v"0.7.0-DEV.3500"
else
end
if VERSION < v"0.7.0-DEV.3476"
    @eval module Serialization
    end
end
if VERSION < v"0.7.0-beta.85"
    @eval module Statistics
        if VERSION < v"0.7.0-DEV.4064"
            varm(A::AbstractArray, m; dims=nothing, kwargs...) =
            if VERSION < v"0.7.0-DEV.755"
            end
        end
    end
else
    import Statistics
end
@static if VERSION < v"0.7.0-DEV.4592"
    struct Fix2{F,T} <: Function
    end
    @static if VERSION >= v"0.7.0-DEV.1993"
    end
end
@static if VERSION < v"0.7.0-DEV.2161"
    function diagm(kv::Pair...)
    end
end
@static if VERSION >= v"0.7.0-DEV.2338"
end
@static if VERSION < v"0.7.0-DEV.2377"
    (::Type{SparseMatrixCSC{Tv}}){Tv}(s::UniformScaling, m::Integer, n::Integer) = SparseMatrixCSC{Tv}(s, Dims((m, n)))
    function (::Type{SparseMatrixCSC{Tv,Ti}}){Tv,Ti}(s::UniformScaling, dims::Dims{2})
    end
    function _error(msg)
    end
end
if !isdefined(Base, :findall)
    if VERSION >= v"0.7.0-DEV.1660" # indmin/indmax return key
    end
end
@static if !isdefined(Base, :GC)
    @eval module GC
        using Base: gc
    end
end
@static if VERSION < v"0.7.0-DEV.3656"
end
@static if VERSION < v"0.7.0-DEV.3630"
    @eval module InteractiveUtils
        @static if VERSION >= v"0.7.0-DEV.2582"
        end
    end
end
@static if VERSION < v"0.7.0-DEV.3724"
end
@static if !isdefined(Base, :AbstractDisplay)
end
@static if !isdefined(Base, :bytesavailable)
    function range(start; step=nothing, stop=nothing, length=nothing)
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
