__precompile__()
module Compat
using Base.Meta
export @compat
withincurly(ex) = isexpr(ex, :curly) ? ex.args[1] : ex
is_index_style(ex::Expr) = ex == :(Compat.IndexStyle) || ex == :(Base.IndexStyle) ||
    (ex.head == :(.) && (ex.args[1] == :Compat || ex.args[1] == :Base) &&
         ex.args[2] == Expr(:quote, :IndexStyle))
is_index_style(arg) = false
istopsymbol(ex, mod, sym) = ex in (sym, Expr(:(.), mod, Expr(:quote, sym)))
if VERSION < v"0.6.0-dev.2782"
    function new_style_typealias(ex::ANY)
        isexpr(ex, :(=)) || return false
        ex = ex::Expr
        return length(ex.args) == 2 && isexpr(ex.args[1], :curly)
    end
else
    new_style_typealias(ex) = false
end
function _compat(ex::Expr)
    if ex.head === :call
        f = ex.args[1]
    elseif ex.head === :curly
        f = ex.args[1]
        if VERSION < v"0.6.0-dev.2575" #20414
            ex.args[2], ex.args[3] = ex.args[3], ex.args[2]
        end
        return true
    end
    return name, body
end
function _compat_primitive(typedecl)
    name, body = _get_typebody(typedecl)
    if length(body) != 1
        throw(ArgumentError("Invalid primitive type declaration: $typedecl"))
    end
    return Expr(:bitstype, body[1], name)
end
function _compat_abstract(typedecl)
    name, body = _get_typebody(typedecl)
    if length(body) != 0
        throw(ArgumentError("Invalid abstract type declaration: $typedecl"))
    end
    return Expr(:abstract, name)
end
macro compat(ex...)
    if VERSION < v"0.6.0-dev.2746" && length(ex) == 2 && ex[1] === :primitive
        return esc(_compat_primitive(ex[2]))
    elseif length(ex) != 1
        throw(ArgumentError("@compat called with wrong number of arguments: $ex"))
    elseif (VERSION < v"0.6.0-dev.2746" && isexpr(ex[1], :abstract) &&
            length(ex[1].args) == 1 && isexpr(ex[1].args[1], :type))
        # This can in principle be handled in nested case but we do not
        # do that to be consistent with primitive types.
        return esc(_compat_abstract(ex[1].args[1]))
    end
    esc(_compat(ex[1]))
end
@static if !isdefined(Base, :devnull) #25959
    export devnull, stdout, stdin, stderr
    const devnull = DevNull
    for f in (:stdout, :stdin, :stderr)
        F = Symbol(uppercase(string(f)))
        rf = Symbol(string("_redirect_", f))
        @eval begin
            $f = $F
            # overload internal _redirect_std* functions
            # so that they change Compat.std*
            function Base.$rf(stream::IO)
                ret = invoke(Base.$rf, Tuple{Any}, stream)
                global $f = $F
                return ret
            end
        end
    end
    # in __init__ because these can't be saved during precompiling
    function __init__()
        global stdout = STDOUT
        global stdin = STDIN
        global stderr = STDERR
    end
    Base.include_string(mod::Module, code::String, fname::String) =
        eval(mod, :(include_string($code, $fname)))
    Base.include_string(mod::Module, code::AbstractString, fname::AbstractString="string") =
        eval(mod, :(include_string($code, $fname)))
end
if VERSION < v"0.6.0-dev.2042"
    include_string(@__MODULE__, """
        immutable ExponentialBackOff
            n::Int
            first_delay::Float64
            max_delay::Float64
            factor::Float64
            jitter::Float64
            function ExponentialBackOff(n, first_delay, max_delay, factor, jitter)
                all(x->x>=0, (n, first_delay, max_delay, factor, jitter)) || error("all inputs must be non-negative")
                new(n, first_delay, max_delay, factor, jitter)
            end
        end
    """)
    """
        ExponentialBackOff(; n=1, first_delay=0.05, max_delay=10.0, factor=5.0, jitter=0.1)
    A [`Float64`](@ref) iterator of length `n` whose elements exponentially increase at a
    rate in the interval `factor` * (1 ± `jitter`).  The first element is
    `first_delay` and all elements are clamped to `max_delay`.
    """
    ExponentialBackOff(; n=1, first_delay=0.05, max_delay=10.0, factor=5.0, jitter=0.1) =
        ExponentialBackOff(n, first_delay, max_delay, factor, jitter)
    Base.start(ebo::ExponentialBackOff) = (ebo.n, min(ebo.first_delay, ebo.max_delay))
    function Base.next(ebo::ExponentialBackOff, state)
        next_n = state[1]-1
        curr_delay = state[2]
        next_delay = min(ebo.max_delay, state[2] * ebo.factor * (1.0 - ebo.jitter + (rand() * 2.0 * ebo.jitter)))
        (curr_delay, (next_n, next_delay))
    end
    Base.done(ebo::ExponentialBackOff, state) = state[1]<1
    Base.length(ebo::ExponentialBackOff) = ebo.n
    export @__DIR__
end
if VERSION < v"0.7.0-DEV.1211"
    macro dep_vectorize_1arg(S, f)
        S = esc(S)
        f = esc(f)
        T = esc(:T)
        x = esc(:x)
        AbsArr = esc(:AbstractArray)
        ## Depwarn to be enabled when 0.5 support is dropped.
        # depwarn("Implicit vectorized function is deprecated in favor of compact broadcast syntax.",
        #         Symbol("@dep_vectorize_1arg"))
        :(@deprecate $f{$T<:$S}($x::$AbsArr{$T}) @compat($f.($x)))
    end
    macro dep_vectorize_2arg(S, f)
        S = esc(S)
        f = esc(f)
        T1 = esc(:T1)
        T2 = esc(:T2)
        x = esc(:x)
        y = esc(:y)
        AbsArr = esc(:AbstractArray)
        ## Depwarn to be enabled when 0.5 support is dropped.
        # depwarn("Implicit vectorized function is deprecated in favor of compact broadcast syntax.",
        #         Symbol("@dep_vectorize_2arg"))
        quote
            @deprecate $f{$T1<:$S}($x::$S, $y::$AbsArr{$T1}) @compat($f.($x,$y))
            @deprecate $f{$T1<:$S}($x::$AbsArr{$T1}, $y::$S) @compat($f.($x,$y))
            @deprecate $f{$T1<:$S,$T2<:$S}($x::$AbsArr{$T1}, $y::$AbsArr{$T2}) @compat($f.($x,$y))
        end
    end
else
    macro dep_vectorize_1arg(S, f)
        AbstractArray = GlobalRef(Base, :AbstractArray)
        return esc(:(@deprecate $f(x::$AbstractArray{T}) where {T<:$S} $f.(x)))
    end
    macro dep_vectorize_2arg(S, f)
        AbstractArray = GlobalRef(Base, :AbstractArray)
        return esc(quote
            @deprecate $f(x::$S, y::$AbstractArray{T1}) where {T1<:$S} $f.(x, y)
            @deprecate $f(x::$AbstractArray{T1}, y::$S) where {T1<:$S} $f.(x, y)
            @deprecate $f(x::$AbstractArray{T1}, y::$AbstractArray{T2}) where {T1<:$S, T2<:$S} $f.(x, y)
        end)
    end
end
@static if VERSION < v"0.6.0-dev.693"
    Base.Broadcast.broadcast{N}(f, t::NTuple{N}, ts::Vararg{NTuple{N}}) = map(f, t, ts...)
end
@static if !isdefined(Base, :xor)
    # 0.6
    const xor = $
    const ⊻ = xor
    export xor, ⊻
end
@static if !isdefined(Base, :numerator)
    # 0.6
    const numerator = num
    const denominator = den
    export numerator, denominator
end
@static if !isdefined(Base, :iszero)
    # 0.6
    iszero(x) = x == zero(x)
    iszero(x::Number) = x == 0
    iszero(x::AbstractArray) = all(iszero, x)
    export iszero
end
@static if !isdefined(Base, :(>:))
    # 0.6
    const >: = let
        _issupertype(a::ANY, b::ANY) = issubtype(b, a)
    end
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
    const Markdown = Base.Markdown
else
    import Markdown
end
if VERSION < v"0.7.0-DEV.2609"
    @eval module SuiteSparse
        if Base.USE_GPL_LIBS
            using Compat.SparseArrays: CHOLMOD, SPQR, UMFPACK
        end
        using Compat.SparseArrays: increment, increment!, decrement, decrement!
    end
else
    import SuiteSparse
end
@static if VERSION < v"0.7.0-DEV.3500"
else
    import Serialization
end
@static if VERSION < v"0.7.0-DEV.4592"
    struct Fix2{F,T} <: Function
        f::F
        x::T
        Fix2(f::F, x::T) where {F,T} = new{F,T}(f, x)
        Fix2(f::Type{F}, x::T) where {F,T} = new{F,T}(f, x)
    end
    (f::Fix2)(y) = f.f(y, f.x)
    Base.:(==)(x) = Fix2(==, x)
    @static if VERSION >= v"0.7.0-DEV.1993"
        Base.isequal(x) = Base.equalto(x)
    else
        Base.isequal(x) = Fix2(isequal, x)
    end
    @static if VERSION >= v"0.7.0-DEV.3272"
        Base.in(x) = Base.occursin(x)
    else
        Base.in(x) = Fix2(in, x)
    end
else
    import Base: Fix2
end
@static if VERSION < v"0.7.0-DEV.1993"
    const EqualTo{T} = Fix2{typeof(isequal),T}
    export ComplexF64
end
module Unicode
    export graphemes, textwidth, isvalid,
           islower, isupper, isalpha, isdigit, isxdigit, isnumeric, isalnum,
           iscntrl, ispunct, isspace, isprint, isgraph,
           lowercase, uppercase, titlecase, lcfirst, ucfirst
    if VERSION < v"0.7.0-DEV.2915"
        # 0.7.0-DEV.1930
        if !isdefined(Base, :titlecase)
            titlecase(c::Char) = isascii(c) ? ('a' <= c <= 'z' ? c - 0x20 : c) :
                Char(ccall(:utf8proc_totitle, UInt32, (UInt32,), c))
            function titlecase(s::AbstractString)
                startword = true
                b = IOBuffer()
                for c in s
                    if isspace(c)
                        print(b, c)
                        startword = true
                    else
                        print(b, startword ? titlecase(c) : c)
                        startword = false
                    end
                end
                return String(take!(b))
            end
        end
    else
        using Unicode
        import Unicode: isassigned, normalize # not exported from Unicode module due to conflicts
    end
end
@static if !isdefined(Base, :AbstractDict)
    const AbstractDict = Associative
    export AbstractDict
end
@static if !isdefined(Base, :axes)
    const axes = Base.indices
    # NOTE: Intentionally not exported to avoid conflicts with AxisArrays
    #export axes
end
@static if !isdefined(Base, :Nothing)
    const Nothing = Void
    const Cvoid = Void
    export Nothing, Cvoid
end
@static if !isdefined(Base, :Some)
    import Base: promote_rule, convert
    if VERSION >= v"0.6.0"
        include_string(@__MODULE__, """
            struct Some{T}
                value::T
            end
            promote_rule(::Type{Some{S}}, ::Type{Some{T}}) where {S,T} = Some{promote_type(S, T)}
            end
            promote_rule{S,T}(::Type{Some{S}}, ::Type{Some{T}}) = Some{promote_type(S, T)}
            promote_rule{T}(::Type{Some{T}}, ::Type{Nothing}) = Union{Some{T}, Nothing}
            convert{T}(::Type{Some{T}}, x::Some) = Some{T}(convert(T, x.value))
            convert{T}(::Type{Union{Some{T}, Nothing}}, x::Some) = convert(Some{T}, x)
            convert{T}(::Type{Union{T, Nothing}}, x::Any) = convert(T, x)
        """)
    end
    convert(::Type{Nothing}, x::Any) = throw(MethodError(convert, (Nothing, x)))
    export pushfirst!, popfirst!
end
@static if VERSION < v"0.7.0-DEV.3309"
    const IteratorSize = Base.iteratorsize
    const IteratorEltype = Base.iteratoreltype
else
    const IteratorSize = Base.IteratorSize
    const IteratorEltype = Base.IteratorEltype
end
@static if !isdefined(Base, :invpermute!)
    # AbstractArray implementation
    Base.IndexStyle(::Type{LinearIndices{N,R}}) where {N,R} = IndexCartesian()
    Compat.axes(iter::LinearIndices{N,R}) where {N,R} = iter.indices
    Base.size(iter::LinearIndices{N,R}) where {N,R} = length.(iter.indices)
    @inline function Base.getindex(iter::LinearIndices{N,R}, I::Vararg{Int, N}) where {N,R}
        dims = length.(iter.indices)
        #without the inbounds, this is slower than Base._sub2ind(iter.indices, I...)
        @inbounds result = reshape(1:Base.prod(dims), dims)[(I .- first.(iter.indices) .+ 1)...]
        return result
    end
elseif VERSION < v"0.7.0-DEV.3395"
    Base.size(iter::LinearIndices{N,R}) where {N,R} = length.(iter.indices)
end
@static if !isdefined(Base, Symbol("@info"))
    macro info(msg, args...)
        return :(info($(esc(msg)), prefix = "Info: "))
    end
    ceil(x, digits; base = base) = Base.ceil(x, digits, base)
    round(x, digits; base = base) = Base.round(x, digits, base)
    signif(x, digits; base = base) = Base.signif(x, digits, base)
end
if VERSION < v"0.7.0-DEV.3734"
    if isdefined(Base, :open_flags)
        import Base.open_flags
    else
        # copied from Base:
        function open_flags(; read=nothing, write=nothing, create=nothing, truncate=nothing, append=nothing)
            if write === true && read !== true && append !== true
                create   === nothing && (create   = true)
                truncate === nothing && (truncate = true)
            end
            if truncate === true || append === true
                write  === nothing && (write  = true)
                create === nothing && (create = true)
            end
            buf.data[:] = 0
        end
        if flags[4] # flags.truncate
            buf.size = 0
        end
        return buf
        for (val, ind) in zip(b, inds)
            get!(bdict, val, ind)
        end
        return Union{eltype(inds), Nothing}[
             get(bdict, i, nothing) for i in a
         ]
    end
else
    const indexin = Base.indexin
end
if VERSION < v"0.7.0-DEV.4585"
    export isuppercase, islowercase, uppercasefirst, lowercasefirst
    const isuppercase = isupper
    const islowercase = islower
    const uppercasefirst = ucfirst
    const lowercasefirst = lcfirst
end
if VERSION < v"0.7.0-DEV.4064"
    for f in (:mean, :cumsum, :cumprod, :sum, :prod, :maximum, :minimum, :all, :any, :median)
        @eval begin
            $f(a::AbstractArray; dims=nothing) =
                dims===nothing ? Base.$f(a) : Base.$f(a, dims)
        end
    end
    for f in (:sum, :prod, :maximum, :minimum, :all, :any, :accumulate)
        @eval begin
            $f(f, a::AbstractArray; dims=nothing) =
                dims===nothing ? Base.$f(f, a) : Base.$f(f, a, dims)
        end
    end
    for f in (:findmax, :findmin)
        @eval begin
            $f(a::AbstractVector; dims=nothing) =
                dims===nothing ? Base.$f(a) : Base.$f(a, dims)
            function $f(a::AbstractArray; dims=nothing)
                vs, inds = dims===nothing ? Base.$f(a) : Base.$f(a, dims)
                cis = CartesianIndices(a)
                return (vs, map(i -> cis[i], inds))
            end
        end
    end
    for f in (:var, :std, :sort)
        @eval begin
            $f(a::AbstractArray; dims=nothing, kwargs...) =
                dims===nothing ? Base.$f(a; kwargs...) : Base.$f(a, dims; kwargs...)
        end
    end
    for f in (:cumsum!, :cumprod!)
        @eval $f(out, a; dims=nothing) =
            dims===nothing ? Base.$f(out, a) : Base.$f(out, a, dims)
    end
end
if VERSION < v"0.7.0-DEV.4064"
    varm(A::AbstractArray, m; dims=nothing, kwargs...) =
        dims===nothing ? Base.varm(A, m; kwargs...) : Base.varm(A, m, dims; kwargs...)
    if VERSION < v"0.7.0-DEV.755"
        cov(a::AbstractMatrix; dims=1, corrected=true) = Base.cov(a, dims, corrected)
        cov(a::AbstractVecOrMat, b::AbstractVecOrMat; dims=1, corrected=true) =
            Base.cov(a, b, dims, corrected)
    else
        cov(a::AbstractMatrix; dims=nothing, kwargs...) =
            dims===nothing ? Base.cov(a; kwargs...) : Base.cov(a, dims; kwargs...)
        cov(a::AbstractVecOrMat, b::AbstractVecOrMat; dims=nothing, kwargs...) =
            dims===nothing ? Base.cov(a, b; kwargs...) : Base.cov(a, b, dims; kwargs...)
    end
    cor(a::AbstractMatrix; dims=nothing) = dims===nothing ? Base.cor(a) : Base.cor(a, dims)
    import Base.promote_eltype_op
    import Base.@irrational
    import Base.LinAlg.BLAS.@blasfunc
end
if VERSION < v"0.7.0-DEV.2915"
    const textwidth = Compat.Unicode.textwidth
    export textwidth
end
end # module Compat
