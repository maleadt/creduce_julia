""" """ function geomean(a::RealArray)
    s = 0.0
    n = length(a)
    for i = 1 : n
        @inbounds s += log(a[i])
    end
    return exp(s / n)
end
""" """ function harmmean(a::RealArray)
    s = 0.0
    n = length(a)
    for i in 1 : n
        @inbounds s += inv(a[i])
    end
    return n / s
end
""" """ function genmean(a::RealArray, p::Real)
    if p == 0
        return geomean(a)
    end
    s = 0.0
    n = length(a)
    for x in a
        @inbounds s += convert(Float64, x)^p
    end
    return (s/n)^(1/p)
end
""" """ function mode(a::AbstractArray{T}, r::UnitRange{T}) where T<:Integer
    isempty(a) && error("mode: input array cannot be empty.")
    len = length(a)
    r0 = r[1]
    r1 = r[end]
    cnts = zeros(Int, length(r))
    mc = 0    # maximum count
    mv = r0   # a value corresponding to maximum count
    for i = 1:len
        @inbounds x = a[i]
        if r0 <= x <= r1
            @inbounds c = (cnts[x - r0 + 1] += 1)
            if c > mc
                mc = c
                mv = x
            end
        end
    end
    return mv
end
""" """ function modes(a::AbstractArray{T}, r::UnitRange{T}) where T<:Integer
    r0 = r[1]
    r1 = r[end]
    n = length(r)
    cnts = zeros(Int, n)
    mc = 0
    for i = 1:length(a)
        @inbounds x = a[i]
        if r0 <= x <= r1
            @inbounds c = (cnts[x - r0 + 1] += 1)
            if c > mc
                mc = c
            end
        end
    end
    ms = T[]
    for i = 1:n
        @inbounds if cnts[i] == mc
            push!(ms, r[i])
        end
    end
    return ms
end
function mode(a::AbstractArray{T}) where T
    isempty(a) && error("mode: input array cannot be empty.")
    cnts = Dict{T,Int}()
    mc = 1
    mv = a[1]
    cnts[mv] = 1
    for i = 2 : length(a)
        @inbounds x = a[i]
        if haskey(cnts, x)
            c = (cnts[x] += 1)
            if c > mc
                mc = c
                mv = x
            end
        else
            cnts[x] = 1
        end
    end
    return mv
end
function modes(a::AbstractArray{T}) where T
    isempty(a) && error("modes: input array cannot be empty.")
    cnts = Dict{T,Int}()
    mc = 1
    cnts[a[1]] = 1
    for i = 2 : length(a)
        @inbounds x = a[i]
        if haskey(cnts, x)
            c = (cnts[x] += 1)
            if c > mc
                mc = c
            end
        else
            cnts[x] = 1
        end
    end
    ms = T[]
    for (x, c) in cnts
        if c == mc
            push!(ms, x)
        end
    end
    return ms
end
""" """ percentile(v::AbstractArray{<:Real}, p) = quantile(v, p * 0.01)
quantile(v::AbstractArray{<:Real}) = quantile(v, [.0, .25, .5, .75, 1.0])
""" """ nquantile(v::AbstractArray{<:Real}, n::Integer) = quantile(v, (0:n)/n)
""" """ span(x::AbstractArray{<:Integer}) = ((a, b) = extrema(x); a:b)
""" """ variation(x::AbstractArray{<:Real}, m::Real) = stdm(x, m) / m
variation(x::AbstractArray{<:Real}) = variation(x, mean(x))
""" """ sem(a::AbstractArray{<:Real}) = sqrt(var(a) / length(a))
""" """ function mad(v::AbstractArray{T};
             center::Union{Real,Nothing}=nothing,
             normalize::Union{Bool, Nothing}=nothing) where T<:Real
    isempty(v) && throw(ArgumentError("mad is not defined for empty arrays"))
    S = promote_type(T, typeof(middle(first(v))))
    v2 = LinearAlgebra.copy_oftype(v, S)
    if normalize === nothing
        Base.depwarn("the `normalize` keyword argument will be false by default in future releases: set it explicitly to silence this deprecation", :mad)
        normalize = true
    end
    mad!(v2, center=center === nothing ? median!(v2) : center, normalize=normalize)
end
@irrational mad_constant 1.4826022185056018 BigFloat("1.482602218505601860547076529360423431326703202590312896536266275245674447622701")
""" """ function mad!(v::AbstractArray{<:Real};
              center::Real=median!(v),
              normalize::Union{Bool,Nothing}=true,
              constant=nothing)
    isempty(v) && throw(ArgumentError("mad is not defined for empty arrays"))
    v .= abs.(v .- center)
    m = median!(v)
    if normalize isa Nothing
        Base.depwarn("the `normalize` keyword argument will be false by default in future releases: set it explicitly to silence this deprecation", :mad)
        normalize = true
    end
    if !isa(constant, Nothing)
        Base.depwarn("keyword argument `constant` is deprecated, use `normalize` instead or apply the multiplication directly", :mad)
        m * constant
    elseif normalize
        m * mad_constant
    else
        m
    end
end
""" """ iqr(v::AbstractArray{<:Real}) = (q = quantile(v, [.25, .75]); q[2] - q[1])
""" """ genvar(X::AbstractMatrix) = size(X, 2) == 1 ? var(vec(X)) : det(cov(X))
genvar(itr) = var(itr)
""" """ totalvar(X::AbstractMatrix) = sum(var(X, dims=1))
totalvar(itr) = var(itr)
function _zscore!(Z::AbstractArray, X::AbstractArray, μ::Real, σ::Real)
    iσ = inv(σ)
    if μ == zero(μ)
        for i = 1 : length(X)
            @inbounds Z[i] = X[i] * iσ
        end
    else
        for i = 1 : length(X)
            @inbounds Z[i] = (X[i] - μ) * iσ
        end
    end
    return Z
end
@generated function _zscore!(Z::AbstractArray{S,N}, X::AbstractArray{T,N},
                             μ::AbstractArray, σ::AbstractArray) where {S,T,N}
    quote
        siz1 = size(X, 1)
        @nextract $N ud d->size(μ, d)
        if size(μ, 1) == 1 && siz1 > 1
            @nloops $N i d->(d>1 ? (1:size(X,d)) : (1:1)) d->(j_d = ud_d ==1 ? 1 : i_d) begin
                v = (@nref $N μ j)
                c = inv(@nref $N σ j)
                for i_1 = 1:siz1
                    (@nref $N Z i) = ((@nref $N X i) - v) * c
                end
            end
        else
            @nloops $N i X d->(j_d = ud_d ==1 ? 1 : i_d) begin
                (@nref $N Z i) = ((@nref $N X i) - (@nref $N μ j)) / (@nref $N σ j)
            end
        end
        return Z
    end
end
function _zscore_chksize(X::AbstractArray, μ::AbstractArray, σ::AbstractArray)
    size(μ) == size(σ) || throw(DimensionMismatch("μ and σ should have the same size."))
    for i=1:ndims(X)
        dμ_i = size(μ,i)
        (dμ_i == 1 || dμ_i == size(X,i)) || throw(DimensionMismatch("X and μ have incompatible sizes."))
    end
end
""" """ function zscore!(Z::AbstractArray{ZT}, X::AbstractArray{T}, μ::Real, σ::Real) where {ZT<:AbstractFloat,T<:Real}
    size(Z) == size(X) || throw(DimensionMismatch("Z and X must have the same size."))
    _zscore!(Z, X, μ, σ)
end
function zscore!(Z::AbstractArray{<:AbstractFloat}, X::AbstractArray{<:Real},
                 μ::AbstractArray{<:Real}, σ::AbstractArray{<:Real})
    size(Z) == size(X) || throw(DimensionMismatch("Z and X must have the same size."))
    _zscore_chksize(X, μ, σ)
    _zscore!(Z, X, μ, σ)
end
zscore!(X::AbstractArray{<:AbstractFloat}, μ::Real, σ::Real) = _zscore!(X, X, μ, σ)
zscore!(X::AbstractArray{<:AbstractFloat}, μ::AbstractArray{<:Real}, σ::AbstractArray{<:Real}) =
    (_zscore_chksize(X, μ, σ); _zscore!(X, X, μ, σ))
""" """ function zscore(X::AbstractArray{T}, μ::Real, σ::Real) where T<:Real
    ZT = typeof((zero(T) - zero(μ)) / one(σ))
    _zscore!(Array{ZT}(undef, size(X)), X, μ, σ)
end
function zscore(X::AbstractArray{T}, μ::AbstractArray{U}, σ::AbstractArray{S}) where {T<:Real,U<:Real,S<:Real}
    _zscore_chksize(X, μ, σ)
    ZT = typeof((zero(T) - zero(U)) / one(S))
    _zscore!(Array{ZT}(undef, size(X)), X, μ, σ)
end
zscore(X::AbstractArray{<:Real}) = ((μ, σ) = mean_and_std(X); zscore(X, μ, σ))
zscore(X::AbstractArray{<:Real}, dim::Int) = ((μ, σ) = mean_and_std(X, dim); zscore(X, μ, σ))
""" """ function entropy(p::AbstractArray{T}) where T<:Real
    s = zero(T)
    z = zero(T)
    for i = 1:length(p)
        @inbounds pi = p[i]
        if pi > z
            s += pi * log(pi)
        end
    end
    return -s
end
entropy(p::AbstractArray{<:Real}, b::Real) = entropy(p) / log(b)
""" """ function renyientropy(p::AbstractArray{T}, α::Real) where T<:Real
    α < 0 && throw(ArgumentError("Order of Rényi entropy not legal, $(α) < 0."))
    s = zero(T)
    z = zero(T)
    scale = sum(p)
    if α ≈ 0
        for i = 1:length(p)
            @inbounds pi = p[i]
            if pi > z
                s += 1
            end
        end
        s = log(s / scale)
    elseif α ≈ 1
        for i = 1:length(p)
            @inbounds pi = p[i]
            if pi > z
                s -= pi * log(pi)
            end
        end
        s = s / scale
    elseif (isinf(α))
        s = -log(maximum(p))
    else # a normal Rényi entropy
        for i = 1:length(p)
            @inbounds pi = p[i]
            if pi > z
                s += pi ^ α
            end
        end
        s = log(s / scale) / (1 - α)
    end
    return s
end
""" """ function crossentropy(p::AbstractArray{T}, q::AbstractArray{T}) where T<:Real
    length(p) == length(q) || throw(DimensionMismatch("Inconsistent array length."))
    s = 0.
    z = zero(T)
    for i = 1:length(p)
        @inbounds pi = p[i]
        @inbounds qi = q[i]
        if pi > z
            s += pi * log(qi)
        end
    end
    return -s
end
crossentropy(p::AbstractArray{T}, q::AbstractArray{T}, b::Real) where {T<:Real} =
    crossentropy(p,q) / log(b)
""" """ function kldivergence(p::AbstractArray{T}, q::AbstractArray{T}) where T<:Real
    length(p) == length(q) || throw(DimensionMismatch("Inconsistent array length."))
    s = 0.
    z = zero(T)
    for i = 1:length(p)
        @inbounds pi = p[i]
        @inbounds qi = q[i]
        if pi > z
            s += pi * log(pi / qi)
        end
    end
    return s
end
kldivergence(p::AbstractArray{T}, q::AbstractArray{T}, b::Real) where {T<:Real} =
    kldivergence(p,q) / log(b)
struct SummaryStats{T<:AbstractFloat}
    mean::T
    min::T
    q25::T
    median::T
    q75::T
    max::T
end
""" """ function summarystats(a::AbstractArray{T}) where T<:Real
    m = mean(a)
    qs = quantile(a, [0.00, 0.25, 0.50, 0.75, 1.00])
    R = typeof(convert(AbstractFloat, zero(T)))
    SummaryStats{R}(
        convert(R, m),
        convert(R, qs[1]),
        convert(R, qs[2]),
        convert(R, qs[3]),
        convert(R, qs[4]),
        convert(R, qs[5]))
end
function Base.show(io::IO, ss::SummaryStats)
    println(io, "Summary Stats:")
    @printf(io, "Mean:           %.6f\n", ss.mean)
    @printf(io, "Minimum:        %.6f\n", ss.min)
    @printf(io, "1st Quartile:   %.6f\n", ss.q25)
    @printf(io, "Median:         %.6f\n", ss.median)
    @printf(io, "3rd Quartile:   %.6f\n", ss.q75)
    @printf(io, "Maximum:        %.6f\n", ss.max)
end
""" """ describe(a::AbstractArray) = describe(stdout, a)
function describe(io::IO, a::AbstractArray{T}) where T<:Real
    show(io, summarystats(a))
    println(io, "Length:         $(length(a))")
    println(io, "Type:           $(string(eltype(a)))")
end
function describe(io::IO, a::AbstractArray)
    println(io, "Summary Stats:")
    println(io, "Length:         $(length(a))")
    println(io, "Type:           $(string(eltype(a)))")
    println(io, "Number Unique:  $(length(unique(a)))")
    return
end
