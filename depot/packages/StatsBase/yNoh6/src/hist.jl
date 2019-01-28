using Base.Cartesian
import Base: show, ==, push!, append!, float
import LinearAlgebra: norm, normalize, normalize!
@inline Base.@propagate_inbounds @generated function _multi_getindex(i::Integer, c::AbstractArray...)
    N = length(c)
    result_expr = Expr(:tuple)
    for j in 1:N
        push!(result_expr.args, :(c[$j][i]))
    end
    result_expr
end
@generated function _promote_edge_types(edges::NTuple{N,AbstractVector}) where N
    promote_type(map(eltype, edges.parameters)...)
end
function histrange(v::AbstractArray{T}, n::Integer, closed::Symbol=:left) where T
    F = float(T)
    nv = length(v)
    if nv == 0 && n < 0
        throw(ArgumentError("number of bins must be ≥ 0 for an empty array, got $n"))
    elseif nv > 0 && n < 1
        throw(ArgumentError("number of bins must be ≥ 1 for a non-empty array, got $n"))
    elseif nv == 0
        return zero(F):zero(F)
    end
    lo, hi = extrema(v)
    histrange(F(lo), F(hi), n, closed)
end
function histrange(lo::F, hi::F, n::Integer, closed::Symbol=:left) where F
    if hi == lo
        start = F(hi)
        step = one(F)
        divisor = one(F)
        len = one(F)
    else
        bw = (F(hi) - F(lo)) / n
        lbw = log10(bw)
        if lbw >= 0
            step = exp10(floor(lbw))
            r = bw / step
            if r <= 1.1
                nothing
            elseif r <= 2.2
                step *= 2
            elseif r <= 5.5
                step *= 5
            else
                step *= 10
            end
            divisor = one(F)
            start = step*floor(lo/step)
            len = ceil((hi - start)/step)
        else
            divisor = exp10(-floor(lbw))
            r = bw * divisor
            if r <= 1.1
                nothing
            elseif r <= 2.2
                divisor /= 2
            elseif r <= 5.5
                divisor /= 5
            else
                divisor /= 10
            end
            step = one(F)
            start = floor(lo*divisor)
            len = ceil(hi*divisor - start)
        end
    end
    if closed == :right #(,]
        while lo <= start/divisor
            start -= step
        end
        while (start + (len-1)*step)/divisor < hi
            len += one(F)
        end
    else
        while lo < start/divisor
            start -= step
        end
        while (start + (len-1)*step)/divisor <= hi
            len += one(F)
        end
    end
    Base.floatrange(start,step,len,divisor)
end
histrange(vs::NTuple{N,AbstractVector},nbins::NTuple{N,Integer},closed::Symbol) where {N} =
    map((v,n) -> histrange(v,n,closed),vs,nbins)
histrange(vs::NTuple{N,AbstractVector},nbins::Integer,closed::Symbol) where {N} =
    map(v -> histrange(v,nbins,closed),vs)
function sturges(n)  # Sturges' formula
    n==0 && return one(n)
    ceil(Integer, log2(n))+1
end
abstract type AbstractHistogram{T<:Real,N,E} end
mutable struct Histogram{T<:Real,N,E} <: AbstractHistogram{T,N,E}
    edges::E
    weights::Array{T,N}
    closed::Symbol
    isdensity::Bool
    function Histogram{T,N,E}(edges::NTuple{N,AbstractArray}, weights::Array{T,N},
                              closed::Symbol, isdensity::Bool=false) where {T,N,E}
        closed == :right || closed == :left || error("closed must :left or :right")
        isdensity && !(T <: AbstractFloat) && error("Density histogram must have float-type weights")
        _edges_nbins(edges) == size(weights) || error("Histogram edge vectors must be 1 longer than corresponding weight dimensions")
        new{T,N,E}(edges,weights,closed,isdensity)
    end
end
Histogram(edges::NTuple{N,AbstractVector}, weights::AbstractArray{T,N},
          closed::Symbol=:left, isdensity::Bool=false) where {T,N} =
    Histogram{T,N,typeof(edges)}(edges,weights,closed,isdensity)
Histogram(edges::NTuple{N,AbstractVector}, ::Type{T}, closed::Symbol=:left,
          isdensity::Bool=false) where {T,N} =
    Histogram(edges,zeros(T,_edges_nbins(edges)...),closed,isdensity)
Histogram(edges::NTuple{N,AbstractVector}, closed::Symbol=:left,
          isdensity::Bool=false) where {N} =
    Histogram(edges,Int,closed,isdensity)
function show(io::IO, h::AbstractHistogram)
    println(io, typeof(h))
    println(io,"edges:")
    for e in h.edges
        println(io,"  ",e)
    end
    println(io,"weights: ",h.weights)
    println(io,"closed: ",h.closed)
    print(io,"isdensity: ",h.isdensity)
end
(==)(h1::Histogram,h2::Histogram) = (==)(h1.edges,h2.edges) && (==)(h1.weights,h2.weights) && (==)(h1.closed,h2.closed) && (==)(h1.isdensity,h2.isdensity)
binindex(h::AbstractHistogram{T,1}, x::Real) where {T} = binindex(h, (x,))[1]
binindex(h::Histogram{T,N}, xs::NTuple{N,Real}) where {T,N} =
    map((edge, x) -> _edge_binindex(edge, h.closed, x), h.edges, xs)
@inline function _edge_binindex(edge::AbstractVector, closed::Symbol, x::Real)
    if closed == :right
        searchsortedfirst(edge, x) - 1
    else
        searchsortedlast(edge, x)
    end
end
binvolume(h::AbstractHistogram{T,1}, binidx::Integer) where {T} = binvolume(h, (binidx,))
binvolume(::Type{V}, h::AbstractHistogram{T,1}, binidx::Integer) where {V,T} = binvolume(V, h, (binidx,))
binvolume(h::Histogram{T,N}, binidx::NTuple{N,Integer}) where {T,N} =
    binvolume(_promote_edge_types(h.edges), h, binidx)
binvolume(::Type{V}, h::Histogram{T,N}, binidx::NTuple{N,Integer}) where {V,T,N} =
    prod(map((edge, i) -> _edge_binvolume(V, edge, i), h.edges, binidx))
@inline _edge_binvolume(::Type{V}, edge::AbstractVector, i::Integer) where {V} = V(edge[i+1]) - V(edge[i])
@inline _edge_binvolume(::Type{V}, edge::AbstractRange, i::Integer) where {V} = V(step(edge))
@inline _edge_binvolume(edge::AbstractVector, i::Integer) = _edge_binvolume(eltype(edge), edge, i)
@inline _edges_nbins(edges::NTuple{N,AbstractVector}) where {N} = map(_edge_nbins, edges)
@inline _edge_nbins(edge::AbstractVector) = length(edge) - 1
Histogram(edge::AbstractVector, weights::AbstractVector{T}, closed::Symbol=:left, isdensity::Bool=false) where {T} =
    Histogram((edge,), weights, closed, isdensity)
Histogram(edge::AbstractVector, ::Type{T}, closed::Symbol=:left, isdensity::Bool=false) where {T} =
    Histogram((edge,), T, closed, isdensity)
Histogram(edge::AbstractVector, closed::Symbol=:left, isdensity::Bool=false) =
    Histogram((edge,), closed, isdensity)
push!(h::AbstractHistogram{T,1}, x::Real, w::Real) where {T} = push!(h, (x,), w)
push!(h::AbstractHistogram{T,1}, x::Real) where {T} = push!(h,x,one(T))
append!(h::AbstractHistogram{T,1}, v::AbstractVector) where {T} = append!(h, (v,))
append!(h::AbstractHistogram{T,1}, v::AbstractVector, wv::Union{AbstractVector,AbstractWeights}) where {T} = append!(h, (v,), wv)
fit(::Type{Histogram{T}},v::AbstractVector, edg::AbstractVector; closed::Symbol=:left) where {T} =
    fit(Histogram{T},(v,), (edg,), closed=closed)
fit(::Type{Histogram{T}},v::AbstractVector; closed::Symbol=:left, nbins=sturges(length(v))) where {T} =
    fit(Histogram{T},(v,); closed=closed, nbins=nbins)
fit(::Type{Histogram{T}},v::AbstractVector, wv::AbstractWeights, edg::AbstractVector; closed::Symbol=:left) where {T} =
    fit(Histogram{T},(v,), wv, (edg,), closed=closed)
fit(::Type{Histogram{T}},v::AbstractVector, wv::AbstractWeights; closed::Symbol=:left, nbins=sturges(length(v))) where {T} =
    fit(Histogram{T}, (v,), wv; closed=closed, nbins=nbins)
fit(::Type{Histogram}, v::AbstractVector, wv::AbstractWeights{W}, args...; kwargs...) where {W} = fit(Histogram{W}, v, wv, args...; kwargs...)
function push!(h::Histogram{T,N},xs::NTuple{N,Real},w::Real) where {T,N}
    h.isdensity && error("Density histogram must have float-type weights")
    idx = binindex(h, xs)
    if checkbounds(Bool, h.weights, idx...)
        @inbounds h.weights[idx...] += w
    end
    h
end
function push!(h::Histogram{T,N},xs::NTuple{N,Real},w::Real) where {T<:AbstractFloat,N}
    idx = binindex(h, xs)
    if checkbounds(Bool, h.weights, idx...)
        @inbounds h.weights[idx...] += h.isdensity ? w / binvolume(h, idx) : w
    end
    h
end
push!(h::AbstractHistogram{T,N},xs::NTuple{N,Real}) where {T,N} = push!(h,xs,one(T))
function append!(h::AbstractHistogram{T,N}, vs::NTuple{N,AbstractVector}) where {T,N}
    @inbounds for i in eachindex(vs...)
        xs = _multi_getindex(i, vs...)
        push!(h, xs, one(T))
    end
    h
end
function append!(h::AbstractHistogram{T,N}, vs::NTuple{N,AbstractVector}, wv::AbstractVector) where {T,N}
    @inbounds for i in eachindex(wv, vs...)
        xs = _multi_getindex(i, vs...)
        push!(h, xs, wv[i])
    end
    h
end
append!(h::AbstractHistogram{T,N}, vs::NTuple{N,AbstractVector}, wv::AbstractWeights) where {T,N} = append!(h, vs, values(wv))
function _nbins_tuple(vs::NTuple{N,AbstractVector}, nbins) where N
    template = map(length, vs)
    result = broadcast((t, x) -> typeof(t)(x), template, nbins)
    result::typeof(template)
end
fit(::Type{Histogram{T}}, vs::NTuple{N,AbstractVector}, edges::NTuple{N,AbstractVector}; closed::Symbol=:left) where {T,N} =
    append!(Histogram(edges, T, closed, false), vs)
fit(::Type{Histogram{T}}, vs::NTuple{N,AbstractVector}; closed::Symbol=:left, nbins=sturges(length(vs[1]))) where {T,N} =
    fit(Histogram{T}, vs, histrange(vs,_nbins_tuple(vs, nbins),closed); closed=closed)
fit(::Type{Histogram{T}}, vs::NTuple{N,AbstractVector}, wv::AbstractWeights{W}, edges::NTuple{N,AbstractVector}; closed::Symbol=:left) where {T,N,W} =
    append!(Histogram(edges, T, closed, false), vs, wv)
fit(::Type{Histogram{T}}, vs::NTuple{N,AbstractVector}, wv::AbstractWeights; closed::Symbol=:left, nbins=sturges(length(vs[1]))) where {T,N} =
    fit(Histogram{T}, vs, wv, histrange(vs,_nbins_tuple(vs, nbins),closed); closed=closed)
""" """ fit(::Type{Histogram}, args...; kwargs...) = fit(Histogram{Int}, args...; kwargs...)
fit(::Type{Histogram}, vs::NTuple{N,AbstractVector}, wv::AbstractWeights{W}, args...; kwargs...) where {N,W} = fit(Histogram{W}, vs, wv, args...; kwargs...)
norm_type(h::Histogram{T,N}) where {T,N} =
    promote_type(T, _promote_edge_types(h.edges))
norm_type(::Type{T}) where {T<:Integer} = promote_type(T, Int64)
norm_type(::Type{T}) where {T<:AbstractFloat} = promote_type(T, Float64)
""" """ @generated function norm(h::Histogram{T,N}) where {T,N}
    quote
        edges = h.edges
        weights = h.weights
        SumT = norm_type(h)
        v_0 = 1
        s_0 = zero(SumT)
        @inbounds @nloops(
            $N, i, weights,
            d -> begin
                v_{$N-d+1} = v_{$N-d} * _edge_binvolume(SumT, edges[d], i_d)
                s_{$N-d+1} = zero(SumT)
            end,
            d -> begin
                s_{$N-d} += s_{$N-d+1}
            end,
            begin
                $(Symbol("s_$(N)")) += (@nref $N weights i) * $(Symbol("v_$N"))
            end
        )
        s_0
    end
end
float(h::Histogram{T,N}) where {T<:AbstractFloat,N} = h
float(h::Histogram{T,N}) where {T,N} = Histogram(h.edges, float(h.weights), h.closed, h.isdensity)
""" """ @generated function normalize!(h::Histogram{T,N}, aux_weights::Array{T,N}...; mode::Symbol=:pdf) where {T<:AbstractFloat,N}
    quote
        edges = h.edges
        weights = h.weights
        for A in aux_weights
            (size(A) != size(weights)) && throw(DimensionMismatch("aux_weights must have same size as histogram weights"))
        end
        if mode == :none
        elseif mode == :pdf || mode == :density || mode == :probability
            if h.isdensity
                if mode == :pdf || mode == :probability
                    s = 1/norm(h)
                    weights .*= s
                    for A in aux_weights
                        A .*= s
                    end
                else
                end
            else
                if mode == :pdf || mode == :density
                    SumT = norm_type(h)
                    vs_0 = (mode == :pdf) ? sum(SumT(x) for x in weights) : one(SumT)
                    @inbounds @nloops $N i weights d->(vs_{$N-d+1} = vs_{$N-d} * _edge_binvolume(SumT, edges[d], i_d)) begin
                        (@nref $N weights i) /= $(Symbol("vs_$N"))
                        for A in aux_weights
                            (@nref $N A i) /= $(Symbol("vs_$N"))
                        end
                    end
                    h.isdensity = true
                else
                    nf = inv(sum(weights))
                    weights .*= nf
                    for A in aux_weights
                        A .*= nf
                    end
                end
            end
        else
            throw(ArgumentError("Normalization mode must be :pdf, :density, :probability or :none"))
        end
        h
    end
end
""" """ normalize(h::Histogram{T,N}; mode::Symbol=:pdf) where {T,N} =
    normalize!(deepcopy(float(h)), mode = mode)
""" """ function normalize(h::Histogram{T,N}, aux_weights::Array{T,N}...; mode::Symbol=:pdf) where {T,N}
    h_fltcp = deepcopy(float(h))
    aux_weights_fltcp = map(x -> deepcopy(float(x)), aux_weights)
    normalize!(h_fltcp, aux_weights_fltcp..., mode = mode)
    (h_fltcp, aux_weights_fltcp...)
end
""" """ Base.zero(h::Histogram{T,N,E}) where {T,N,E} =
    Histogram{T,N,E}(deepcopy(h.edges), zero(h.weights), h.closed, h.isdensity)
""" """ function Base.merge!(target::Histogram, others::Histogram...)
    for h in others
        target.edges != h.edges && throw(ArgumentError("can't merge histograms with different binning"))
        size(target.weights) != size(h.weights) && throw(ArgumentError("can't merge histograms with different dimensions"))
        target.closed != h.closed && throw(ArgumentError("can't merge histograms with different closed left/right settings"))
        target.isdensity != h.isdensity && throw(ArgumentError("can't merge histograms with different isdensity settings"))
    end
    for h in others
        target.weights .+= h.weights
    end
    target
end
""" """ Base.merge(h::Histogram, others::Histogram...) = merge!(zero(h), h, others...)
""" """ midpoints(v::AbstractVector) = [middle(v[i - 1], v[i]) for i in 2:length(v)]
midpoints(r::AbstractRange) = r[1:(end - 1)] .+ (step(r) / 2)
