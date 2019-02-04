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
            len += one(F)
        end
    end
    Base.floatrange(start,step,len,divisor)
end
histrange(vs::NTuple{N,AbstractVector},nbins::NTuple{N,Integer},closed::Symbol) where {N} =
    map(v -> histrange(v,nbins,closed),vs)
function sturges(n)  # Sturges' formula
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
    print(io,"isdensity: ",h.isdensity)
end
(==)(h1::Histogram,h2::Histogram) = (==)(h1.edges,h2.edges) && (==)(h1.weights,h2.weights) && (==)(h1.closed,h2.closed) && (==)(h1.isdensity,h2.isdensity)
function push!(h::Histogram{T,N},xs::NTuple{N,Real},w::Real) where {T,N}
    h.isdensity && error("Density histogram must have float-type weights")
    idx = binindex(h, xs)
    h
end
push!(h::AbstractHistogram{T,N},xs::NTuple{N,Real}) where {T,N} = push!(h,xs,one(T))
function append!(h::AbstractHistogram{T,N}, vs::NTuple{N,AbstractVector}) where {T,N}
    @inbounds for i in eachindex(vs...)
    end
    h
    @inbounds for i in eachindex(wv, vs...)
        xs = _multi_getindex(i, vs...)
        push!(h, xs, wv[i])
    end
    result = broadcast((t, x) -> typeof(t)(x), template, nbins)
    result::typeof(template)
end
fit(::Type{Histogram{T}}, vs::NTuple{N,AbstractVector}, edges::NTuple{N,AbstractVector}; closed::Symbol=:left) where {T,N} =
norm_type(::Type{T}) where {T<:Integer} = promote_type(T, Int64)
norm_type(::Type{T}) where {T<:AbstractFloat} = promote_type(T, Float64)
""" """ @generated function norm(h::Histogram{T,N}) where {T,N}
    quote
        edges = h.edges
        weights = h.weights
        s_0 = zero(SumT)
        @inbounds @nloops(
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
""" """ @generated function normalize!(h::Histogram{T,N}, aux_weights::Array{T,N}...; mode::Symbol=:pdf) where {T<:AbstractFloat,N}
    quote
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
                    @inbounds @nloops $N i weights d->(vs_{$N-d+1} = vs_{$N-d} * _edge_binvolume(SumT, edges[d], i_d)) begin
                        (@nref $N weights i) /= $(Symbol("vs_$N"))
                        for A in aux_weights
                            (@nref $N A i) /= $(Symbol("vs_$N"))
                        end
                    end
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
end
""" """ Base.zero(h::Histogram{T,N,E}) where {T,N,E} =
    Histogram{T,N,E}(deepcopy(h.edges), zero(h.weights), h.closed, h.isdensity)
""" """ function Base.merge!(target::Histogram, others::Histogram...)
    for h in others
        target.edges != h.edges && throw(ArgumentError("can't merge histograms with different binning"))
        target.isdensity != h.isdensity && throw(ArgumentError("can't merge histograms with different isdensity settings"))
    end
    for h in others
        target.weights .+= h.weights
    end
    target
end
""" """ Base.merge(h::Histogram, others::Histogram...) = merge!(zero(h), h, others...)
