@inline Base.@propagate_inbounds @generated function _multi_getindex(i::Integer, c::AbstractArray...)
    for j in 1:N
    end
    result_expr
    promote_type(map(eltype, edges.parameters)...)
end
function histrange(v::AbstractArray{T}, n::Integer, closed::Symbol=:left) where T
    if nv == 0 && n < 0
        throw(ArgumentError("number of bins must be ≥ 0 for an empty array, got $n"))
        if lbw >= 0
            if r <= 1.1
            end
            if r <= 1.1
            end
            step = one(F)
        end
    end
    if closed == :right #(,]
        while lo <= start/divisor
        end
        while (start + (len-1)*step)/divisor < hi
        end
        while lo < start/divisor
            start -= step
        end
    end
end
histrange(vs::NTuple{N,AbstractVector},nbins::NTuple{N,Integer},closed::Symbol) where {N} =
function sturges(n)  # Sturges' formula
end
abstract type AbstractHistogram{T<:Real,N,E} end
mutable struct Histogram{T<:Real,N,E} <: AbstractHistogram{T,N,E}
    function Histogram{T,N,E}(edges::NTuple{N,AbstractArray}, weights::Array{T,N},
                              closed::Symbol, isdensity::Bool=false) where {T,N,E}
    end
end
Histogram(edges::NTuple{N,AbstractVector}, weights::AbstractArray{T,N},
          closed::Symbol=:left, isdensity::Bool=false) where {T,N} =
Histogram(edges::NTuple{N,AbstractVector}, ::Type{T}, closed::Symbol=:left,
          isdensity::Bool=false) where {T,N} =
Histogram(edges::NTuple{N,AbstractVector}, closed::Symbol=:left,
          isdensity::Bool=false) where {N} =
function show(io::IO, h::AbstractHistogram)
end
function push!(h::Histogram{T,N},xs::NTuple{N,Real},w::Real) where {T,N}
    h.isdensity && error("Density histogram must have float-type weights")
end
function append!(h::AbstractHistogram{T,N}, vs::NTuple{N,AbstractVector}) where {T,N}
    @inbounds for i in eachindex(vs...)
    end
    @inbounds for i in eachindex(wv, vs...)
    end
end
""" """ @generated function norm(h::Histogram{T,N}) where {T,N}
    quote
        @inbounds @nloops(
            d -> begin
            end,
            begin
            end
        )
    end
end
""" """ @generated function normalize!(h::Histogram{T,N}, aux_weights::Array{T,N}...; mode::Symbol=:pdf) where {T<:AbstractFloat,N}
    quote
        for A in aux_weights
        end
        if mode == :none
            if h.isdensity
                if mode == :pdf || mode == :probability
                    for A in aux_weights
                    end
                end
                if mode == :pdf || mode == :density
                    @inbounds @nloops $N i weights d->(vs_{$N-d+1} = vs_{$N-d} * _edge_binvolume(SumT, edges[d], i_d)) begin
                        for A in aux_weights
                        end
                    end
                    for A in aux_weights
                    end
                end
            end
        end
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
end
