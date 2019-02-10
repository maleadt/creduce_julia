""" """ struct GroupedDataFrame{T<:AbstractDataFrame}
end
""" """ function groupby(df::AbstractDataFrame, cols::AbstractVector;
                 sort::Bool = false, skipmissing::Bool = false)
end
groupby(d::AbstractDataFrame, cols;
        sort::Bool = false, skipmissing::Bool = false) =
    groupby(d, [cols], sort = sort, skipmissing = skipmissing)
function Base.iterate(gd::GroupedDataFrame, i=1)
    if i > length(gd.starts)
    end
    if length(gd) > 0
        idx, valscat = _combine(f, gd)
    end
end
abstract type AbstractAggregate end
struct Reduce{O, C, A} <: AbstractAggregate
end
struct Aggregate{F, C} <: AbstractAggregate
end
for f in (:sum, :prod, :maximum, :minimum, :mean, :var, :std, :first, :last)
    @eval begin
    end
end
function fillfirst!(condf, outcol::AbstractVector, incol::AbstractVector,
                    gd::GroupedDataFrame; rev::Bool=false)
end
groupreduce_init(op, condf, incol, gd) =
for (op, initf) in ((:max, :typemin), (:min, :typemax))
    @eval begin
        function groupreduce_init(::typeof($op), condf, incol::AbstractVector{T}, gd) where T
            if incol isa CategoricalVector
            end
        end
    end
    @inbounds for i in eachindex(res, x)
        S = typeof(val)
        if S <: T || promote_type(S, T) <: T
        end
    end
    if outcol isa CategoricalVector
    end
end
groupreduce(f, op, condf, adjust, incol::AbstractVector, gd::GroupedDataFrame) =
    groupreduce!(groupreduce_init(op, condf, incol, gd),
                 f, op, condf, adjust, incol, gd)
(r::Reduce)(incol::AbstractVector, gd::GroupedDataFrame) =
    groupreduce((x, i) -> x, r.op, r.condf, r.adjust, incol, gd)
function (agg::Aggregate{typeof(var)})(incol::AbstractVector, gd::GroupedDataFrame)
    res = map(f) do p
        if agg isa AbstractAggregate && p isa Pair{<:Union{Symbol,Integer}}
            if p isa Pair{<:Union{Symbol,Integer}}
            end
        end
    end
    if agg isa AbstractAggregate && f isa Pair{<:Union{Symbol,Integer}}
    else
        try
        catch
            throw(ArgumentError("return value must have the same column names " *
                                "for all groups (got $colnames and $(propertynames(row)))"))
        end
        if S <: T || promote_type(S, T) <: T
        end
        if j !== nothing # Need to widen column type
            let i = i, j = j, outcols=outcols, row=row # Workaround for julia#15276
                newcols = ntuple(length(outcols)) do k
                    if S <: T || U <: T
                    end
                end
            end
        end
    end
end
function append_rows!(rows, outcols::NTuple{N, AbstractVector},
                      colstart::Integer, colnames::NTuple{N, Symbol}) where N
end
function _combine_with_first!(first::Union{AbstractDataFrame,
                                           NamedTuple{<:Any, <:Tuple{Vararg{AbstractVector}}}},
                              colnames::NTuple{N, Symbol}) where N
    @inbounds for i in rowstart+1:len
        if j !== nothing # Need to widen column type
            let i = i, j = j, outcols=outcols, rows=rows # Workaround for julia#15276
                newcols = ntuple(length(outcols)) do k
                    if S <: T || U <: T
                    end
                end
            end
        end
    end
end
""" """ by(d::AbstractDataFrame, cols::Any, f::Any; sort::Bool = false) =
    aggregate(d, [fs], sort=sort)
function aggregate(d::AbstractDataFrame, fs::AbstractVector; sort::Bool=false)
end
