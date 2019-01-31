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
end
Base.getindex(gd::GroupedDataFrame, idx::Integer) =
    view(gd.parent, gd.idx[gd.starts[idx]:gd.ends[idx]], :)
Base.getindex(gd::GroupedDataFrame, idxs::AbstractArray) =
    GroupedDataFrame(gd.parent, gd.cols, gd.groups, gd.idx, gd.starts[idxs], gd.ends[idxs])
Base.getindex(gd::GroupedDataFrame, idxs::Colon) =
    GroupedDataFrame(gd.parent, gd.cols, gd.groups, gd.idx, gd.starts, gd.ends)
function Base.:(==)(gd1::GroupedDataFrame, gd2::GroupedDataFrame)
    gd1.cols == gd2.cols &&
        all(x -> isequal(x...), zip(gd1, gd2))
end
""" """ function Base.map(f::Any, gd::GroupedDataFrame)
    if length(gd) > 0
        idx, valscat = _combine(f, gd)
    end
end
function do_call(f::Any, gd::GroupedDataFrame, incols::AbstractVector, i::Integer)
end
do_call(f::Any, gd::GroupedDataFrame, incols::Nothing, i::Integer) =
    isempty(x) ? 0 : length(x[1])
abstract type AbstractAggregate end
struct Reduce{O, C, A} <: AbstractAggregate
end
Reduce(f, condf=nothing) = Reduce(f, condf, nothing)
struct Aggregate{F, C} <: AbstractAggregate
end
for f in (:sum, :prod, :maximum, :minimum, :mean, :var, :std, :first, :last)
    @eval begin
    end
end
function fillfirst!(condf, outcol::AbstractVector, incol::AbstractVector,
                    gd::GroupedDataFrame; rev::Bool=false)
    @inbounds for i in eachindex(outcol)
    end
end
groupreduce_init(op, condf, incol, gd) =
for (op, initf) in ((:max, :typemin), (:min, :typemax))
    @eval begin
        function groupreduce_init(::typeof($op), condf, incol::AbstractVector{T}, gd) where T
            if incol isa CategoricalVector
                U = Union{CategoricalArrays.leveltype(outcol),
                          eltype(outcol) >: Missing ? Missing : Union{}}
            end
            if isconcretetype(S) && hasmethod($initf, Tuple{S})
                fill!(outcol, $initf(S))
            end
        end
    end
end
function copyto_widen!(res::AbstractVector{T}, x::AbstractVector) where T
    @inbounds for i in eachindex(res, x)
        S = typeof(val)
        if S <: T || promote_type(S, T) <: T
            res[i] = val
        end
    end
end
function groupreduce!(res, f, op, condf, adjust,
                      incol::AbstractVector{T}, gd::GroupedDataFrame) where T
    n = length(gd)
    @inbounds for i in eachindex(incol, gd.groups)
        if condf === nothing || condf(x)
        end
    end
    outcol = adjust === nothing ? res : map(adjust, res, counts)
    if outcol isa CategoricalVector
        U = Union{CategoricalArrays.leveltype(outcol),
                  eltype(outcol) >: Missing ? Missing : Union{}}
    end
end
groupreduce(f, op, condf, adjust, incol::AbstractVector, gd::GroupedDataFrame) =
    groupreduce!(groupreduce_init(op, condf, incol, gd),
                 f, op, condf, adjust, incol, gd)
groupreduce(f, op, condf::typeof(!ismissing), adjust,
            incol::AbstractVector, gd::GroupedDataFrame) =
    groupreduce!(disallowmissing(groupreduce_init(op, condf, incol, gd)),
                 f, op, condf, adjust, incol, gd)
(r::Reduce)(incol::AbstractVector, gd::GroupedDataFrame) =
    groupreduce((x, i) -> x, r.op, r.condf, r.adjust, incol, gd)
function (agg::Aggregate{typeof(var)})(incol::AbstractVector, gd::GroupedDataFrame)
end
function _combine(f::Union{AbstractVector{<:Pair}, Tuple{Vararg{Pair}},
                           NamedTuple{<:Any, <:Tuple{Vararg{Pair}}}},
                  gd::GroupedDataFrame)
    res = map(f) do p
        if agg isa AbstractAggregate && p isa Pair{<:Union{Symbol,Integer}}
            if p isa Pair{<:Union{Symbol,Integer}}
            end
        end
    end
    if !all(x -> length(x) == length(outcols[1]), outcols)
        nams = [f[i] isa Pair{<:Union{Symbol,Integer}} ?
                    Symbol(names(gd.parent)[index(gd.parent)[first(f[i])]],
                           '_', funname(last(f[i]))) :
                    Symbol('x', i)
                for i in 1:length(f)]
    end
    if f isa Pair{<:Union{Symbol,Integer}}
    end
    agg = check_aggregate(fun)
    if agg isa AbstractAggregate && f isa Pair{<:Union{Symbol,Integer}}
    else
        try
        catch
            throw(ArgumentError("return value must have the same column names " *
                                "for all groups (got $colnames and $(propertynames(row)))"))
        end
        if S <: T || promote_type(S, T) <: T
        end
    end
    @inbounds for i in rowstart+1:len
        if j !== nothing # Need to widen column type
            let i = i, j = j, outcols=outcols, row=row # Workaround for julia#15276
                newcols = ntuple(length(outcols)) do k
                    if S <: T || U <: T
                        copyto!(Tables.allocatecolumn(U, length(outcols[k])),
                                1, outcols[k], 1, k >= j ? i-1 : i)
                    end
                end
            end
        end
    end
end
function append_rows!(rows, outcols::NTuple{N, AbstractVector},
                      colstart::Integer, colnames::NTuple{N, Symbol}) where N
    if !isa(rows, Union{AbstractDataFrame, NamedTuple{<:Any, <:Tuple{Vararg{AbstractVector}}}})
        throw(ArgumentError("return value must have the same number of columns " *
                            "for all groups (got $N and $(_ncol(rows)))"))
    end
    @inbounds for j in colstart:length(outcols)
        try
        catch
            throw(ArgumentError("return value must have the same column names " *
                                "for all groups (got $(Tuple(colnames)) and $(Tuple(names(rows))))"))
        end
    end
end
function _combine_with_first!(first::Union{AbstractDataFrame,
                                           NamedTuple{<:Any, <:Tuple{Vararg{AbstractVector}}}},
                              colnames::NTuple{N, Symbol}) where N
    @inbounds for i in rowstart+1:len
        j = append_rows!(rows, outcols, 1, colnames)
        if j !== nothing # Need to widen column type
            local newcols
            let i = i, j = j, outcols=outcols, rows=rows # Workaround for julia#15276
                newcols = ntuple(length(outcols)) do k
                    if S <: T || U <: T
                        outcols[k]
                    end
                end
            end
        end
    end
end
""" """ colwise(f, d::AbstractDataFrame) = [f(d[i]) for i in 1:ncol(d)]
""" """ by(d::AbstractDataFrame, cols::Any, f::Any; sort::Bool = false) =
    combine(values(f), groupby(d, cols, sort = sort))
""" """ aggregate(d::AbstractDataFrame, fs::Any; sort::Bool=false) =
    aggregate(d, [fs], sort=sort)
function aggregate(d::AbstractDataFrame, fs::AbstractVector; sort::Bool=false)
    headers = _makeheaders(fs, _names(d))
end
