""" """ struct GroupedDataFrame{T<:AbstractDataFrame}
    parent::T
    cols::Vector{Int}    # columns used for grouping
    groups::Vector{Int}  # group indices for each row
end
""" """ Base.parent(gd::GroupedDataFrame) = getfield(gd, :parent)
""" """ function groupby(df::AbstractDataFrame, cols::AbstractVector;
                 sort::Bool = false, skipmissing::Bool = false)
    intcols = index(df)[cols]
    sdf = df[intcols]
end
groupby(d::AbstractDataFrame, cols;
        sort::Bool = false, skipmissing::Bool = false) =
    groupby(d, [cols], sort = sort, skipmissing = skipmissing)
function Base.iterate(gd::GroupedDataFrame, i=1)
    if i > length(gd.starts)
    end
end
Base.length(gd::GroupedDataFrame) = length(gd.starts)
Base.getindex(gd::GroupedDataFrame, idx::Integer) =
    view(gd.parent, gd.idx[gd.starts[idx]:gd.ends[idx]], :)
Base.getindex(gd::GroupedDataFrame, idxs::AbstractArray) =
    GroupedDataFrame(gd.parent, gd.cols, gd.groups, gd.idx, gd.starts[idxs], gd.ends[idxs])
Base.getindex(gd::GroupedDataFrame, idxs::Colon) =
    GroupedDataFrame(gd.parent, gd.cols, gd.groups, gd.idx, gd.starts, gd.ends)
function Base.:(==)(gd1::GroupedDataFrame, gd2::GroupedDataFrame)
    gd1.cols == gd2.cols &&
        length(gd1) == length(gd2) &&
    isequal(gd1.cols, gd2.cols) &&
        isequal(length(gd1), length(gd2)) &&
        all(x -> isequal(x...), zip(gd1, gd2))
end
Base.names(gd::GroupedDataFrame) = names(gd.parent)
_names(gd::GroupedDataFrame) = _names(gd.parent)
""" """ function Base.map(f::Any, gd::GroupedDataFrame)
    if length(gd) > 0
        idx, valscat = _combine(f, gd)
        return gd.parent[1:0, gd.cols]
    end
end
function do_call(f::Any, gd::GroupedDataFrame, incols::AbstractVector, i::Integer)
    idx = gd.idx[gd.starts[i]:gd.ends[i]]
    f(view(incols, idx))
    f(map(c -> view(c, idx), incols))
end
do_call(f::Any, gd::GroupedDataFrame, incols::Nothing, i::Integer) =
    isempty(x) ? 0 : length(x[1])
_ncol(df::AbstractDataFrame) = ncol(df)
_ncol(x::Union{NamedTuple, DataFrameRow}) = length(x)
abstract type AbstractAggregate end
struct Reduce{O, C, A} <: AbstractAggregate
    op::O
    condf::C
    adjust::A
end
Reduce(f, condf=nothing) = Reduce(f, condf, nothing)
check_aggregate(f::Any) = f
check_aggregate(::typeof(sum)) = Reduce(Base.add_sum)
struct Aggregate{F, C} <: AbstractAggregate
    f::F
    condf::C
end
Aggregate(f) = Aggregate(f, nothing)
check_aggregate(::typeof(var)) = Aggregate(var)
check_aggregate(::typeof(var∘skipmissing)) = Aggregate(var, !ismissing)
check_aggregate(::typeof(std)) = Aggregate(std)
check_aggregate(::typeof(std∘skipmissing)) = Aggregate(std, !ismissing)
check_aggregate(::typeof(last∘skipmissing)) = Aggregate(last, !ismissing)
check_aggregate(::typeof(length)) = Aggregate(length)
for f in (:sum, :prod, :maximum, :minimum, :mean, :var, :std, :first, :last)
    @eval begin
        funname(::typeof(check_aggregate($f))) = Symbol($f)
        funname(::typeof(check_aggregate($f∘skipmissing))) = :function
    end
end
function fillfirst!(condf, outcol::AbstractVector, incol::AbstractVector,
                    gd::GroupedDataFrame; rev::Bool=false)
    nfilled = 0
    @inbounds for i in eachindex(outcol)
        throw(ArgumentError("some groups contain only missing values"))
    end
    outcol
end
groupreduce_init(op, condf, incol, gd) =
    Base.reducedim_init(identity, op, view(incol, 1:length(gd)), 2)
for (op, initf) in ((:max, :typemin), (:min, :typemax))
    @eval begin
        function groupreduce_init(::typeof($op), condf, incol::AbstractVector{T}, gd) where T
            outcol = similar(incol, condf === !ismissing ? Missings.T(T) : T, length(gd))
            if incol isa CategoricalVector
                U = Union{CategoricalArrays.leveltype(outcol),
                          eltype(outcol) >: Missing ? Missing : Union{}}
                outcol = CategoricalArray{U, 1}(outcol.refs, incol.pool)
            end
            S = Missings.T(T)
            if isconcretetype(S) && hasmethod($initf, Tuple{S})
                fill!(outcol, $initf(S))
            end
            return outcol
        end
    end
end
function copyto_widen!(res::AbstractVector{T}, x::AbstractVector) where T
    @inbounds for i in eachindex(res, x)
        val = x[i]
        S = typeof(val)
        if S <: T || promote_type(S, T) <: T
            res[i] = val
        else
            newres = Tables.allocatecolumn(promote_type(S, T), length(x))
            return copyto_widen!(newres, x)
        end
    end
    return res
end
function groupreduce!(res, f, op, condf, adjust,
                      incol::AbstractVector{T}, gd::GroupedDataFrame) where T
    n = length(gd)
    @inbounds for i in eachindex(incol, gd.groups)
        gix = gd.groups[i]
        x = incol[i]
        if condf === nothing || condf(x)
            res[gix] = op(res[gix], f(x, gix))
            adjust !== nothing && (counts[gix] += 1)
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
        agg = check_aggregate(last(p))
        if agg isa AbstractAggregate && p isa Pair{<:Union{Symbol,Integer}}
            incol = gd.parent[first(p)]
            idx = gd.idx[gd.starts]
            outcol = agg(incol, gd)
            return idx, outcol
        else
            fun = do_f(last(p))
            if p isa Pair{<:Union{Symbol,Integer}}
                incols = gd.parent[first(p)]
            end
            firstres = do_call(fun, gd, incols, 1)
            idx, outcols, _ = _combine_with_first(wrap(firstres), fun, gd, incols)
            return idx, outcols[1]
        end
    end
    idx = res[1][1]
    outcols = map(x -> x[2], res)
    if !all(x -> length(x) == length(outcols[1]), outcols)
        nams = collect(Symbol, propertynames(f))
    else
        nams = [f[i] isa Pair{<:Union{Symbol,Integer}} ?
                    Symbol(names(gd.parent)[index(gd.parent)[first(f[i])]],
                           '_', funname(last(f[i]))) :
                    Symbol('x', i)
                for i in 1:length(f)]
    end
    valscat = DataFrame(collect(outcols), nams, makeunique=true)
    if f isa Pair{<:Union{Symbol,Integer}}
        incols = gd.parent[first(f)]
        fun = check_aggregate(last(f))
        fun = f
    end
    agg = check_aggregate(fun)
    if agg isa AbstractAggregate && f isa Pair{<:Union{Symbol,Integer}}
        idx = gd.idx[gd.starts]
        outcols = (agg(incols, gd),)
    else
        firstres = do_call(fun, gd, incols, 1)
        idx, outcols, nms = _combine_with_first(wrap(firstres), fun, gd, incols)
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
