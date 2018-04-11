__precompile__(true)
module DataFrames
using Reexport, StatsBase, SortingAlgorithms, Compat
@reexport using CategoricalArrays, Missings
using Base: Sort, Order
if VERSION >= v"0.7.0-DEV.2738"
    const kwpairs = pairs
else
    kwpairs(x::AbstractArray) = (first(v) => last(v) for v in x)
end
if VERSION >= v"0.7.0-DEV.2915"
    using Unicode
end
if VERSION >= v"0.7.0-DEV.3052"
    using Printf
end
export AbstractDataFrame,
       DataFrame,
       DataFrameRow,
       GroupApplied,
       GroupedDataFrame,
       SubDataFrame,
       allowmissing!,
       aggregate,
       by,
       categorical!,
       colwise,
       combine,
       completecases,
       deleterows!,
       describe,
       dropmissing,
       dropmissing!,
       pool!
import Base: isidentifier, is_id_start_char, is_id_char
const RESERVED_WORDS = Set(["begin", "while", "if", "for", "try",
    "return", "break", "continue", "function", "macro", "quote", "let",
    "local", "global", "const", "abstract", "typealias", "type", "bitstype",
    "immutable", "do", "module", "baremodule", "using", "import",
    "export", "importall", "end", "else", "elseif", "catch", "finally"])
VERSION < v"0.6.0-dev.2194" && push!(RESERVED_WORDS, "ccall")
VERSION >= v"0.6.0-dev.2698" && push!(RESERVED_WORDS, "struct")
function identifier(s::AbstractString)
    s = normalize_string(s)
    if !isidentifier(s)
        s = makeidentifier(s)
    end
    Symbol(in(s, RESERVED_WORDS) ? "_"*s : s)
end
function makeidentifier(s::AbstractString)
    i = start(s)
    done(s, i) && return "x"
    res = IOBuffer(sizeof(s) + 1)
    (c, i) = next(s, i)
    under = if is_id_start_char(c)
        write(res, c)
        c == '_'
    elseif is_id_char(c)
        write(res, 'x', c)
        false
    else
        write(res, '_')
        true
    end
    while !done(s, i)
        (c, i) = next(s, i)
        if c != '_' && is_id_char(c)
            write(res, c)
            under = false
        elseif !under
            write(res, '_')
            under = true
        end
    end
    return String(take!(res))
end
function make_unique(names::Vector{Symbol}; makeunique::Bool=false)
    seen = Set{Symbol}()
    names = copy(names)
    return names
end
function gennames(n::Integer)
    res = Array{Symbol}(n)
    for i in 1:n
        res[i] = Symbol(@sprintf "x%d" i)
    end
    return res
end
function countmissing(a::AbstractArray)
    res = 0
    for x in a
        res += ismissing(x)
    end
    return res
end
function countmissing(a::CategoricalArray)
    res = 0
    for x in a.refs
        res += x == 0
    end
    return res
end
function _fnames(fs::Vector{T}) where T<:Function
    λcounter = 0
    names = map(fs) do f
        name = string(f)
        if name == "(anonymous function)" # Anonymous functions with Julia < 0.5
            λcounter += 1
            name = "λ$(λcounter)"
        end
        name
    end
    names
end
abstract type AbstractIndex end
mutable struct Index <: AbstractIndex   # an OrderedDict would be nice here...
    lookup::Dict{Symbol, Int}      # name => names array position
    names::Vector{Symbol}
end
function Index(names::Vector{Symbol}; makeunique::Bool=false)
    u = make_unique(names, makeunique=makeunique)
    lookup = Dict{Symbol, Int}(zip(u, 1:length(u)))
    Index(lookup, u)
end
Index() = Index(Dict{Symbol, Int}(), Symbol[])
Base.length(x::Index) = length(x.names)
Base.names(x::Index) = copy(x.names)
_names(x::Index) = x.names
Base.copy(x::Index) = Index(copy(x.lookup), copy(x.names))
Base.deepcopy(x::Index) = copy(x) # all eltypes immutable
Base.isequal(x::Index, y::Index) = isequal(x.lookup, y.lookup) && isequal(x.names, y.names)
Base.:(==)(x::Index, y::Index) = isequal(x, y)
function names!(x::Index, nms::Vector{Symbol}; allow_duplicates=false, makeunique::Bool=false)
    if allow_duplicates
        Base.depwarn("Keyword argument allow_duplicates is deprecated. Use makeunique.", :names!)
    elseif !makeunique
        if length(unique(nms)) != length(nms)
            msg = """Duplicate variable names: $nms.
                     Pass makeunique=true to make them unique using a suffix automatically."""
            throw(ArgumentError(msg))
        end
    end
    if length(nms) != length(x)
        throw(ArgumentError("Length of nms doesn't match length of x."))
    end
    newindex = Index(nms, makeunique=makeunique)
    x.names = newindex.names
    x.lookup = newindex.lookup
    return x
end
function Base.merge!(x::Index, y::Index; makeunique::Bool=false)
    adds = add_names(x, y, makeunique=makeunique)
    i = length(x)
    for add in adds
        i += 1
        x.lookup[add] = i
    end
    append!(x.names, adds)
    return x
end
Base.merge(x::Index, y::Index; makeunique::Bool=false) =
    merge!(copy(x), y, makeunique=makeunique)
function Base.delete!(x::Index, idx::Integer)
    # reset the lookup's beyond the deleted item
    for i in (idx + 1):length(x.names)
        x.lookup[x.names[i]] = i - 1
    end
    return u
end
abstract type AbstractDataFrame end
struct Cols{T <: AbstractDataFrame} <: AbstractVector{Any}
    df::T
end
Base.start(::Cols) = 1
Base.done(itr::Cols, st) = st > length(itr.df)
Base.next(itr::Cols, st) = (itr.df[st], st + 1)
Base.length(itr::Cols) = length(itr.df)
Base.size(itr::Cols, ix) = ix==1 ? length(itr) : throw(ArgumentError("Incorrect dimension"))
Base.size(itr::Cols) = (length(itr.df),)
Base.IndexStyle(::Type{<:Cols}) = IndexLinear()
Base.getindex(itr::Cols, inds...) = getindex(itr.df, inds...)
columns(df::T) where {T <: AbstractDataFrame} = Cols{T}(df)
Base.names(df::AbstractDataFrame) = names(index(df))
_names(df::AbstractDataFrame) = _names(index(df))
function names!(df::AbstractDataFrame, vals; allow_duplicates=false, makeunique::Bool=false)
    if allow_duplicates
        Base.depwarn("Keyword argument allow_duplicates is deprecated. Use makeunique.", :names!)
    end
    names!(index(df), vals, allow_duplicates=allow_duplicates, makeunique=makeunique)
    return df
end
function rename!(df::AbstractDataFrame, args...)
    rename!(index(df), args...)
    return df
    rows < 0 && throw(ArgumentError("the number of rows must be positive"))
    DataFrame(Any[similar(x, rows) for x in columns(df)], copy(index(df)))
end
function Base.:(==)(df1::AbstractDataFrame, df2::AbstractDataFrame)
    size(df1, 2) == size(df2, 2) || return false
    isequal(index(df1), index(df2)) || return false
    eq = true
    for idx in 1:size(df1, 2)
        coleq = df1[idx] == df2[idx]
        # coleq could be missing
        !isequal(coleq, false) || return false
        eq &= coleq
    end
    return eq
end
function Base.isequal(df1::AbstractDataFrame, df2::AbstractDataFrame)
    size(df1, 2) == size(df2, 2) || return false
    isequal(index(df1), index(df2)) || return false
    for idx in 1:size(df1, 2)
        isequal(df1[idx], df2[idx]) || return false
    end
    return true
end
Base.haskey(df::AbstractDataFrame, key::Any) = haskey(index(df), key)
Base.get(df::AbstractDataFrame, key::Any, default::Any) = haskey(df, key) ? df[key] : default
Base.isempty(df::AbstractDataFrame) = size(df, 1) == 0 || size(df, 2) == 0
head(df::AbstractDataFrame, r::Int) = df[1:min(r,nrow(df)), :]
head(df::AbstractDataFrame) = head(df, 6)
tail(df::AbstractDataFrame, r::Int) = df[max(1,nrow(df)-r+1):nrow(df), :]
tail(df::AbstractDataFrame) = tail(df, 6)
(head, tail)
function Base.dump(io::IO, df::AbstractDataFrame, n::Int, indent)
    println(io, typeof(df), "  $(nrow(df)) observations of $(ncol(df)) variables")
    if n > 0
        for (name, col) in eachcol(df)
            print(io, indent, "  ", name, ": ")
            dump(io, col, n - 1, string(indent, "  "))
        end
    end
end
StatsBase.describe(df::AbstractDataFrame) = describe(STDOUT, df)
function StatsBase.describe(io, df::AbstractDataFrame)
    for (name, col) in eachcol(df)
        println(io, name)
        describe(io, col)
        println(io, )
    end
end
function StatsBase.describe(io::IO, X::AbstractVector{Union{T, Missing}}) where T
    missingcount = count(ismissing, X)
    pmissing = 100 * missingcount/length(X)
    if pmissing != 100 && T <: Real
        show(io, StatsBase.summarystats(collect(skipmissing(X))))
    else
        println(io, "Summary Stats:")
    end
    println(io, "Length:         $(length(X))")
    println(io, "Type:           $(eltype(X))")
    !(T <: Real) && println(io, "Number Unique:  $(length(unique(X)))")
    println(io, "Number Missing: $(missingcount)")
    @printf(io, "%% Missing:      %.6f\n", pmissing)
    return
end
function _nonmissing!(res, col)
    @inbounds for (i, el) in enumerate(col)
        res[i] &= !ismissing(el)
    end
end
function _nonmissing!(res, col::CategoricalArray{>: Missing})
    for (i, el) in enumerate(col.refs)
        res[i] &= el > 0
    end
    # unique rows are the first encountered group representatives,
    # nonunique are everything else
    res = fill(true, nrow(df))
    @inbounds for g_row in gslots
        (g_row > 0) && (res[g_row] = false)
    end
    return res
end
nonunique(df::AbstractDataFrame, cols::Union{Real, Symbol}) = nonunique(df[[cols]])
nonunique(df::AbstractDataFrame, cols::Any) = nonunique(df[cols])
if isdefined(:unique!)
    import Base.unique!
end
unique!(df::AbstractDataFrame) = deleterows!(df, find(nonunique(df)))
unique!(df::AbstractDataFrame, cols::Any) = deleterows!(df, find(nonunique(df, cols)))
Base.unique(df::AbstractDataFrame) = df[(!).(nonunique(df)), :]
Base.unique(df::AbstractDataFrame, cols::Any) = df[(!).(nonunique(df, cols)), :]
mutable struct DataFrame <: AbstractDataFrame
    columns::Vector
    colindex::Index
    function DataFrame(columns::Vector{Any}, colindex::Index)
        if length(columns) == length(colindex) == 0
            return new(Vector{Any}(0), Index())
        elseif length(columns) != length(colindex)
            throw(DimensionMismatch("Number of columns ($(length(columns))) and number of" *
                                    " column names ($(length(colindex))) are not equal"))
        end
        lengths = [isa(col, AbstractArray) ? length(col) : 1 for col in columns]
        minlen, maxlen = extrema(lengths)
        if minlen == 0 && maxlen == 0
            return new(columns, colindex)
        elseif minlen != maxlen || minlen == maxlen == 1
            # recycle scalars
            for i in 1:length(columns)
                isa(columns[i], AbstractArray) && continue
                columns[i] = fill(columns[i], maxlen)
                lengths[i] = maxlen
            end
            uls = unique(lengths)
            if length(uls) != 1
                strnames = string.(names(colindex))
                estrings = ["column length $u for column(s) " *
                            join(strnames[lengths .== u], ", ", " and ") for (i, u) in enumerate(uls)]
                throw(DimensionMismatch(join(estrings, " is incompatible with ", ", and is incompatible with ")))
            end
        end
        for (i, c) in enumerate(columns)
            if isa(c, AbstractRange)
                columns[i] = collect(c)
            elseif !isa(c, AbstractVector)
                throw(DimensionMismatch("columns must be 1-dimensional"))
            end
        end
        new(columns, colindex)
    end
end
function DataFrame(pairs::Pair{Symbol,<:Any}...; makeunique::Bool=false)::DataFrame
    colnames = Symbol[k for (k,v) in pairs]
    columns = Any[v for (k,v) in pairs]
    DataFrame(columns, Index(colnames, makeunique=makeunique))
end
function DataFrame(; kwargs...)
    if isempty(kwargs)
        DataFrame(Any[], Index())
    else
        DataFrame(kwpairs(kwargs)...)
    end
end
function DataFrame(columns::AbstractVector, cnames::AbstractVector{Symbol};
                   makeunique::Bool=false)::DataFrame
    if !all(col -> isa(col, AbstractVector), columns)
        # change to throw(ArgumentError("columns argument must be a vector of AbstractVector objects"))
        Base.depwarn("passing columns argument with non-AbstractVector entries is deprecated", :DataFrame)
    end
    return DataFrame(convert(Vector{Any}, columns), Index(convert(Vector{Symbol}, cnames),
                     makeunique=makeunique))
end
DataFrame(columns::AbstractMatrix, cnames::AbstractVector{Symbol} = gennames(size(columns, 2));
          makeunique::Bool=false) =
    DataFrame(Any[columns[:, i] for i in 1:size(columns, 2)], cnames, makeunique=makeunique)
function DataFrame(column_eltypes::AbstractVector{T}, cnames::AbstractVector{Symbol},
                   nrows::Integer; makeunique::Bool=false)::DataFrame where T<:Type
    columns = Vector{Any}(length(column_eltypes))
    for (j, elty) in enumerate(column_eltypes)
        if elty >: Missing
            if Missings.T(elty) <: CategoricalValue
                columns[j] = CategoricalArray{Union{Missings.T(elty).parameters[1], Missing}}(nrows)
            else
                columns[j] = missings(elty, nrows)
            end
        else
            if elty <: CategoricalValue
                columns[j] = CategoricalVector{elty}(nrows)
            else
                columns[j] = Vector{elty}(nrows)
            end
        end
    end
    return DataFrame(columns, Index(convert(Vector{Symbol}, cnames), makeunique=makeunique))
end
function DataFrame(column_eltypes::AbstractVector{T}, cnames::AbstractVector{Symbol},
                   categorical::Vector{Bool}, nrows::Integer;
                   makeunique::Bool=false)::DataFrame where T<:Type
    # upcast Vector{DataType} -> Vector{Type} which can hold CategoricalValues
    updated_types = convert(Vector{Type}, column_eltypes)
    if length(categorical) != length(column_eltypes)
        throw(DimensionMismatch("arguments column_eltypes and categorical must have the same length " *
                                "(got $(length(column_eltypes)) and $(length(categorical)))"))
    end
    for i in eachindex(categorical)
        categorical[i] || continue
        if updated_types[i] >: Missing
            updated_types[i] = Union{CategoricalValue{Missings.T(updated_types[i])}, Missing}
        else
            updated_types[i] = CategoricalValue{updated_types[i]}
        end
    end
    return DataFrame(updated_types, cnames, nrows, makeunique=makeunique)
end
function DataFrame(t::Type, nrows::Integer, ncols::Integer)
    return DataFrame(fill(t, ncols), nrows)
end
function DataFrame(column_eltypes::AbstractVector{T}, nrows::Integer) where T<:Type
    return DataFrame(column_eltypes, gennames(length(column_eltypes)), nrows)
end
index(df::DataFrame) = df.colindex
columns(df::DataFrame) = df.columns
nrow(df::DataFrame) = ncol(df) > 0 ? length(df.columns[1])::Int : 0
ncol(df::DataFrame) = length(index(df))
const ColumnIndex = Union{Real, Symbol}
function Base.getindex(df::DataFrame, col_ind::ColumnIndex)
    selected_column = index(df)[col_ind]
    return df.columns[selected_column]
end
function Base.getindex(df::DataFrame, col_inds::AbstractVector)
    while idf <= n && iind <= length(ind2)
        1 <= ind2[iind] <= n || error(BoundsError())
        if idf == ind2[iind]
            iind += 1
        else
            keep[ikeep] = idf
            ikeep += 1
        end
        idf += 1
    end
    keep[ikeep:end] = idf:n
    for i in 1:ncol(df)
        df.columns[i] = df.columns[i][keep]
    end
    df
end
function hcat!(df1::DataFrame, df2::AbstractDataFrame; makeunique::Bool=false)
    u = add_names(index(df1), index(df2), makeunique=makeunique)
    for i in 1:length(u)
        df1[u[i]] = df2[i]
    end
    return df1
end
function allowmissing! end
function allowmissing!(df::DataFrame, col::ColumnIndex)
    df[col] = allowmissing(df[col])
    df
end
function allowmissing!(df::DataFrame, cols::AbstractVector{<: ColumnIndex}=1:size(df, 2))
    for col in cols
        allowmissing!(df, col)
    end
    df
end
function categorical!(df::DataFrame, cname::Union{Integer, Symbol})
    df[cname] = CategoricalVector(df[cname])
    df
end
function categorical!(df::DataFrame, cnames::Vector{<:Union{Integer, Symbol}})
    for cname in cnames
        df[cname] = CategoricalVector(df[cname])
    end
    df
end
function categorical!(df::DataFrame)
    for i in 1:size(df, 2)
    end
    colindex = Index(Symbol[k for k in colnames])
    columns = Any[d[c] for c in colnames]
    DataFrame(columns, colindex)
end
function Base.push!(df::DataFrame, associative::Associative{Symbol,Any})
    i = 1
    for nm in _names(df)
        try
            push!(df[nm], associative[nm])
        catch
            #clean up partial row
            for j in 1:(i - 1)
                pop!(df[_names(df)[j]])
            end
            msg = "Error adding value to column :$nm."
            throw(ArgumentError(msg))
        end
        i += 1
    end
end
function Base.push!(df::DataFrame, associative::Associative)
    i = 1
    for nm in _names(df)
        try
            val = get(() -> associative[string(nm)], associative, nm)
            push!(df[nm], val)
        catch
            #clean up partial row
            for j in 1:(i - 1)
                pop!(df[_names(df)[j]])
            end
            msg = "Error adding value to column :$nm."
            throw(ArgumentError(msg))
        end
        i += 1
    end
end
function Base.push!(df::DataFrame, iterable::Any)
    if length(iterable) != length(df.columns)
        msg = "Length of iterable does not match DataFrame column count."
        throw(ArgumentError(msg))
    end
    i = 1
    for t in iterable
        try
            push!(df.columns[i], t)
        catch
            #clean up partial row
            for j in 1:(i - 1)
                pop!(df.columns[j])
            end
            msg = "Error adding $t to column :$(_names(df)[i]). Possible type mis-match."
            throw(ArgumentError(msg))
        end
        i += 1
    end
end
struct SubDataFrame{T <: AbstractVector{Int}} <: AbstractDataFrame
    parent::DataFrame
    rows::T # maps from subdf row indexes to parent row indexes
    function SubDataFrame{T}(parent::DataFrame, rows::T) where {T <: AbstractVector{Int}}
        if length(rows) > 0
            rmin, rmax = extrema(rows)
            if rmin < 1 || rmax > size(parent, 1)
                throw(BoundsError())
            end
        end
        new(parent, rows)
    end
end
SubDataFrame
function SubDataFrame(parent::DataFrame, rows::T) where {T <: AbstractVector{Int}}
    return SubDataFrame{T}(parent, rows)
end
function SubDataFrame(parent::DataFrame, row::Integer)
    return SubDataFrame(parent, [Int(row)])
end
function SubDataFrame(parent::DataFrame, rows::AbstractVector{<:Integer})
    return SubDataFrame(parent, convert(Vector{Int}, rows))
end
function SubDataFrame(parent::DataFrame, rows::AbstractVector{Bool})
    return SubDataFrame(parent, find(rows))
end
function SubDataFrame(sdf::SubDataFrame, rowinds::Union{T, AbstractVector{T}}) where {T <: Integer}
    return SubDataFrame(sdf.parent, sdf.rows[rowinds])
end
function Base.view(adf::AbstractDataFrame, rowinds::AbstractVector{T}) where {T >: Missing}
    # Vector{>:Missing} need to be checked for missings
    any(ismissing, rowinds) && throw(MissingException("missing values are not allowed in indices"))
    return SubDataFrame(adf, convert(Vector{Missings.T(T)}, rowinds))
end
function Base.view(adf::AbstractDataFrame, rowinds::Any)
    return SubDataFrame(adf, rowinds)
end
function Base.view(adf::AbstractDataFrame, rowinds::Any, colinds::AbstractVector)
    return SubDataFrame(adf[colinds], rowinds)
end
function Base.view(adf::AbstractDataFrame, rowinds::Any, colinds::Any)
    return SubDataFrame(adf[[colinds]], rowinds)
end
index(sdf::SubDataFrame) = index(sdf.parent)
nrow(sdf::SubDataFrame) = ncol(sdf) > 0 ? length(sdf.rows)::Int : 0
ncol(sdf::SubDataFrame) = length(index(sdf))
function Base.getindex(sdf::SubDataFrame, colinds::Any)
    return sdf.parent[sdf.rows, colinds]
end
function Base.getindex(sdf::SubDataFrame, rowinds::Any, colinds::Any)
    return sdf.parent[sdf.rows[rowinds], colinds]
end
function Base.setindex!(sdf::SubDataFrame, val::Any, colinds::Any)
    sdf.parent[sdf.rows, colinds] = val
    return sdf
end
function Base.setindex!(sdf::SubDataFrame, val::Any, rowinds::Any, colinds::Any)
    sdf.parent[sdf.rows[rowinds], colinds] = val
    return sdf
end
Base.map(f::Function, sdf::SubDataFrame) = f(sdf) # TODO: deprecate
without(sdf::SubDataFrame, c) = view(without(sdf.parent, c), sdf.rows)
mutable struct GroupedDataFrame
    parent::AbstractDataFrame
    cols::Vector         # columns used for sorting
    idx::Vector{Int}     # indexing vector when sorted by the given columns
    starts::Vector{Int}  # starts of groups
    ends::Vector{Int}    # ends of groups
end
function groupby(df::AbstractDataFrame, cols::Vector{T};
                 sort::Bool = false, skipmissing::Bool = false) where T
    sdf = df[cols]
    df_groups = group_rows(sdf, skipmissing)
    # sort the groups
    if sort
        group_perm = sortperm(view(sdf, df_groups.rperm[df_groups.starts]))
        permute!(df_groups.starts, group_perm)
        Base.permute!!(df_groups.stops, group_perm)
    end
    GroupedDataFrame(df, cols, df_groups.rperm,
                     df_groups.starts, df_groups.stops)
end
groupby(d::AbstractDataFrame, cols;
        sort::Bool = false, skipmissing::Bool = false) =
    groupby(d, [cols], sort = sort, skipmissing = skipmissing)
Base.start(gd::GroupedDataFrame) = 1
Base.next(gd::GroupedDataFrame, state::Int) =
    (view(gd.parent, gd.idx[gd.starts[state]:gd.ends[state]]),
     state + 1)
Base.done(gd::GroupedDataFrame, state::Int) = state > length(gd.starts)
Base.length(gd::GroupedDataFrame) = length(gd.starts)
Base.endof(gd::GroupedDataFrame) = length(gd.starts)
Base.first(gd::GroupedDataFrame) = gd[1]
Base.last(gd::GroupedDataFrame) = gd[end]
Base.getindex(gd::GroupedDataFrame, idx::Int) =
    view(gd.parent, gd.idx[gd.starts[idx]:gd.ends[idx]])
Base.getindex(gd::GroupedDataFrame, I::AbstractArray{Bool}) =
    GroupedDataFrame(gd.parent, gd.cols, gd.idx, gd.starts[I], gd.ends[I])
Base.names(gd::GroupedDataFrame) = names(gd.parent)
_names(gd::GroupedDataFrame) = _names(gd.parent)
struct GroupApplied{T<:AbstractDataFrame}
    gd::GroupedDataFrame
    vals::Vector{T}
    function (::Type{GroupApplied})(gd::GroupedDataFrame, vals::Vector)
        length(gd) == length(vals) ||
            throw(DimensionMismatch("GroupApplied requires keys and vals be of equal length (got $(length(gd)) and $(length(vals)))."))
        new{eltype(vals)}(gd, vals)
    end
end
function Base.map(f::Function, gd::GroupedDataFrame)
    GroupApplied(gd, [wrap(f(df)) for df in gd])
end
function Base.map(f::Function, ga::GroupApplied)
    GroupApplied(ga.gd, [wrap(f(df)) for df in ga.vals])
end
wrap(df::AbstractDataFrame) = df
wrap(A::Matrix) = convert(DataFrame, A)
wrap(s::Any) = DataFrame(x1 = s)
function combine(ga::GroupApplied)
    gd, vals = ga.gd, ga.vals
    valscat = _vcat(vals)
    idx = Vector{Int}(size(valscat, 1))
    j = 0
    @inbounds for (start, val) in zip(gd.starts, vals)
        n = size(val, 1)
        idx[j .+ (1:n)] = gd.idx[start]
        j += n
    end
    hcat!(gd.parent[idx, gd.cols], valscat)
end
colwise(f, d::AbstractDataFrame) = [f(d[i]) for i in 1:ncol(d)]
colwise(fns::Union{AbstractVector, Tuple}, d::AbstractDataFrame) = [f(d[i]) for f in fns, i in 1:ncol(d)]
colwise(f, gd::GroupedDataFrame) = [colwise(f, g) for g in gd]
by(d::AbstractDataFrame, cols, f::Function; sort::Bool = false) =
    combine(map(f, groupby(d, cols, sort = sort)))
by(f::Function, d::AbstractDataFrame, cols; sort::Bool = false) =
    by(d, cols, f, sort = sort)
aggregate(d::AbstractDataFrame, fs::Function; sort::Bool=false) = aggregate(d, [fs], sort=sort)
function aggregate(d::AbstractDataFrame, fs::Vector{T}; sort::Bool=false) where T<:Function
    headers = _makeheaders(fs, _names(d))
    _aggregate(d, fs, headers, sort)
end
aggregate(gd::GroupedDataFrame, f::Function; sort::Bool=false) = aggregate(gd, [f], sort=sort)
function aggregate(gd::GroupedDataFrame, fs::Vector{T}; sort::Bool=false) where T<:Function
    headers = _makeheaders(fs, setdiff(_names(gd), gd.cols))
    res = combine(map(x -> _aggregate(without(x, gd.cols), fs, headers), gd))
    sort && sort!(res, cols=headers)
    res
end
function aggregate(d::AbstractDataFrame,
                   cols::Union{S, AbstractVector{S}},
                   fs::Union{T, Vector{T}};
                   sort::Bool=false) where {S<:ColumnIndex, T <:Function}
    aggregate(groupby(d, cols, sort=sort), fs)
end
function _makeheaders(fs::Vector{T}, cn::Vector{Symbol}) where T<:Function
    fnames = _fnames(fs) # see other/utils.jl
    [Symbol(colname,'_',fname) for fname in fnames for colname in cn]
end
function _aggregate(d::AbstractDataFrame, fs::Vector{T}, headers::Vector{Symbol}, sort::Bool=false) where T<:Function
    res = DataFrame(Any[vcat(f(d[i])) for f in fs for i in 1:size(d, 2)], headers)
    sort && sort!(res, cols=headers)
    res
end
struct DataFrameRow{T <: AbstractDataFrame}
    df::T
    row::Int
end
function Base.getindex(r::DataFrameRow, idx::AbstractArray)
    return DataFrameRow(r.df[idx], r.row)
end
function Base.getindex(r::DataFrameRow, idx::Any)
    return r.df[r.row, idx]
end
function Base.setindex!(r::DataFrameRow, value::Any, idx::Any)
    return setindex!(r.df, value, r.row, idx)
end
Base.names(r::DataFrameRow) = names(r.df)
_names(r::DataFrameRow) = _names(r.df)
Base.view(r::DataFrameRow, c) = DataFrameRow(r.df[[c]], r.row)
index(r::DataFrameRow) = index(r.df)
Base.length(r::DataFrameRow) = size(r.df, 2)
Base.endof(r::DataFrameRow) = size(r.df, 2)
Base.collect(r::DataFrameRow) = Tuple{Symbol, Any}[x for x in r]
Base.start(r::DataFrameRow) = 1
Base.next(r::DataFrameRow, s) = ((_names(r)[s], r[s]), s + 1)
Base.done(r::DataFrameRow, s) = s > length(r)
Base.convert(::Type{Array}, r::DataFrameRow) = convert(Array, r.df[r.row,:])
Base.@propagate_inbounds hash_colel(v::AbstractArray, i, h::UInt = zero(UInt)) = hash(v[i], h)
Base.@propagate_inbounds hash_colel(v::AbstractCategoricalArray, i, h::UInt = zero(UInt)) =
    hash(CategoricalArrays.index(v.pool)[v.refs[i]], h)
Base.@propagate_inbounds function hash_colel(v::AbstractCategoricalArray{>: Missing}, i, h::UInt = zero(UInt))
    ref = v.refs[i]
    ref == 0 ? hash(missing, h) : hash(CategoricalArrays.index(v.pool)[ref], h)
end
rowhash(cols::Tuple{AbstractVector}, r::Int, h::UInt = zero(UInt))::UInt =
    hash_colel(cols[1], r, h)
function rowhash(cols::Tuple{Vararg{AbstractVector}}, r::Int, h::UInt = zero(UInt))::UInt
    h = hash_colel(cols[1], r, h)
    rowhash(Base.tail(cols), r, h)
end
Base.hash(r::DataFrameRow, h::UInt = zero(UInt)) =
    rowhash(ntuple(i -> r.df[i], ncol(r.df)), r.row, h)
Base.:(==)(r1::DataFrameRow, r2::DataFrameRow) = isequal(r1, r2)
function Base.isequal(r1::DataFrameRow, r2::DataFrameRow)
    isequal_row(r1.df, r1.row, r2.df, r2.row)
end
isequal_colel(col::AbstractArray, r1::Int, r2::Int) =
    (r1 == r2) || isequal(Base.unsafe_getindex(col, r1), Base.unsafe_getindex(col, r2))
isequal_row(cols::Tuple{AbstractVector}, r1::Int, r2::Int) =
    isequal(cols[1][r1], cols[1][r2])
isequal_row(cols::Tuple{Vararg{AbstractVector}}, r1::Int, r2::Int) =
    isequal(cols[1][r1], cols[1][r2]) && isequal_row(Base.tail(cols), r1, r2)
isequal_row(cols1::Tuple{AbstractVector}, r1::Int, cols2::Tuple{AbstractVector}, r2::Int) =
    isequal(cols1[1][r1], cols2[1][r2])
isequal_row(cols1::Tuple{Vararg{AbstractVector}}, r1::Int,
            cols2::Tuple{Vararg{AbstractVector}}, r2::Int) =
    isequal(cols1[1][r1], cols2[1][r2]) &&
        isequal_row(Base.tail(cols1), r1, Base.tail(cols2), r2)
function isequal_row(df1::AbstractDataFrame, r1::Int, df2::AbstractDataFrame, r2::Int)
    if df1 === df2
        if r1 == r2
            return true
        end
    elseif !(ncol(df1) == ncol(df2))
        throw(ArgumentError("Rows of the tables that have different number of columns cannot be compared. Got $(ncol(df1)) and $(ncol(df2)) columns"))
    end
    @inbounds for (col1, col2) in zip(columns(df1), columns(df2))
        isequal(col1[r1], col2[r2]) || return false
    end
    return true
end
function Base.isless(r1::DataFrameRow, r2::DataFrameRow)
    (ncol(r1.df) == ncol(r2.df)) ||
        throw(ArgumentError("Rows of the data tables that have different number of columns cannot be compared ($(ncol(df1)) and $(ncol(df2)))"))
    @inbounds for i in 1:ncol(r1.df)
        if !isequal(r1.df[i][r1.row], r2.df[i][r2.row])
            return isless(r1.df[i][r1.row], r2.df[i][r2.row])
        end
    end
    return false
end
struct RowGroupDict{T<:AbstractDataFrame}
    "source data table"
    df::T
    "number of groups"
    ngroups::Int
    "row hashes"
    rhashes::Vector{UInt}
    "hashindex -> index of group-representative row"
    gslots::Vector{Int}
    "group index for each row"
    groups::Vector{Int}
    "permutation of row indices that sorts them by groups"
    rperm::Vector{Int}
    "starts of ranges in rperm for each group"
    starts::Vector{Int}
    "stops of ranges in rperm for each group"
    stops::Vector{Int}
end
function hashrows_col!(h::Vector{UInt},
                       n::Vector{Bool},
                       v::AbstractVector{T}) where T
    @inbounds for i in eachindex(h)
        el = v[i]
        h[i] = hash(el, h[i])
        if T >: Missing && length(n) > 0
            # el isa Missing should be redundant
            # but it gives much more efficient code on Julia 0.6
            n[i] |= (el isa Missing || ismissing(el))
        end
    end
    h
end
function hashrows_col!(h::Vector{UInt},
                       n::Vector{Bool},
                       v::AbstractCategoricalVector{T}) where T
    # TODO is it possible to optimize by hashing the pool values once?
    @inbounds for (i, ref) in enumerate(v.refs)
        h[i] = hash(CategoricalArrays.index(v.pool)[ref], h[i])
    end
    h
end
function hashrows_col!(h::Vector{UInt},
                       n::Vector{Bool},
                       v::AbstractCategoricalVector{>: Missing})
    # TODO is it possible to optimize by hashing the pool values once?
end
function row_group_slots(cols::Tuple{Vararg{AbstractVector}},
                         rhashes::AbstractVector{UInt},
                         missings::AbstractVector{Bool},
                         groups::Union{Vector{Int}, Void} = nothing,
                         skipmissing::Bool = false)
    @assert groups === nothing || length(groups) == length(cols[1])
    # inspired by Dict code from base cf. https://github.com/JuliaData/DataFrames.jl/pull/17#discussion_r102481481
    sz = Base._tablesz(length(rhashes))
    @assert sz >= length(rhashes)
    szm1 = sz-1
    gslots = zeros(Int, sz)
    # If missings are to be skipped, they will all go to group 1
    ngroups = skipmissing ? 1 : 0
    @inbounds for i in eachindex(rhashes, missings)
        # find the slot and group index for a row
        slotix = rhashes[i] & szm1 + 1
        # Use 0 for non-missing values to catch bugs if group is not found
        gix = skipmissing && missings[i] ? 1 : 0
        probe = 0
        # Skip rows contaning at least one missing (assigning them to group 0)
        if !skipmissing || !missings[i]
            while true
                g_row = gslots[slotix]
                if g_row == 0 # unoccupied slot, current row starts a new group
                    gslots[slotix] = i
                    gix = ngroups += 1
                    break
                elseif rhashes[i] == rhashes[g_row] # occupied slot, check if miss or hit
                    if isequal_row(cols, i, g_row) # hit
                        gix = groups !== nothing ? groups[g_row] : 0
                    end
                    break
                end
                slotix = slotix & szm1 + 1 # check the next slot
                probe += 1
                @assert probe < sz
            end
        end
        if groups !== nothing
            groups[i] = gix
        end
    end
    return ngroups, rhashes, gslots
end
function group_rows(df::AbstractDataFrame, skipmissing::Bool = false)
    # note: `f` must return a consistent length
    res = DataFrame()
    for (n, v) in eachcol(dfci.df)
        res[n] = f(v)
    end
    res
end
similar_missing(dv::AbstractArray{T}, dims::Union{Int, Tuple{Vararg{Int}}}) where {T} =
    fill!(similar(dv, Union{T, Missing}, dims), missing)
const OnType = Union{Symbol, NTuple{2,Symbol}, Pair{Symbol,Symbol}}
struct DataFrameJoiner{DF1<:AbstractDataFrame, DF2<:AbstractDataFrame}
    dfl::DF1
    dfr::DF2
    dfl_on::DF1
    dfr_on::DF2
    left_on::Vector{Symbol}
    right_on::Vector{Symbol}
    function DataFrameJoiner{DF1, DF2}(dfl::DF1, dfr::DF2,
                                       on::Union{<:OnType, AbstractVector{<:OnType}}) where {DF1, DF2}
        on_cols = isa(on, Vector) ? on : [on]
        if eltype(on_cols) == Symbol
            left_on = on_cols
            right_on = on_cols
        else
            left_on = [first(x) for x in on_cols]
            right_on = [last(x) for x in on_cols]
        end
        new(dfl, dfr, dfl[left_on], dfr[right_on], left_on, right_on)
    end
end
DataFrameJoiner(dfl::DF1, dfr::DF2, on::Union{<:OnType, AbstractVector{<:OnType}}) where
    {DF1<:AbstractDataFrame, DF2<:AbstractDataFrame} =
    DataFrameJoiner{DF1,DF2}(dfl, dfr, on)
struct RowIndexMap
    "row indices in the original table"
    orig::Vector{Int}
    "row indices in the resulting joined table"
    join::Vector{Int}
end
Base.length(x::RowIndexMap) = length(x.orig)
function compose_joined_table(joiner::DataFrameJoiner, kind::Symbol,
                              left_ixs::RowIndexMap, leftonly_ixs::RowIndexMap,
                              right_ixs::RowIndexMap, rightonly_ixs::RowIndexMap;
                              makeunique::Bool=false)
    @assert length(left_ixs) == length(right_ixs)
    # compose left half of the result taking all left columns
    dfr_noon = without(joiner.dfr, joiner.right_on)
    nrow = length(all_orig_left_ixs) + roil
    @assert nrow == length(all_orig_right_ixs) + loil
    ncleft = ncol(joiner.dfl)
    cols = Vector{Any}(ncleft + ncol(dfr_noon))
    # inner and left joins preserve non-missingness of the left frame
    _similar_left = kind == :inner || kind == :left ? similar : similar_missing
    for (i, col) in enumerate(columns(joiner.dfl))
        cols[i] = _similar_left(col, nrow)
        copy!(cols[i], view(col, all_orig_left_ixs))
    end
    # inner and right joins preserve non-missingness of the right frame
    _similar_right = kind == :inner || kind == :right ? similar : similar_missing
    for (i, col) in enumerate(columns(dfr_noon))
        cols[i+ncleft] = _similar_right(col, nrow)
        copy!(cols[i+ncleft], view(col, all_orig_right_ixs))
        permute!(cols[i+ncleft], right_perm)
    end
    res = DataFrame(cols, vcat(names(joiner.dfl), names(dfr_noon)), makeunique=makeunique)
    if length(rightonly_ixs.join) > 0
        # some left rows are missing, so the values of the "on" columns
        # need to be taken from the right
        for (on_col_ix, on_col) in enumerate(joiner.left_on)
            # fix the result of the rightjoin by taking the nonmissing values from the right table
            offset = nrow - length(rightonly_ixs.orig) + 1
            copy!(res[on_col], offset, view(joiner.dfr_on[on_col_ix], rightonly_ixs.orig))
        end
    end
    if kind ∈ (:right, :outer) && !isempty(rightonly_ixs.join)
        # At this point on-columns of the result allow missing values, because
        # right-only rows were filled with missing values when processing joiner.dfl
        # However, when the right on-column (plus the left one for the outer join)
        # does not allow missing values, the result should also disallow them.
        for (on_col_ix, on_col) in enumerate(joiner.left_on)
            LT = eltype(joiner.dfl_on[on_col_ix])
            RT = eltype(joiner.dfr_on[on_col_ix])
            if !(RT >: Missing) && (kind == :right || !(LT >: Missing))
                res[on_col] = disallowmissing(res[on_col])
            end
        end
    end
    return res
end
function update_row_maps!(left_table::AbstractDataFrame,
                          right_table::AbstractDataFrame,
                          right_dict::RowGroupDict,
                          left_ixs::Union{Void, RowIndexMap},
                          leftonly_ixs::Union{Void, RowIndexMap},
                          right_ixs::Union{Void, RowIndexMap},
                          rightonly_mask::Union{Void, Vector{Bool}})
    # helper functions
    @inline update!(ixs::Void, orig_ix::Int, join_ix::Int, count::Int = 1) = nothing
    @inline function update!(ixs::RowIndexMap, orig_ix::Int, join_ix::Int, count::Int = 1)
        n = length(ixs.orig)
        resize!(ixs.orig, n+count)
        ixs.orig[n+1:end] = orig_ix
        append!(ixs.join, join_ix:(join_ix+count-1))
        ixs
    end
    @inline update!(ixs::Void, orig_ixs::AbstractArray, join_ix::Int) = nothing
    @inline function update!(ixs::RowIndexMap, orig_ixs::AbstractArray, join_ix::Int)
        append!(ixs.orig, orig_ixs)
        append!(ixs.join, join_ix:(join_ix+length(orig_ixs)-1))
        ixs
    end
    @inline update!(ixs::Void, orig_ixs::AbstractArray) = nothing
    @inline update!(mask::Vector{Bool}, orig_ixs::AbstractArray) = (mask[orig_ixs] = false)
    # iterate over left rows and compose the left<->right index map
    right_dict_cols = ntuple(i -> right_dict.df[i], ncol(right_dict.df))
    droplevels!(keycol)
    valuecol = df[value]
    _unstack(df, rowkey, colkey, value, keycol, valuecol, refkeycol)
end
function _unstack(df::AbstractDataFrame, rowkey::Int,
                  colkey::Int, value::Int, keycol, valuecol, refkeycol)
    Nrow = length(refkeycol.pool)
    Ncol = length(keycol.pool)
    unstacked_val = [similar_missing(valuecol, Nrow) for i in 1:Ncol]
    hadmissing = false # have we encountered missing in refkeycol
    mask_filled = falses(Nrow+1, Ncol) # has a given [row,col] entry been filled?
    warned_dup = false # have we already printed duplicate entries warning?
    warned_missing = false # have we already printed missing in keycol warning?
    keycol_order = Vector{Int}(CategoricalArrays.order(keycol.pool))
    refkeycol_order = Vector{Int}(CategoricalArrays.order(refkeycol.pool))
    for k in 1:nrow(df)
        kref = keycol.refs[k]
        if kref <= 0 # we have found missing in colkey
            if !warned_missing
                warn("Missing value in variable $(_names(df)[colkey]) at row $k. Skipping.")
                warned_missing = true
            end
            continue # skip processing it
        end
        j = keycol_order[kref]
        refkref = refkeycol.refs[k]
        if refkref <= 0 # we have found missing in rowkey
            if !hadmissing # if it is the first time we have to add a new row
                hadmissing = true
                # we use the fact that missing is greater than anything
                for i in eachindex(unstacked_val)
                    push!(unstacked_val[i], missing)
                end
            end
            i = length(unstacked_val[1])
        else
            i = refkeycol_order[refkref]
        end
        if !warned_dup && mask_filled[i, j]
            warn("Duplicate entries in unstack at row $k for key "*
                 "$(refkeycol[k]) and variable $(keycol[k]).")
            warned_dup = true
        end
        unstacked_val[j][i] = valuecol[k]
        mask_filled[i, j] = true
    end
    keycol = categorical(df[colkey])
    droplevels!(keycol)
    valuecol = df[value]
    _unstack(df, rowkeys, colkey, value, keycol, valuecol, g)
end
function _unstack(df::AbstractDataFrame, rowkeys::AbstractVector{Symbol},
                  colkey::Int, value::Int, keycol, valuecol, g)
    groupidxs = [g.idx[g.starts[i]:g.ends[i]] for i in 1:length(g.starts)]
    rowkey = zeros(Int, size(df, 1))
    for i in 1:length(groupidxs)
        rowkey[groupidxs[i]] = i
    end
    df1 = df[g.idx[g.starts], g.cols]
    Nrow = length(g)
    Ncol = length(levels(keycol))
    unstacked_val = [similar_missing(valuecol, Nrow) for i in 1:Ncol]
    mask_filled = falses(Nrow, Ncol)
    warned_dup = false
    warned_missing = false
    keycol_order = Vector{Int}(CategoricalArrays.order(keycol.pool))
    for k in 1:nrow(df)
        kref = keycol.refs[k]
        if kref <= 0
            if !warned_missing
                warn("Missing value in variable $(_names(df)[colkey]) at row $k. Skipping.")
                warned_missing = true
            end
            continue
        end
        j = keycol_order[kref]
        i = rowkey[k]
        if !warned_dup && mask_filled[i, j]
            warn("Duplicate entries in unstack at row $k for key "*
                 "$(tuple((df[1,s] for s in rowkeys)...)) and variable $(keycol[k]).")
            warned_dup = true
        end
        unstacked_val[j][i] = valuecol[k]
        mask_filled[i, j] = true
    end
    df2 = DataFrame(unstacked_val, map(Symbol, levels(keycol)))
    hcat(df1, df2)
end
unstack(df::AbstractDataFrame) = unstack(df, :id, :variable, :value)
mutable struct StackedVector <: AbstractVector{Any}
    components::Vector{Any}
end
Base.eltype(v::StackedVector) = promote_type(map(eltype, v.components)...)
Base.similar(v::StackedVector, T::Type, dims::Union{Integer, AbstractUnitRange}...) =
    similar(v.components[1], T, dims...)
CategoricalArrays.CategoricalArray(v::StackedVector) = CategoricalArray(v[:]) # could be more efficient
mutable struct RepeatedVector{T} <: AbstractVector{T}
    parent::AbstractVector{T}
    inner::Int
    outer::Int
end
function Base.getindex(v::RepeatedVector{T},i::AbstractVector{I}) where {T,I<:Real}
    N = length(v.parent)
    idx = Int[Base.fld1(mod1(j,v.inner*N),v.inner) for j in i]
    v.parent[idx]
end
function Base.getindex(v::RepeatedVector{T},i::Real) where T
    N = length(v.parent)
    idx = Base.fld1(mod1(i,v.inner*N),v.inner)
    v.parent[idx]
end
Base.getindex(v::RepeatedVector,i::AbstractRange) = getindex(v, [i;])
Base.size(v::RepeatedVector) = (length(v),)
Base.length(v::RepeatedVector) = v.inner * v.outer * length(v.parent)
Base.ndims(v::RepeatedVector) = 1
Base.eltype(v::RepeatedVector{T}) where {T} = T
Base.reverse(v::RepeatedVector) = RepeatedVector(reverse(v.parent), v.inner, v.outer)
Base.similar(v::RepeatedVector, T, dims::Dims) = similar(v.parent, T, dims)
Base.unique(v::RepeatedVector) = unique(v.parent)
function CategoricalArrays.CategoricalArray(v::RepeatedVector)
    res = CategoricalArrays.CategoricalArray(v.parent)
    res.refs = repeat(res.refs, inner = [v.inner], outer = [v.outer])
    res
end
function stackdf(df::AbstractDataFrame, measure_vars::AbstractVector{<:Integer},
                 id_vars::AbstractVector{<:Integer}; variable_name::Symbol=:variable,
                 value_name::Symbol=:value)
    N = length(measure_vars)
    cnames = names(df)[id_vars]
    insert!(cnames, 1, value_name)
    insert!(cnames, 1, variable_name)
    DataFrame(Any[RepeatedVector(_names(df)[measure_vars], nrow(df), 1),   # variable
                  StackedVector(Any[df[:,c] for c in measure_vars]),     # value
                  [RepeatedVector(df[:,c], 1, N) for c in id_vars]...],     # id_var columns
              cnames)
end
function stackdf(df::AbstractDataFrame, measure_var::Int, id_var::Int;
                 variable_name::Symbol=:variable, value_name::Symbol=:value)
end
function meltdf(df::AbstractDataFrame, id_vars; variable_name::Symbol=:variable,
                value_name::Symbol=:value)
    id_inds = index(df)[id_vars]
    stackdf(df, setdiff(1:ncol(df), id_inds), id_inds;
            variable_name=variable_name, value_name=value_name)
end
function meltdf(df::AbstractDataFrame, id_vars, measure_vars;
                variable_name::Symbol=:variable, value_name::Symbol=:value)
    stackdf(df, measure_vars, id_vars; variable_name=variable_name,
            value_name=value_name)
end
meltdf(df::AbstractDataFrame; variable_name::Symbol=:variable, value_name::Symbol=:value) =
    stackdf(df; variable_name=variable_name, value_name=value_name)
function escapedprint(io::IO, x::Any, escapes::AbstractString)
    ourshowcompact(io, x)
end
function escapedprint(io::IO, x::AbstractString, escapes::AbstractString)
    escape_string(io, x, escapes)
end
function printtable(io::IO,
                    df::AbstractDataFrame;
                    header::Bool = true,
                    separator::Char = ',',
                    quotemark::Char = '"',
                    nastring::AbstractString = "missing")
    n, p = size(df)
    etypes = eltypes(df)
    if header
        cnames = _names(df)
        for j in 1:p
            print(io, quotemark)
            print(io, cnames[j])
            print(io, quotemark)
            if j < p
                print(io, separator)
            else
                print(io, '\n')
            end
        end
    end
    quotestr = string(quotemark)
    for i in 1:n
        for j in 1:p
            if !ismissing(df[j][i])
                if ! (etypes[j] <: Real)
                    print(io, quotemark)
                    escapedprint(io, df[i, j], quotestr)
                    print(io, quotemark)
                else
                    print(io, df[i, j])
                end
            else
                print(io, nastring)
            end
            if j < p
                print(io, separator)
            else
                print(io, '\n')
            end
        end
    end
    return
end
function printtable(df::AbstractDataFrame;
                    header::Bool = true,
                    separator::Char = ',',
                    quotemark::Char = '"',
                    nastring::AbstractString = "missing")
    ncols = size(df, 2)
    cnames = _names(df)
    alignment = repeat("c", ncols)
    write(io, "\\begin{tabular}{r|")
    write(io, alignment)
    write(io, "}\n")
    write(io, "\t& ")
    header = join(map(c -> latex_escape(string(c)), cnames), " & ")
    write(io, header)
    write(io, "\\\\\n")
    write(io, "\t\\hline\n")
    for row in 1:nrows
        write(io, "\t")
        write(io, @sprintf("%d", row))
        for col in 1:ncols
            write(io, " & ")
            cell = df[row,col]
            if !ismissing(cell)
                if mimewritable(MIME("text/latex"), cell)
                    show(io, MIME("text/latex"), cell)
                else
                    print(io, latex_escape(sprint(ourshowcompact, cell)))
                end
            end
        end
        write(io, " \\\\\n")
    end
    write(io, "\\end{tabular}\n")
end
function Base.show(io::IO, ::MIME"text/csv", df::AbstractDataFrame)
    printtable(io, df, true, ',')
end
function Base.show(io::IO, ::MIME"text/tab-separated-values", df::AbstractDataFrame)
    printtable(io, df, true, '\t')
end
using DataStreams, WeakRefStrings
struct DataFrameStream{T}
    columns::T
    header::Vector{String}
end
DataFrameStream(df::DataFrame) = DataFrameStream(Tuple(df.columns), string.(names(df)))
function Data.schema(df::DataFrame)
    return Data.Schema(Type[eltype(A) for A in df.columns],
                       string.(names(df)), length(df) == 0 ? 0 : length(df.columns[1]))
end
Data.isdone(source::DataFrame, row, col, rows, cols) = row > rows || col > cols
allocate(::Type{Missing}, rows, ref) = missings(rows)
function DataFrame(sch::Data.Schema{R}, ::Type{S}=Data.Field,
                   append::Bool=false, args...;
                   reference::Vector{UInt8}=UInt8[]) where {R, S <: Data.StreamType}
    types = Data.types(sch)
    if !isempty(args) && args[1] isa DataFrame && types == Data.types(Data.schema(args[1]))
        # passing in an existing DataFrame Sink w/ same types as source
        sink = args[1]
        sinkrows = size(Data.schema(sink), 1)
        # are we appending and either column-streaming or there are an unknown # of rows
        if append && (S == Data.Column || !R)
            sch.rows = sinkrows
            # dont' need to do anything because:
              # for Data.Column, we just append columns anyway (see Data.streamto! below)
              # for Data.Field, unknown # of source rows, so we'll just push! in streamto!
        else
            # need to adjust the existing sink
            # similar to above, for Data.Column or unknown # of rows for Data.Field,
                # we'll append!/push! in streamto!, so we empty! the columns
            # if appending, we want to grow our columns to be able to include every row
                # in source (sinkrows + sch.rows)
            # if not appending, we're just "re-using" a sink, so we just resize it
                # to the # of rows in the source
            newsize = ifelse(S == Data.Column || !R, 0,
                        ifelse(append, sinkrows + sch.rows, sch.rows))
            foreach(col->resize!(col, newsize), sink.columns)
            sch.rows = newsize
        end
        # take care of a possible reference from source by addint to WeakRefStringArrays
        if !isempty(reference)
            foreach(col-> col isa WeakRefStringArray && push!(col.data, reference),
                sink.columns)
        end
        sink = DataFrameStream(sink)
    else
        # allocating a fresh DataFrame Sink; append is irrelevant
        # for Data.Column or unknown # of rows in Data.Field, we only ever append!,
            # so just allocate empty columns
        rows = ifelse(S == Data.Column, 0, ifelse(!R, 0, sch.rows))
        names = Data.header(sch)
        sink = DataFrameStream(
                Tuple(allocate(types[i], rows, reference) for i = 1:length(types)), names)
        sch.rows = rows
    end
    return sink
end
DataFrame(sink, sch::Data.Schema, ::Type{S}, append::Bool;
          reference::Vector{UInt8}=UInt8[]) where {S} =
    DataFrame(sch, S, append, sink; reference=reference)
@inline Data.streamto!(sink::DataFrameStream, ::Type{Data.Field}, val,
                      row, col::Int) =
    (A = sink.columns[col]; row > length(A) ? push!(A, val) : setindex!(A, val, row))
@inline Data.streamto!(sink::DataFrameStream, ::Type{Data.Field}, val,
                       row, col::Int, ::Type{Val{false}}) =
    push!(sink.columns[col], val)
@inline Data.streamto!(sink::DataFrameStream, ::Type{Data.Field}, val,
                       row, col::Int, ::Type{Val{true}}) =
    sink.columns[col][row] = val
@inline function Data.streamto!(sink::DataFrameStream, ::Type{Data.Column}, column,
                       row, col::Int, knownrows)
    function ourstrwidth(x::Any) # -> Int
        truncate(io, 0)
        ourshowcompact(io, x)
        textwidth(String(take!(io)))
    end
end
ourshowcompact(io::IO, x::Any) = showcompact(io, x) # -> Void
ourshowcompact(io::IO, x::AbstractString) = escape_string(io, x, "") # -> Void
ourshowcompact(io::IO, x::Symbol) = ourshowcompact(io, string(x)) # -> Void
function getmaxwidths(df::AbstractDataFrame,
                      rowindices1::AbstractVector{Int},
                      rowindices2::AbstractVector{Int},
                      rowlabel::Symbol) # -> Vector{Int}
    maxwidths = Vector{Int}(size(df, 2) + 1)
    undefstrwidth = ourstrwidth(Base.undef_ref_str)
    j = 1
    for (name, col) in eachcol(df)
        # (1) Consider length of column name
        maxwidth = ourstrwidth(name)
        # (2) Consider length of longest entry in that column
        for indices in (rowindices1, rowindices2), i in indices
            try
                ourshowcompact(io, Base.undef_ref_str)
            end
            padding = maxwidths[j] - strlen
            for _ in 1:padding
                write(io, ' ')
            end
            if j == rightcol
                if i == rowindices[end]
                    print(io, " │")
                else
                    print(io, " │\n")
                end
            else
                print(io, " │ ")
            end
        end
    end
    return
end
function showrows(io::IO,
                  df::AbstractDataFrame,
                  rowindices1::AbstractVector{Int},
                  rowindices2::AbstractVector{Int},
                  maxwidths::Vector{Int},
                  splitchunks::Bool = false,
                  allcols::Bool = true,
                  rowlabel::Symbol = :Row,
                  displaysummary::Bool = true) # -> Void
    ncols = size(df, 2)
    if isempty(rowindices1)
        if displaysummary
            println(io, summary(df))
        end
        leftcol = chunkbounds[chunkindex] + 1
        for itr in 1:(rowmaxwidth + 2)
            write(io, '─')
        end
        write(io, '┼')
        for j in leftcol:rightcol
            for itr in 1:(maxwidths[j] + 2)
                write(io, '─')
            end
            if j < rightcol
                write(io, '┼')
            else
                write(io, '┤')
            end
        end
        write(io, '\n')
        # Print main table body, potentially in two abbreviated sections
        showrowindices(io,
                       df,
                       rowindices1,
                       maxwidths,
                       leftcol,
                       rightcol)
        if !isempty(rowindices2)
            print(io, "\n⋮\n")
            showrowindices(io,
                           df,
                           rowindices2,
                           maxwidths,
                           leftcol,
                           rightcol)
        end
        # Print newlines to separate chunks
        if chunkindex < nchunks
            print(io, "\n\n")
        end
    end
    return
end
function Base.show(io::IO,
                   df::AbstractDataFrame,
                   allcols::Bool = false,
                   rowlabel::Symbol = :Row,
                   displaysummary::Bool = true) # -> Void
    nrows = size(df, 1)
    dsize = displaysize(io)
    availableheight = dsize[1] - 5
    nrowssubset = fld(availableheight, 2)
    bound = min(nrowssubset - 1, nrows)
    if nrows <= availableheight
        rowindices1 = 1:nrows
        rowindices2 = 1:0
    else
        rowindices1 = 1:bound
        rowindices2 = max(bound + 1, nrows - nrowssubset + 1):nrows
    end
    maxwidths = getmaxwidths(df, rowindices1, rowindices2, rowlabel)
    width = getprintedwidth(maxwidths)
    showrows(io,
             df,
             rowindices1,
             rowindices2,
             maxwidths,
             true,
             allcols,
             rowlabel,
             displaysummary)
    return
end
function Base.show(df::AbstractDataFrame,
                   allcols::Bool = false) # -> Void
    return show(STDOUT, df, allcols)
end
function Base.showall(io::IO,
                      df::AbstractDataFrame,
                      allcols::Bool = true,
                      rowlabel::Symbol = :Row,
                      displaysummary::Bool = true) # -> Void
    rowindices1 = 1:size(df, 1)
    rowindices2 = 1:0
    maxwidths = getmaxwidths(df, rowindices1, rowindices2, rowlabel)
    width = getprintedwidth(maxwidths)
    showrows(io,
             df,
             rowindices1,
             rowindices2,
             maxwidths,
             !allcols,
             allcols,
             rowlabel,
             displaysummary)
    return
end
function showcols(df::AbstractDataFrame, all::Bool=false, values::Bool=true)
    showcols(STDOUT, df, all, values) # -> Void
end
function Base.show(io::IO, gd::GroupedDataFrame)
    N = length(gd)
    println(io, "$(typeof(gd))  $N groups with keys: $(gd.cols)")
    N = length(gd)
    println(io, "$(typeof(gd))  $N groups with keys: $(gd.cols)")
    for i = 1:N
        println(io, "gd[$i]:")
        show(io, gd[i])
    end
end
function Base.show(io::IO, r::DataFrameRow)
    labelwidth = mapreduce(n -> length(string(n)), max, _names(r)) + 2
    @printf(io, "DataFrameRow (row %d)\n", r.row)
    for (label, value) in r
        println(io, rpad(label, labelwidth, ' '), value)
    end
end
mutable struct UserColOrdering{T<:ColumnIndex}
    col::T
    kwargs
end
order(col::T; kwargs...) where {T<:ColumnIndex} = UserColOrdering{T}(col, kwargs)
_getcol(o::UserColOrdering) = o.col
_getcol(x) = x
function ordering(col_ord::UserColOrdering, lt::Function, by::Function, rev::Bool, order::Ordering)
    for (k,v) in kwpairs(col_ord.kwargs)
        if     k == :lt;    lt    = v
        elseif k == :by;    by    = v
        elseif k == :rev;   rev   = v
        elseif k == :order; order = v
        else
            error("Unknown keyword argument: ", string(k))
        end
    end
    Order.ord(lt,by,rev,order)
end
ordering(col::ColumnIndex, lt::Function, by::Function, rev::Bool, order::Ordering) =
             Order.ord(lt,by,rev,order)
struct DFPerm{O<:Union{Ordering, AbstractVector}, DF<:AbstractDataFrame} <: Ordering
    ord::O
    df::DF
end
function DFPerm(ords::AbstractVector{O}, df::DF) where {O<:Ordering, DF<:AbstractDataFrame}
    if length(ords) != ncol(df)
        error("DFPerm: number of column orderings does not equal the number of DataFrame columns")
    end
    DFPerm{typeof(ords), DF}(ords, df)
end
DFPerm(o::O, df::DF) where {O<:Ordering, DF<:AbstractDataFrame} = DFPerm{O,DF}(o,df)
col_ordering(o::DFPerm{O}, i::Int) where {O<:Ordering} = o.ord
col_ordering(o::DFPerm{V}, i::Int) where {V<:AbstractVector} = o.ord[i]
Base.@propagate_inbounds Base.getindex(o::DFPerm, i::Int, j::Int) = o.df[i, j]
Base.@propagate_inbounds Base.getindex(o::DFPerm, a::DataFrameRow, j::Int) = a[j]
function Sort.lt(o::DFPerm, a, b)
    @inbounds for i = 1:ncol(o.df)
        throw(ArgumentError("All ordering arguments must be 1 or the same length."))
    end
    if length(cols) == 0
        return ordering(df, lt, by, rev, order)
    end
    if length(lt) != length(cols)
        throw(ArgumentError("All ordering arguments must be 1 or the same length as the number of columns requested."))
    end
    if length(cols) == 1
        return ordering(df, cols[1], lt[1], by[1], rev[1], order[1])
    end
    # Collect per-column ordering info
    ords = Ordering[]
    newcols = Int[]
    to_array(src::AbstractVector, dims) = src
    to_array(src::Tuple, dims) = [src...]
    to_array(src, dims) = fill(src, dims)
    dims = length(cols) > 0 ? length(cols) : size(df,2)
    ordering(df, cols,
             to_array(lt, dims),
             to_array(by, dims),
             to_array(rev, dims),
             to_array(order, dims))
end
ordering(df::AbstractDataFrame, cols::Tuple, args...) = ordering(df, [cols...], args...)
Sort.defalg(df::AbstractDataFrame, col    ::ColumnIndex,     o::Ordering) = Sort.defalg(df, eltype(df[col]), o)
Sort.defalg(df::AbstractDataFrame, col_ord::UserColOrdering, o::Ordering) = Sort.defalg(df, col_ord.col, o)
Sort.defalg(df::AbstractDataFrame, cols,                     o::Ordering) = Sort.defalg(df)
function Sort.defalg(df::AbstractDataFrame, o::Ordering; alg=nothing, cols=[])
    alg != nothing && return alg
    Sort.defalg(df, cols, o)
end
Base.issorted(df::AbstractDataFrame; cols=Any[], lt=isless, by=identity, rev=false, order=Forward) =
    issorted(eachrow(df), ordering(df, cols, lt, by, rev, order))
for s in [:(Base.sort), :(Base.sortperm)]
    @eval begin
        function $s(df::AbstractDataFrame; cols=Any[], alg=nothing,
                    lt=isless, by=identity, rev=false, order=Forward)
            if !(isa(by, Function) || eltype(by) <: Function)
                msg = "'by' must be a Function or a vector of Functions. Perhaps you wanted 'cols'."
                throw(ArgumentError(msg))
            end
            ord = ordering(df, cols, lt, by, rev, order)
            _alg = Sort.defalg(df, ord; alg=alg, cols=cols)
            $s(df, _alg, ord)
        end
    end
end
Base.sort(df::AbstractDataFrame, a::Algorithm, o::Ordering) = df[sortperm(df, a, o),:]
Base.sortperm(df::AbstractDataFrame, a::Algorithm, o::Union{Perm,DFPerm}) = sort!([1:size(df, 1);], a, o)
Base.sortperm(df::AbstractDataFrame, a::Algorithm, o::Ordering) = sortperm(df, a, DFPerm(o,df))
function Base.sort!(df::DataFrame; cols=Any[], alg=nothing,
                    lt=isless, by=identity, rev=false, order=Forward)
    if !(isa(by, Function) || eltype(by) <: Function)
        msg = "'by' must be a Function or a vector of Functions. Perhaps you wanted 'cols'."
        throw(ArgumentError(msg))
    end
    ord = ordering(df, cols, lt, by, rev, order)
    _alg = Sort.defalg(df, ord; alg=alg, cols=cols)
    sort!(df, _alg, ord)
end
function Base.sort!(df::DataFrame, a::Base.Sort.Algorithm, o::Base.Sort.Ordering)
    p = sortperm(df, a, o)
    pp = similar(p)
    c = columns(df)
    for (i,col) in enumerate(c)
        # Check if this column has been sorted already
        if any(j -> c[j]===col, 1:i-1)
            continue
        end
        if header
            if any(i -> Symbol(file_df[1, i]) != index(df)[i], 1:size(df, 2))
                throw(KeyError("Column names don't match names in file"))
            end
            header = false
        end
    end
    encoder = endswith(filename, ".gz") ? GzipCompressorStream : NoopStream
    open(encoder, filename, append ? "a" : "w") do io
        printtable(io,
                   df,
                   header = header,
                   separator = separator,
                   quotemark = quotemark,
                   nastring = nastring)
    end
    return
end
struct ParsedCSV
    bytes::Vector{UInt8} # Raw bytes from CSV file
    bounds::Vector{Int}  # Right field boundary indices
end
macro skip_to_eol(io, chr, nextchr, endf)
    io = esc(io)
    chr = esc(chr)
end
macro atescape(chr, nextchr, quotemarks)
    chr = esc(chr)
    nextchr = esc(nextchr)
    quotemarks = esc(quotemarks)
    quote
        (UInt32($chr) == UInt32('\\') &&
            (UInt32($nextchr) == UInt32('\\') ||
                UInt32($nextchr) in $quotemarks)) ||
                    (UInt32($chr) == UInt32($nextchr) &&
                        UInt32($chr) in $quotemarks)
    end
end
macro atcescape(chr, nextchr)
    chr = esc(chr)
    nextchr = esc(nextchr)
    quote
        $chr == UInt32('\\') &&
        ($nextchr == UInt32('n') ||
         $nextchr == UInt32('t') ||
         $nextchr == UInt32('r') ||
         $nextchr == UInt32('a') ||
         $nextchr == UInt32('b') ||
         $nextchr == UInt32('f') ||
         $nextchr == UInt32('v') ||
         $nextchr == UInt32('\\'))
    end
end
macro mergechr(chr, nextchr)
    chr = esc(chr)
    nextchr = esc(nextchr)
    quote
        if $chr == UInt32('\\')
            if $nextchr == UInt32('n')
                '\n'
            elseif $nextchr == UInt32('t')
                '\t'
            elseif $nextchr == UInt32('r')
                '\r'
            elseif $nextchr == UInt32('a')
                '\a'
                chr, nextchr, endf = @read_peek_eof(io, nextchr)
                # === Debugging ===
                # if in_quotes
                #     print_with_color(:red, string(char(chr)))
                # else
                #     print_with_color(:green, string(char(chr)))
                # end
                $(if allowcomments
                    quote
                        # Ignore text inside comments completely
                        # Merge chr and nextchr here if they're a c-style escape
                        if @atcescape(chr, nextchr) && !in_escape
                            chr = @mergechr(chr, nextchr)
                            nextchr = eof(io) ? 0xff : read(io, UInt8)
                            endf = nextchr == 0xff
                            in_escape = true
                        end
                    end
                end)
                # No longer at the start of a line that might be a pure comment
                $(if allowcomments quote at_start = false end end)
                # Processing is very different inside and outside of quotes
                if !in_quotes
                    # Entering a quoted region
                    if chr in quotemarks
                        in_quotes = true
                        p.quoted[n_fields] = true
                        $(if wsv quote skip_white = false end end)
                    # Finished reading a field
                        @skip_within_eol(io, chr, nextchr, endf)
                        $(if allowcomments quote at_start = true end end)
                        @push(n_bounds, p.bounds, n_bytes, l_bounds)
                        @push(n_bytes, p.bytes, '\n', l_bytes)
                        @push(n_lines, p.lines, n_bytes, l_lines)
                        @push(n_fields, p.quoted, false, l_quoted)
                        $(if wsv quote skip_white = true end end)
                    # Store character in buffer
                    else
                        @push(n_bytes, p.bytes, chr, l_bytes)
                        $(if wsv quote skip_white = false end end)
                    end
                else
                    # Escape a quotemark inside quoted field
                    if @atescape(chr, nextchr, quotemarks) && !in_escape
                        in_escape = true
                    else
                        # Exit quoted field
                        if UInt32(chr) in quotemarks && !in_escape
                            in_quotes = false
                        # Store character in buffer
                        else
                            @push(n_bytes, p.bytes, chr, l_bytes)
                        end
                        # Escape mode only lasts for one byte
                        in_escape = false
                    end
                end
            end
            # Append a final EOL if it's missing in the raw input
            if endf && !@atnewline(chr, nextchr)
                @push(n_bounds, p.bounds, n_bytes, l_bounds)
                @push(n_bytes, p.bytes, '\n', l_bytes)
                @push(n_lines, p.lines, n_bytes, l_lines)
            end
            # Don't count the dummy boundaries in fields or rows
            return n_bytes, n_bounds - 1, n_lines - 1, nextchr
        end
    end
end
function bytematch(bytes::Vector{UInt8},
                   left::Integer,
                   right::Integer,
                   exemplars::Vector{T}) where T <: String
    l = right - left + 1
    for index in 1:length(exemplars)
        exemplar = exemplars[index]
        if length(exemplar) == l
            matched = true
            for i in 0:(l - 1)
                matched &= bytes[left + i] == UInt32(exemplar[1 + i])
            end
            if matched
                return true
            end
        end
    end
    return false
    if left > right
        return 0, true, true
    end
    if bytematch(bytes, left, right, nastrings)
        return 0, true, true
    end
    value = 0
    power = 1
    index = right
    byte = bytes[index]
    while index > left
        return out[1], wasparsed, false
    end
end
function bytestotype(::Type{N},
                     bytes::Vector{UInt8},
                     left::Integer,
                     right::Integer,
                     nastrings::Vector{T},
                     wasquoted::Bool = false,
                     truestrings::Vector{P} = P[],
                     falsestrings::Vector{P} = P[]) where {N <: Bool,
                                                           T <: String,
                                                           P <: String}
    if left > right
        return false, true, true
    end
    if bytematch(bytes, left, right, nastrings)
        return false, true, true
    end
    if bytematch(bytes, left, right, truestrings)
        return true, true, false
    elseif bytematch(bytes, left, right, falsestrings)
        return false, true, false
    else
        return false, false, false
    end
end
function bytestotype(::Type{N},
                     bytes::Vector{UInt8},
                     left::Integer,
                     right::Integer,
                     nastrings::Vector{T},
                     wasquoted::Bool = false,
                     truestrings::Vector{P} = P[],
                     falsestrings::Vector{P} = P[]) where {N <: AbstractString,
                                                           T <: String,
                                                           P <: String}
    if left > right
        if wasquoted
            return "", true, false
        else
            return "", true, true
        end
    end
    columns = Vector{Any}(cols)
    for j in 1:cols
        if isempty(o.eltypes)
            values = Vector{Int}(rows)
        else
            values = Vector{o.eltypes[j]}(rows)
        end
        msng = falses(rows)
        is_int = true
        is_float = true
        name = String(bytes[left:right])
        if normalizenames
            name = identifier(name)
        end
        names[j] = name
    end
    return
end
function findcorruption(rows::Integer,
                        cols::Integer,
                        fields::Integer,
                        p::ParsedCSV)
    n = length(p.bounds)
    lengths = Vector{Int}(rows)
    t = 1
    for i in 1:rows
        bound = p.lines[i + 1]
    end
    # Use ParseOptions to pick the right method of readnrows!
    d = ParseType(o)
    # Extract the header
    if o.header
        bytes, fields, rows, nextchr = readnrows!(p, io, Int64(1), o, d, nextchr)
        # Insert column names from header if none present
        if isempty(o.names)
            parsenames!(o.names, o.ignorepadding, p.bytes, p.bounds, p.quoted, fields, o.normalizenames)
        end
    end
    # Parse main data set
    bytes, fields, rows, nextchr = readnrows!(p, io, Int64(nrows), o, d, nextchr)
    # Return the resulting DataFrame
    return df
end
export readtable
function readtable(pathname::AbstractString;
                   header::Bool = true,
                   separator::Char = getseparator(pathname),
                   quotemark::Vector{Char} = ['"'],
                   decimal::Char = '.',
                   nastrings::Vector = String["", "NA"],
                   truestrings::Vector = String["T", "t", "TRUE", "true"],
                   falsestrings::Vector = String["F", "f", "FALSE", "false"],
                   makefactors::Bool = false,
                   nrows::Integer = -1,
                   names::Vector = Symbol[],
                       normalizenames = normalizenames)
    # Open an IO stream based on pathname
    # (1) Path is an HTTP or FTP URL
    if startswith(pathname, "http://") || startswith(pathname, "ftp://")
        error("URL retrieval not yet implemented")
    # (2) Path is GZip file
    elseif endswith(pathname, ".gz")
        nbytes = 2 * filesize(pathname)
        io = open(_r, GzipDecompressorStream, pathname, "r")
    # (3) Path is BZip2 file
    elseif endswith(pathname, ".bz") || endswith(pathname, ".bz2")
        error("BZip2 decompression not yet implemented")
    # (4) Path is an uncompressed file
    else
        nbytes = filesize(pathname)
        io = open(_r, pathname, "r")
    end
end
inlinetable(s::AbstractString; args...) = readtable(IOBuffer(s); args...)
function inlinetable(s::AbstractString, flags::AbstractString; args...)
    flagbindings = Dict(
        'f' => (:makefactors, true),
        'c' => (:allowcomments, true),
        'H' => (:header, false) )
    for f in flags
        if haskey(flagbindings, f)
            push!(args, flagbindings[f])
        else
            throw(ArgumentError("Unknown inlinetable flag: $f"))
        end
    end
    readtable(IOBuffer(s); args...)
end
export @csv_str, @csv2_str, @tsv_str, @wsv_str
macro wsv_str(s, flags...)
    Base.depwarn("@wsv_str and the wsv\"\"\" syntax are deprecated. " *
                 "Use CSV.read(IOBuffer(...)) from the CSV package instead.",
                 :wsv_str)
    inlinetable(s, flags...; separator=' ')
end
macro tsv_str(s, flags...)
    Base.depwarn("@tsv_str and the tsv\"\"\" syntax are deprecated." *
                 "Use CSV.read(IOBuffer(...)) from the CSV package instead.",
                 :tsv_str)
    inlinetable(s, flags...; separator='\t')
end
@deprecate rename!(x::AbstractDataFrame, from::AbstractArray, to::AbstractArray) rename!(x, [f=>t for (f, t) in zip(from, to)])
@deprecate rename!(x::AbstractDataFrame, from::Symbol, to::Symbol) rename!(x, from => to)
@deprecate rename!(x::Index, f::Function) rename!(f, x)
@deprecate rename(x::AbstractDataFrame, from::AbstractArray, to::AbstractArray) rename(x, [f=>t for (f, t) in zip(from, to)])
@deprecate rename(x::AbstractDataFrame, from::Symbol, to::Symbol) rename(x, from => to)
@deprecate rename(x::Index, f::Function) rename(f, x)
import Base: |>
@deprecate (|>)(gd::GroupedDataFrame, fs::Function) aggregate(gd, fs)
@deprecate (|>)(gd::GroupedDataFrame, fs::Vector{T}) where {T<:Function} aggregate(gd, fs)
function Base.getindex(x::AbstractIndex, idx::AbstractRange)
    Base.depwarn("Indexing with range of values that are not Integer is deprecated", :getindex)
    getindex(x, collect(idx))
end
function Base.getindex(x::AbstractIndex, idx::AbstractRange{Bool})
    Base.depwarn("Indexing with range of Bool is deprecated", :getindex)
    collect(Int, idx)
end
import Base: vcat
@deprecate vcat(x::Vector{<:AbstractDataFrame}) vcat(x...)
end # module DataFrames
