__precompile__(true)
module DataFrames

##############################################################################
##
## Dependencies
##
##############################################################################

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

##############################################################################
##
## Exported methods and types (in addition to everything reexported above)
##
##############################################################################

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
       eachcol,
       eachrow,
       eltypes,
       groupby,
       head,
       melt,
       meltdf,
       names!,
       ncol,
       nonunique,
       nrow,
       order,
       rename!,
       rename,
       showcols,
       stack,
       stackdf,
       unique!,
       unstack,
       head,
       tail,

       # Remove after deprecation period
       pool,
       pool!


##############################################################################
##
## Load files
##
##############################################################################



#
# expanded from: include("other/utils.jl")
#

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
    dups = Int[]
    for i in 1:length(names)
        name = names[i]
        in(name, seen) ? push!(dups, i) : push!(seen, name)
    end

    if length(dups) > 0
        if !makeunique
            Base.depwarn("Duplicate variable names are deprecated: pass makeunique=true to add a suffix automatically.", :make_unique)
            # TODO: uncomment the lines below after deprecation period
            # msg = """Duplicate variable names: $(u[dups]).
            #          Pass makeunique=true to make them unique using a suffix automatically."""
            # throw(ArgumentError(msg))
        end
    end

    for i in dups
        nm = names[i]
        k = 1
        while true
            newnm = Symbol("$(nm)_$k")
            if !in(newnm, seen)
                names[i] = newnm
                push!(seen, newnm)
                break
            end
            k += 1
        end
    end

    return names
end

"""
    gennames(n::Integer)

Generate standardized names for columns of a DataFrame. The first name will be `:x1`, the
second `:x2`, etc.
"""
function gennames(n::Integer)
    res = Array{Symbol}(n)
    for i in 1:n
        res[i] = Symbol(@sprintf "x%d" i)
    end
    return res
end


"""
    countmissing(a::AbstractArray)

Count the number of `missing` values in an array.
"""
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

# Gets the name of a function. Used in groupeDataFrame/grouping.jl
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


#
# expanded from: include("other/index.jl")
#

# an AbstractIndex is a thing that can be used to look up ordered things by name, but that
# will also accept a position or set of positions or range or other things and pass them
# through cleanly.
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
# Imported in DataFrames.jl for compatibility across Julia 0.4 and 0.5
Base.:(==)(x::Index, y::Index) = isequal(x, y)

# TODO: after deprecation period remove allow_duplicates part of code
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

function rename!(x::Index, nms)
    for (from, to) in nms
        from == to && continue # No change, nothing to do
        if haskey(x, to)
            error("Tried renaming $from to $to, when $to already exists in the Index.")
        end
        x.lookup[to] = col = pop!(x.lookup, from)
        x.names[col] = to
    end
    return x
end

rename!(x::Index, nms::Pair{Symbol,Symbol}...) = rename!(x::Index, collect(nms))
rename!(f::Function, x::Index) = rename!(x, [(x=>f(x)) for x in x.names])

rename(x::Index, args...) = rename!(copy(x), args...)
rename(f::Function, x::Index) = rename!(f, copy(x))

Base.haskey(x::Index, key::Symbol) = haskey(x.lookup, key)
Base.haskey(x::Index, key::Real) = 1 <= key <= length(x.names)
Base.keys(x::Index) = names(x)

# TODO: If this should stay 'unsafe', perhaps make unexported
function Base.push!(x::Index, nm::Symbol)
    x.lookup[nm] = length(x) + 1
    push!(x.names, nm)
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
    delete!(x.lookup, x.names[idx])
    deleteat!(x.names, idx)
    return x
end

function Base.delete!(x::Index, nm::Symbol)
    if !haskey(x.lookup, nm)
        return x
    end
    idx = x.lookup[nm]
    return delete!(x, idx)
end

function Base.empty!(x::Index)
    empty!(x.lookup)
    empty!(x.names)
    x
end

function Base.insert!(x::Index, idx::Integer, nm::Symbol)
    1 <= idx <= length(x.names)+1 || error(BoundsError())
    for i = idx:length(x.names)
        x.lookup[x.names[i]] = i + 1
    end
    x.lookup[nm] = idx
    insert!(x.names, idx, nm)
    x
end

Base.getindex(x::AbstractIndex, idx::Symbol) = x.lookup[idx]
Base.getindex(x::AbstractIndex, idx::AbstractVector{Symbol}) = [x.lookup[i] for i in idx]
Base.getindex(x::AbstractIndex, idx::Integer) = Int(idx)
Base.getindex(x::AbstractIndex, idx::AbstractVector{Int}) = idx
Base.getindex(x::AbstractIndex, idx::AbstractRange{Int}) = idx
Base.getindex(x::AbstractIndex, idx::AbstractRange{<:Integer}) = collect(Int, idx)

function Base.getindex(x::AbstractIndex, idx::AbstractVector{Bool})
    length(x) == length(idx) || throw(BoundsError(x, idx))
    find(idx)
end

function Base.getindex(x::AbstractIndex, idx::AbstractVector{Union{Bool, Missing}})
    if any(ismissing, idx)
        # TODO: this line should be changed to throw an error after deprecation
        Base.depwarn("using missing in column indexing is deprecated", :getindex)
    end
    getindex(x, collect(Missings.replace(idx, false)))
end

function Base.getindex(x::AbstractIndex, idx::AbstractVector{<:Integer})
    # TODO: this line should be changed to throw an error after deprecation
    if any(v -> v isa Bool, idx)
        Base.depwarn("Indexing with Bool values is deprecated except for Vector{Bool}")
    end
    Vector{Int}(idx)
end

# catch all method handling cases when type of idx is not narrowest possible, Any in particular
# also it handles passing missing values in idx
function Base.getindex(x::AbstractIndex, idx::AbstractVector)
    # TODO: passing missing will throw an error after deprecation
    idxs = filter(!ismissing, idx)
    if length(idxs) != length(idx)
        Base.depwarn("using missing in column indexing is deprecated", :getindex)
    end
    length(idxs) == 0 && return Int[] # special case of empty idxs
    if idxs[1] isa Real
        if !all(v -> v isa Integer && !(v isa Bool), idxs)
            # TODO: this line should be changed to throw an error after deprecation
            Base.depwarn("indexing by vector of numbers other than Integer is deprecated", :getindex)
        end
        return Vector{Int}(idxs)
    end
    idxs[1] isa Symbol && return getindex(x, Vector{Symbol}(idxs))
    throw(ArgumentError("idx[1] has type $(typeof(idx[1])); "*
                        "DataFrame only supports indexing columns with integers, symbols or boolean vectors"))
end

# Helpers

function add_names(ind::Index, add_ind::Index; makeunique::Bool=false)
    u = names(add_ind)

    seen = Set(_names(ind))
    dups = Int[]

    for i in 1:length(u)
        name = u[i]
        in(name, seen) ? push!(dups, i) : push!(seen, name)
    end
    if length(dups) > 0
        if !makeunique
            Base.depwarn("Duplicate variable names are deprecated: pass makeunique=true to add a suffix automatically.", :add_names)
            # TODO: uncomment the lines below after deprecation period
            # msg = """Duplicate variable names: $(u[dups]).
            #          Pass makeunique=true to make them unique using a suffix automatically."""
            # throw(ArgumentError(msg))
        end
    end
    for i in dups
        nm = u[i]
        k = 1
        while true
            newnm = Symbol("$(nm)_$k")
            if !in(newnm, seen)
                u[i] = newnm
                push!(seen, newnm)
                break
            end
            k += 1
        end
    end

    return u
end



#
# expanded from: include("abstractdataframe/abstractdataframe.jl")
#


"""
An abstract type for which all concrete types expose a database-like
interface.

**Common methods**

An AbstractDataFrame is a two-dimensional table with Symbols for
column names. An AbstractDataFrame is also similar to an Associative
type in that it allows indexing by a key (the columns).

The following are normally implemented for AbstractDataFrames:

* [`describe`](@ref) : summarize columns
* [`dump`](@ref) : show structure
* `hcat` : horizontal concatenation
* `vcat` : vertical concatenation
* `names` : columns names
* [`names!`](@ref) : set columns names
* [`rename!`](@ref) : rename columns names based on keyword arguments
* [`eltypes`](@ref) : `eltype` of each column
* `length` : number of columns
* `size` : (nrows, ncols)
* [`head`](@ref) : first `n` rows
* [`tail`](@ref) : last `n` rows
* `convert` : convert to an array
* [`completecases`](@ref) : boolean vector of complete cases (rows with no missings)
* [`dropmissing`](@ref) : remove rows with missing values
* [`dropmissing!`](@ref) : remove rows with missing values in-place
* [`nonunique`](@ref) : indexes of duplicate rows
* [`unique!`](@ref) : remove duplicate rows
* `similar` : a DataFrame with similar columns as `d`

**Indexing**

Table columns are accessed (`getindex`) by a single index that can be
a symbol identifier, an integer, or a vector of each. If a single
column is selected, just the column object is returned. If multiple
columns are selected, some AbstractDataFrame is returned.

```julia
d[:colA]
d[3]
d[[:colA, :colB]]
d[[1:3; 5]]
```

Rows and columns can be indexed like a `Matrix` with the added feature
of indexing columns by name.

```julia
d[1:3, :colA]
d[3,3]
d[3,:]
d[3,[:colA, :colB]]
d[:, [:colA, :colB]]
d[[1:3; 5], :]
```

`setindex` works similarly.
"""
abstract type AbstractDataFrame end

##############################################################################
##
## Interface (not final)
##
##############################################################################

# index(df) => AbstractIndex
# nrow(df) => Int
# ncol(df) => Int
# getindex(...)
# setindex!(...) exclusive of methods that add new columns

##############################################################################
##
## Basic properties of a DataFrame
##
##############################################################################

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

# N.B. where stored as a vector, 'columns(x) = x.vector' is a bit cheaper
columns(df::T) where {T <: AbstractDataFrame} = Cols{T}(df)

Base.names(df::AbstractDataFrame) = names(index(df))
_names(df::AbstractDataFrame) = _names(index(df))

"""
Set column names


```julia
names!(df::AbstractDataFrame, vals)
```

**Arguments**

* `df` : the AbstractDataFrame
* `vals` : column names, normally a Vector{Symbol} the same length as
  the number of columns in `df`
* `makeunique` : if `false` (the default), an error will be raised
  if duplicate names are found; if `true`, duplicate names will be suffixed
  with `_i` (`i` starting at 1 for the first duplicate).

**Result**

* `::AbstractDataFrame` : the updated result


**Examples**

```julia
df = DataFrame(i = 1:10, x = rand(10), y = rand(["a", "b", "c"], 10))
names!(df, [:a, :b, :c])
names!(df, [:a, :b, :a])  # throws ArgumentError
names!(df, [:a, :b, :a], makeunique=true)  # renames second :a to :a_1
```

"""
# TODO: remove allow_duplicates after deprecation period
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
end
function rename!(f::Function, df::AbstractDataFrame)
    rename!(f, index(df))
    return df
end

rename(df::AbstractDataFrame, args...) = rename!(copy(df), args...)
rename(f::Function, df::AbstractDataFrame) = rename!(f, copy(df))

"""
Rename columns

```julia
rename!(df::AbstractDataFrame, (from => to)::Pair{Symbol, Symbol}...)
rename!(df::AbstractDataFrame, d::Associative{Symbol,Symbol})
rename!(df::AbstractDataFrame, d::AbstractArray{Pair{Symbol,Symbol}})
rename!(f::Function, df::AbstractDataFrame)
rename(df::AbstractDataFrame, (from => to)::Pair{Symbol, Symbol}...)
rename(df::AbstractDataFrame, d::Associative{Symbol,Symbol})
rename(df::AbstractDataFrame, d::AbstractArray{Pair{Symbol,Symbol}})
rename(f::Function, df::AbstractDataFrame)
```

**Arguments**

* `df` : the AbstractDataFrame
* `d` : an Associative type or an AbstractArray of pairs that maps
  the original names to new names
* `f` : a function which for each column takes the old name (a Symbol)
  and returns the new name (a Symbol)

**Result**

* `::AbstractDataFrame` : the updated result

New names are processed sequentially. A new name must not already exist in the `DataFrame`
at the moment an attempt to rename a column is performed.

**Examples**

```julia
df = DataFrame(i = 1:10, x = rand(10), y = rand(["a", "b", "c"], 10))
rename(df, :i => :A, :x => :X)
rename(df, [:i => :A, :x => :X])
rename(df, Dict(:i => :A, :x => :X))
rename(x -> Symbol(uppercase(string(x))), df)
rename!(df, Dict(:i =>: A, :x => :X))
```

"""
(rename!, rename)

"""
Return element types of columns

```julia
eltypes(df::AbstractDataFrame)
```

**Arguments**

* `df` : the AbstractDataFrame

**Result**

* `::Vector{Type}` : the element type of each column

**Examples**

```julia
df = DataFrame(i = 1:10, x = rand(10), y = rand(["a", "b", "c"], 10))
eltypes(df)
```

"""
eltypes(df::AbstractDataFrame) = map!(eltype, Vector{Type}(size(df,2)), columns(df))

Base.size(df::AbstractDataFrame) = (nrow(df), ncol(df))
function Base.size(df::AbstractDataFrame, i::Integer)
    if i == 1
        nrow(df)
    elseif i == 2
        ncol(df)
    else
        throw(ArgumentError("DataFrames only have two dimensions"))
    end
end

Base.length(df::AbstractDataFrame) = ncol(df)
Base.endof(df::AbstractDataFrame) = ncol(df)

Base.ndims(::AbstractDataFrame) = 2

##############################################################################
##
## Similar
##
##############################################################################

"""
    similar(df::DataFrame[, rows::Integer])

Create a new `DataFrame` with the same column names and column element types
as `df`. An optional second argument can be provided to request a number of rows
that is different than the number of rows present in `df`.
"""
function Base.similar(df::AbstractDataFrame, rows::Integer = size(df, 1))
    rows < 0 && throw(ArgumentError("the number of rows must be positive"))
    DataFrame(Any[similar(x, rows) for x in columns(df)], copy(index(df)))
end

##############################################################################
##
## Equality
##
##############################################################################

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

##############################################################################
##
## Associative methods
##
##############################################################################

Base.haskey(df::AbstractDataFrame, key::Any) = haskey(index(df), key)
Base.get(df::AbstractDataFrame, key::Any, default::Any) = haskey(df, key) ? df[key] : default
Base.isempty(df::AbstractDataFrame) = size(df, 1) == 0 || size(df, 2) == 0

##############################################################################
##
## Description
##
##############################################################################

head(df::AbstractDataFrame, r::Int) = df[1:min(r,nrow(df)), :]
head(df::AbstractDataFrame) = head(df, 6)
tail(df::AbstractDataFrame, r::Int) = df[max(1,nrow(df)-r+1):nrow(df), :]
tail(df::AbstractDataFrame) = tail(df, 6)

"""
Show the first or last part of an AbstractDataFrame

```julia
head(df::AbstractDataFrame, r::Int = 6)
tail(df::AbstractDataFrame, r::Int = 6)
```

**Arguments**

* `df` : the AbstractDataFrame
* `r` : the number of rows to show

**Result**

* `::AbstractDataFrame` : the first or last part of `df`

**Examples**

```julia
df = DataFrame(i = 1:10, x = rand(10), y = rand(["a", "b", "c"], 10))
head(df)
tail(df)
```

"""
(head, tail)

# get the structure of a df
"""
Show the structure of an AbstractDataFrame, in a tree-like format

```julia
dump(df::AbstractDataFrame, n::Int = 5)
dump(io::IO, df::AbstractDataFrame, n::Int = 5)
```

**Arguments**

* `df` : the AbstractDataFrame
* `n` : the number of levels to show
* `io` : optional output descriptor

**Result**

* nothing

**Examples**

```julia
df = DataFrame(i = 1:10, x = rand(10), y = rand(["a", "b", "c"], 10))
dump(df)
```

"""
function Base.dump(io::IO, df::AbstractDataFrame, n::Int, indent)
    println(io, typeof(df), "  $(nrow(df)) observations of $(ncol(df)) variables")
    if n > 0
        for (name, col) in eachcol(df)
            print(io, indent, "  ", name, ": ")
            dump(io, col, n - 1, string(indent, "  "))
        end
    end
end

# summarize the columns of a df
# TODO: clever layout in rows
"""
Summarize the columns of an AbstractDataFrame

```julia
describe(df::AbstractDataFrame)
describe(io, df::AbstractDataFrame)
```

**Arguments**

* `df` : the AbstractDataFrame
* `io` : optional output descriptor

**Result**

* nothing

**Details**

If the column's base type derives from Number, compute the minimum, first
quantile, median, mean, third quantile, and maximum. Missings are filtered and
reported separately.

For boolean columns, report trues, falses, and missings.

For other types, show column characteristics and number of missings.

**Examples**

```julia
df = DataFrame(i = 1:10, x = rand(10), y = rand(["a", "b", "c"], 10))
describe(df)
```

"""
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

##############################################################################
##
## Miscellaneous
##
##############################################################################

function _nonmissing!(res, col)
    @inbounds for (i, el) in enumerate(col)
        res[i] &= !ismissing(el)
    end
end

function _nonmissing!(res, col::CategoricalArray{>: Missing})
    for (i, el) in enumerate(col.refs)
        res[i] &= el > 0
    end
end


"""
Indexes of complete cases (rows without missing values)

```julia
completecases(df::AbstractDataFrame)
```

**Arguments**

* `df` : the AbstractDataFrame

**Result**

* `::Vector{Bool}` : indexes of complete cases

See also [`dropmissing`](@ref) and [`dropmissing!`](@ref).

**Examples**

```julia
df = DataFrame(i = 1:10,
               x = Vector{Union{Missing, Float64}}(rand(10)),
               y = Vector{Union{Missing, String}}(rand(["a", "b", "c"], 10)))
df[[1,4,5], :x] = missing
df[[9,10], :y] = missing
completecases(df)
```

"""
function completecases(df::AbstractDataFrame)
    res = trues(size(df, 1))
    for i in 1:size(df, 2)
        _nonmissing!(res, df[i])
    end
    res
end

"""
Remove rows with missing values.

```julia
dropmissing(df::AbstractDataFrame)
```

**Arguments**

* `df` : the AbstractDataFrame

**Result**

* `::AbstractDataFrame` : the updated copy

See also [`completecases`](@ref) and [`dropmissing!`](@ref).

**Examples**

```julia
df = DataFrame(i = 1:10,
               x = Vector{Union{Missing, Float64}}(rand(10)),
               y = Vector{Union{Missing, String}}(rand(["a", "b", "c"], 10)))
df[[1,4,5], :x] = missing
df[[9,10], :y] = missing
dropmissing(df)
```

"""
dropmissing(df::AbstractDataFrame) = deleterows!(copy(df), find(!, completecases(df)))

"""
Remove rows with missing values in-place.

```julia
dropmissing!(df::AbstractDataFrame)
```

**Arguments**

* `df` : the AbstractDataFrame

**Result**

* `::AbstractDataFrame` : the updated version

See also [`dropmissing`](@ref) and [`completecases`](@ref).

**Examples**

```julia
df = DataFrame(i = 1:10,
               x = Vector{Union{Missing, Float64}}(rand(10)),
               y = Vector{Union{Missing, String}}(rand(["a", "b", "c"], 10)))
df[[1,4,5], :x] = missing
df[[9,10], :y] = missing
dropmissing!(df)
```

"""
dropmissing!(df::AbstractDataFrame) = deleterows!(df, find(!, completecases(df)))

"""
    filter(function, df::AbstractDataFrame)

Return a copy of data frame `df` containing only rows for which `function`
returns `true`. The function is passed a `DataFrameRow` as its only argument.

# Examples
```
julia> df = DataFrame(x = [3, 1, 2, 1], y = ["b", "c", "a", "b"])
4×2 DataFrames.DataFrame
│ Row │ x │ y │
├─────┼───┼───┤
│ 1   │ 3 │ b │
│ 2   │ 1 │ c │
│ 3   │ 2 │ a │
│ 4   │ 1 │ b │

julia> filter(row -> row[:x] > 1, df)
2×2 DataFrames.DataFrame
│ Row │ x │ y │
├─────┼───┼───┤
│ 1   │ 3 │ b │
│ 2   │ 2 │ a │
```
"""
Base.filter(f, df::AbstractDataFrame) = df[collect(f(r)::Bool for r in eachrow(df)), :]

"""
    filter!(function, df::AbstractDataFrame)

Remove rows from data frame `df` for which `function` returns `false`.
The function is passed a `DataFrameRow` as its only argument.

# Examples
```
julia> df = DataFrame(x = [3, 1, 2, 1], y = ["b", "c", "a", "b"])
4×2 DataFrames.DataFrame
│ Row │ x │ y │
├─────┼───┼───┤
│ 1   │ 3 │ b │
│ 2   │ 1 │ c │
│ 3   │ 2 │ a │
│ 4   │ 1 │ b │

julia> filter!(row -> row[:x] > 1, df);

julia> df
2×2 DataFrames.DataFrame
│ Row │ x │ y │
├─────┼───┼───┤
│ 1   │ 3 │ b │
│ 2   │ 2 │ a │
```
"""
Base.filter!(f, df::AbstractDataFrame) =
    deleterows!(df, find(!f, eachrow(df)))

function Base.convert(::Type{Array}, df::AbstractDataFrame)
    convert(Matrix, df)
end
function Base.convert(::Type{Matrix}, df::AbstractDataFrame)
    T = reduce(promote_type, eltypes(df))
    convert(Matrix{T}, df)
end
function Base.convert(::Type{Array{T}}, df::AbstractDataFrame) where T
    convert(Matrix{T}, df)
end
function Base.convert(::Type{Matrix{T}}, df::AbstractDataFrame) where T
    n, p = size(df)
    res = Matrix{T}(n, p)
    idx = 1
    for (name, col) in zip(names(df), columns(df))
        !(T >: Missing) && any(ismissing, col) && error("cannot convert a DataFrame containing missing values to array (found for column $name)")
        copy!(res, idx, convert(Vector{T}, col))
        idx += n
    end
    return res
end

"""
Indexes of duplicate rows (a row that is a duplicate of a prior row)

```julia
nonunique(df::AbstractDataFrame)
nonunique(df::AbstractDataFrame, cols)
```

**Arguments**

* `df` : the AbstractDataFrame
* `cols` : a column indicator (Symbol, Int, Vector{Symbol}, etc.) specifying the column(s) to compare

**Result**

* `::Vector{Bool}` : indicates whether the row is a duplicate of some
  prior row

See also [`unique`](@ref) and [`unique!`](@ref).

**Examples**

```julia
df = DataFrame(i = 1:10, x = rand(10), y = rand(["a", "b", "c"], 10))
df = vcat(df, df)
nonunique(df)
nonunique(df, 1)
```

"""
function nonunique(df::AbstractDataFrame)
    gslots = row_group_slots(df)[3]
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

# Unique rows of an AbstractDataFrame.
Base.unique(df::AbstractDataFrame) = df[(!).(nonunique(df)), :]
Base.unique(df::AbstractDataFrame, cols::Any) = df[(!).(nonunique(df, cols)), :]

"""
Delete duplicate rows

```julia
unique(df::AbstractDataFrame)
unique(df::AbstractDataFrame, cols)
unique!(df::AbstractDataFrame)
unique!(df::AbstractDataFrame, cols)
```

**Arguments**

* `df` : the AbstractDataFrame
* `cols` :  column indicator (Symbol, Int, Vector{Symbol}, etc.)
specifying the column(s) to compare.

**Result**

* `::AbstractDataFrame` : the updated version of `df` with unique rows.
When `cols` is specified, the return DataFrame contains complete rows,
retaining in each case the first instance for which `df[cols]` is unique.

See also [`nonunique`](@ref).

**Examples**

```julia
df = DataFrame(i = 1:10, x = rand(10), y = rand(["a", "b", "c"], 10))
df = vcat(df, df)
unique(df)   # doesn't modify df
unique(df, 1)
unique!(df)  # modifies df
```

"""
(unique, unique!)

# Count the number of missing values in every column of an AbstractDataFrame.
function colmissing(df::AbstractDataFrame) # -> Vector{Int}
    nrows, ncols = size(df)
    missing = zeros(Int, ncols)
    for j in 1:ncols
        missing[j] = countmissing(df[j])
    end
    return missing
end

function without(df::AbstractDataFrame, icols::Vector{Int})
    newcols = setdiff(1:ncol(df), icols)
    df[newcols]
end
without(df::AbstractDataFrame, i::Int) = without(df, [i])
without(df::AbstractDataFrame, c::Any) = without(df, index(df)[c])

##############################################################################
##
## Hcat / vcat
##
##############################################################################

# hcat's first argument must be an AbstractDataFrame
# or AbstractVector if the second argument is AbstractDataFrame
# Trailing arguments (currently) may also be vectors.

# hcat! is defined in DataFrames/DataFrames.jl
# Its first argument (currently) must be a DataFrame.

# catch-all to cover cases where indexing returns a DataFrame and copy doesn't

Base.hcat(df::AbstractDataFrame, x; makeunique::Bool=false) =
    hcat!(df[:, :], x, makeunique=makeunique)
Base.hcat(x, df::AbstractDataFrame; makeunique::Bool=false) =
    hcat!(x, df[:, :], makeunique=makeunique)
Base.hcat(df1::AbstractDataFrame, df2::AbstractDataFrame; makeunique::Bool=false) =
    hcat!(df1[:, :], df2, makeunique=makeunique)
Base.hcat(df::AbstractDataFrame, x, y...; makeunique::Bool=false) =
    hcat!(hcat(df, x, makeunique=makeunique), y..., makeunique=makeunique)
Base.hcat(df1::AbstractDataFrame, df2::AbstractDataFrame, dfn::AbstractDataFrame...;
          makeunique::Bool=false) =
    hcat!(hcat(df1, df2, makeunique=makeunique), dfn..., makeunique=makeunique)

@generated function promote_col_type(cols::AbstractVector...)
    T = mapreduce(x -> Missings.T(eltype(x)), promote_type, cols)
    if CategoricalArrays.iscatvalue(T)
        T = CategoricalArrays.leveltype(T)
    end
    if any(col -> eltype(col) >: Missing, cols)
        if any(col -> col <: AbstractCategoricalArray, cols)
            return :(CategoricalVector{Union{$T, Missing}})
        else
            return :(Vector{Union{$T, Missing}})
        end
    else
        if any(col -> col <: AbstractCategoricalArray, cols)
            return :(CategoricalVector{$T})
        else
            return :(Vector{$T})
        end
    end
end

"""
    vcat(dfs::AbstractDataFrame...)

Vertically concatenate `AbstractDataFrames` that have the same column names in
the same order.

# Example
```jldoctest
julia> df1 = DataFrame(A=1:3, B=1:3);
julia> df2 = DataFrame(A=4:6, B=4:6);
julia> vcat(df1, df2)
6×2 DataFrames.DataFrame
│ Row │ A │ B │
├─────┼───┼───┤
│ 1   │ 1 │ 1 │
│ 2   │ 2 │ 2 │
│ 3   │ 3 │ 3 │
│ 4   │ 4 │ 4 │
│ 5   │ 5 │ 5 │
│ 6   │ 6 │ 6 │
```
"""
Base.vcat(df::AbstractDataFrame) = df
Base.vcat(dfs::AbstractDataFrame...) = _vcat(collect(dfs))
function _vcat(dfs::AbstractVector{<:AbstractDataFrame})
    isempty(dfs) && return DataFrame()
    allheaders = map(names, dfs)
    if all(h -> length(h) == 0, allheaders)
        return DataFrame()
    end
    uniqueheaders = unique(allheaders)
    if length(uniqueheaders) > 1
        unionunique = union(uniqueheaders...)
        coldiff = setdiff(unionunique, intersect(uniqueheaders...))
        if !isempty(coldiff)
            # if any DataFrames are a full superset of names, skip them
            filter!(u -> Set(u) != Set(unionunique), uniqueheaders)
            estrings = Vector{String}(length(uniqueheaders))
            for (i, u) in enumerate(uniqueheaders)
                matching = find(h -> u == h, allheaders)
                headerdiff = setdiff(coldiff, u)
                cols = join(headerdiff, ", ", " and ")
                args = join(matching, ", ", " and ")
                estrings[i] = "column(s) $cols are missing from argument(s) $args"
            end
            throw(ArgumentError(join(estrings, ", ", ", and ")))
        else
            estrings = Vector{String}(length(uniqueheaders))
            for (i, u) in enumerate(uniqueheaders)
                indices = find(a -> a == u, allheaders)
                estrings[i] = "column order of argument(s) $(join(indices, ", ", " and "))"
            end
            throw(ArgumentError(join(estrings, " != ")))
        end
    else
        header = uniqueheaders[1]
        cols = Vector{Any}(length(header))
        for i in 1:length(cols)
            data = [df[i] for df in dfs]
            lens = map(length, data)
            cols[i] = promote_col_type(data...)(sum(lens))
            offset = 1
            for j in 1:length(data)
                copy!(cols[i], offset, data[j])
                offset += lens[j]
            end
        end
        return DataFrame(cols, header)
    end
end

##############################################################################
##
## Hashing
##
##############################################################################

const hashdf_seed = UInt == UInt32 ? 0xfd8bb02e : 0x6215bada8c8c46de

function Base.hash(df::AbstractDataFrame, h::UInt)
    h += hashdf_seed
    h += hash(size(df))
    for i in 1:size(df, 2)
        h = hash(df[i], h)
    end
    return h
end


## Documentation for methods defined elsewhere


#
# expanded from: include("dataframe/dataframe.jl")
#

"""
An AbstractDataFrame that stores a set of named columns

The columns are normally AbstractVectors stored in memory,
particularly a Vector or CategoricalVector.

**Constructors**

```julia
DataFrame(columns::Vector, names::Vector{Symbol}; makeunique::Bool=false)
DataFrame(columns::Matrix, names::Vector{Symbol}; makeunique::Bool=false)
DataFrame(kwargs...)
DataFrame(pairs::Pair{Symbol}...; makeunique::Bool=false)
DataFrame() # an empty DataFrame
DataFrame(t::Type, nrows::Integer, ncols::Integer) # an empty DataFrame of arbitrary size
DataFrame(column_eltypes::Vector, names::Vector, nrows::Integer; makeunique::Bool=false)
DataFrame(column_eltypes::Vector, cnames::Vector, categorical::Vector, nrows::Integer;
          makeunique::Bool=false)
DataFrame(ds::Vector{Associative})
```

**Arguments**

* `columns` : a Vector with each column as contents or a Matrix
* `names` : the column names
* `makeunique` : if `false` (the default), an error will be raised
  if duplicates in `names` are found; if `true`, duplicate names will be suffixed
  with `_i` (`i` starting at 1 for the first duplicate).
* `kwargs` : the key gives the column names, and the value is the
  column contents
* `t` : elemental type of all columns
* `nrows`, `ncols` : number of rows and columns
* `column_eltypes` : elemental type of each column
* `categorical` : `Vector{Bool}` indicating which columns should be converted to
                  `CategoricalVector`
* `ds` : a vector of Associatives

Each column in `columns` should be the same length.

**Notes**

A `DataFrame` is a lightweight object. As long as columns are not
manipulated, creation of a DataFrame from existing AbstractVectors is
inexpensive. For example, indexing on columns is inexpensive, but
indexing by rows is expensive because copies are made of each column.

Because column types can vary, a DataFrame is not type stable. For
performance-critical code, do not index into a DataFrame inside of
loops.

**Examples**

```julia
df = DataFrame()
v = ["x","y","z"][rand(1:3, 10)]
df1 = DataFrame(Any[collect(1:10), v, rand(10)], [:A, :B, :C])
df2 = DataFrame(A = 1:10, B = v, C = rand(10))
dump(df1)
dump(df2)
describe(df2)
head(df1)
df1[:A] + df2[:C]
df1[1:4, 1:2]
df1[[:A,:C]]
df1[1:2, [:A,:C]]
df1[:, [:A,:C]]
df1[:, [1,3]]
df1[1:4, :]
df1[1:4, :C]
df1[1:4, :C] = 40. * df1[1:4, :C]
[df1; df2]  # vcat
[df1  df2]  # hcat
size(df1)
```

"""
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

# Initialize an empty DataFrame with specific eltypes and names
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

# Initialize an empty DataFrame with specific eltypes and names
# and whether a CategoricalArray should be created
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

# Initialize empty DataFrame objects of arbitrary size
function DataFrame(t::Type, nrows::Integer, ncols::Integer)
    return DataFrame(fill(t, ncols), nrows)
end

# Initialize an empty DataFrame with specific eltypes
function DataFrame(column_eltypes::AbstractVector{T}, nrows::Integer) where T<:Type
    return DataFrame(column_eltypes, gennames(length(column_eltypes)), nrows)
end

##############################################################################
##
## AbstractDataFrame interface
##
##############################################################################

index(df::DataFrame) = df.colindex
columns(df::DataFrame) = df.columns

# TODO: Remove these
nrow(df::DataFrame) = ncol(df) > 0 ? length(df.columns[1])::Int : 0
ncol(df::DataFrame) = length(index(df))

##############################################################################
##
## getindex() definitions
##
##############################################################################

# Cases:
#
# df[SingleColumnIndex] => AbstractDataVector
# df[MultiColumnIndex] => DataFrame
# df[SingleRowIndex, SingleColumnIndex] => Scalar
# df[SingleRowIndex, MultiColumnIndex] => DataFrame
# df[MultiRowIndex, SingleColumnIndex] => AbstractVector
# df[MultiRowIndex, MultiColumnIndex] => DataFrame
#
# General Strategy:
#
# Let getindex(index(df), col_inds) from Index() handle the resolution
#  of column indices
# Let getindex(df.columns[j], row_inds) from AbstractVector() handle
#  the resolution of row indices

# TODO: change Real to Integer in this union after deprecation period
const ColumnIndex = Union{Real, Symbol}

# df[SingleColumnIndex] => AbstractDataVector
function Base.getindex(df::DataFrame, col_ind::ColumnIndex)
    selected_column = index(df)[col_ind]
    return df.columns[selected_column]
end

# df[MultiColumnIndex] => DataFrame
function Base.getindex(df::DataFrame, col_inds::AbstractVector)
    selected_columns = index(df)[col_inds]
    new_columns = df.columns[selected_columns]
    return DataFrame(new_columns, Index(_names(df)[selected_columns]))
end

# df[:] => DataFrame
Base.getindex(df::DataFrame, col_inds::Colon) = copy(df)

# df[SingleRowIndex, SingleColumnIndex] => Scalar
function Base.getindex(df::DataFrame, row_ind::Real, col_ind::ColumnIndex)
    selected_column = index(df)[col_ind]
    return df.columns[selected_column][row_ind]
end

# df[SingleRowIndex, MultiColumnIndex] => DataFrame
function Base.getindex(df::DataFrame, row_ind::Real, col_inds::AbstractVector)
    selected_columns = index(df)[col_inds]
    new_columns = Any[dv[[row_ind]] for dv in df.columns[selected_columns]]
    return DataFrame(new_columns, Index(_names(df)[selected_columns]))
end

# df[MultiRowIndex, SingleColumnIndex] => AbstractVector
function Base.getindex(df::DataFrame, row_inds::AbstractVector, col_ind::ColumnIndex)
    selected_column = index(df)[col_ind]
    return df.columns[selected_column][row_inds]
end

# df[MultiRowIndex, MultiColumnIndex] => DataFrame
function Base.getindex(df::DataFrame, row_inds::AbstractVector, col_inds::AbstractVector)
    selected_columns = index(df)[col_inds]
    new_columns = Any[dv[row_inds] for dv in df.columns[selected_columns]]
    return DataFrame(new_columns, Index(_names(df)[selected_columns]))
end

# df[:, SingleColumnIndex] => AbstractVector
# df[:, MultiColumnIndex] => DataFrame
Base.getindex(df::DataFrame, row_ind::Colon, col_inds) = df[col_inds]

# df[SingleRowIndex, :] => DataFrame
Base.getindex(df::DataFrame, row_ind::Real, col_inds::Colon) = df[[row_ind], col_inds]

# df[MultiRowIndex, :] => DataFrame
function Base.getindex(df::DataFrame, row_inds::AbstractVector, col_inds::Colon)
    new_columns = Any[dv[row_inds] for dv in df.columns]
    return DataFrame(new_columns, copy(index(df)))
end

# df[:, :] => DataFrame
Base.getindex(df::DataFrame, ::Colon, ::Colon) = copy(df)

##############################################################################
##
## setindex!()
##
##############################################################################

isnextcol(df::DataFrame, col_ind::Symbol) = true
function isnextcol(df::DataFrame, col_ind::Real)
    return ncol(df) + 1 == Int(col_ind)
end

function nextcolname(df::DataFrame)
    return Symbol(string("x", ncol(df) + 1))
end

# Will automatically add a new column if needed
function insert_single_column!(df::DataFrame,
                               dv::AbstractVector,
                               col_ind::ColumnIndex)

    if ncol(df) != 0 && nrow(df) != length(dv)
        error("New columns must have the same length as old columns")
    end
    if haskey(index(df), col_ind)
        j = index(df)[col_ind]
        df.columns[j] = dv
    else
        if typeof(col_ind) <: Symbol
            push!(index(df), col_ind)
            push!(df.columns, dv)
        else
            if isnextcol(df, col_ind)
                push!(index(df), nextcolname(df))
                push!(df.columns, dv)
            else
                error("Cannot assign to non-existent column: $col_ind")
            end
        end
    end
    return dv
end

function insert_single_entry!(df::DataFrame, v::Any, row_ind::Real, col_ind::ColumnIndex)
    if haskey(index(df), col_ind)
        df.columns[index(df)[col_ind]][row_ind] = v
        return v
    else
        error("Cannot assign to non-existent column: $col_ind")
    end
end

function insert_multiple_entries!(df::DataFrame,
                                  v::Any,
                                  row_inds::AbstractVector{<:Real},
                                  col_ind::ColumnIndex)
    if haskey(index(df), col_ind)
        df.columns[index(df)[col_ind]][row_inds] = v
        return v
    else
        error("Cannot assign to non-existent column: $col_ind")
    end
end

function upgrade_scalar(df::DataFrame, v::AbstractArray)
    msg = "setindex!(::DataFrame, ...) only broadcasts scalars, not arrays"
    throw(ArgumentError(msg))
end
function upgrade_scalar(df::DataFrame, v::Any)
    n = (ncol(df) == 0) ? 1 : nrow(df)
    fill(v, n)
end

# df[SingleColumnIndex] = AbstractVector
function Base.setindex!(df::DataFrame, v::AbstractVector, col_ind::ColumnIndex)
    insert_single_column!(df, v, col_ind)
end

# df[SingleColumnIndex] = Single Item (EXPANDS TO NROW(df) if NCOL(df) > 0)
function Base.setindex!(df::DataFrame, v, col_ind::ColumnIndex)
    if haskey(index(df), col_ind)
        fill!(df[col_ind], v)
    else
        insert_single_column!(df, upgrade_scalar(df, v), col_ind)
    end
    return df
end

# df[MultiColumnIndex] = DataFrame
function Base.setindex!(df::DataFrame, new_df::DataFrame, col_inds::AbstractVector{Bool})
    setindex!(df, new_df, find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        col_inds::AbstractVector{<:ColumnIndex})
    for j in 1:length(col_inds)
        insert_single_column!(df, new_df[j], col_inds[j])
    end
    return df
end

# df[MultiColumnIndex] = AbstractVector (REPEATED FOR EACH COLUMN)
function Base.setindex!(df::DataFrame, v::AbstractVector, col_inds::AbstractVector{Bool})
    setindex!(df, v, find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        df[col_ind] = v
    end
    return df
end

# df[MultiColumnIndex] = Single Item (REPEATED FOR EACH COLUMN; EXPANDS TO NROW(df) if NCOL(df) > 0)
function Base.setindex!(df::DataFrame,
                        val::Any,
                        col_inds::AbstractVector{Bool})
    setindex!(df, val, find(col_inds))
end
function Base.setindex!(df::DataFrame, val::Any, col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        df[col_ind] = val
    end
    return df
end

# df[:] = AbstractVector or Single Item
Base.setindex!(df::DataFrame, v, ::Colon) = (df[1:size(df, 2)] = v; df)

# df[SingleRowIndex, SingleColumnIndex] = Single Item
function Base.setindex!(df::DataFrame, v::Any, row_ind::Real, col_ind::ColumnIndex)
    insert_single_entry!(df, v, row_ind, col_ind)
end

# df[SingleRowIndex, MultiColumnIndex] = Single Item
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_ind::Real,
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, row_ind, find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_ind::Real,
                        col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        insert_single_entry!(df, v, row_ind, col_ind)
    end
    return df
end

# df[SingleRowIndex, MultiColumnIndex] = 1-Row DataFrame
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_ind::Real,
                        col_inds::AbstractVector{Bool})
    setindex!(df, new_df, row_ind, find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_ind::Real,
                        col_inds::AbstractVector{<:ColumnIndex})
    for j in 1:length(col_inds)
        insert_single_entry!(df, new_df[j][1], row_ind, col_inds[j])
    end
    return df
end

# df[MultiRowIndex, SingleColumnIndex] = AbstractVector
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{Bool},
                        col_ind::ColumnIndex)
    setindex!(df, v, find(row_inds), col_ind)
end
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{<:Real},
                        col_ind::ColumnIndex)
    insert_multiple_entries!(df, v, row_inds, col_ind)
    return df
end

# df[MultiRowIndex, SingleColumnIndex] = Single Item
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_inds::AbstractVector{Bool},
                        col_ind::ColumnIndex)
    setindex!(df, v, find(row_inds), col_ind)
end
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_inds::AbstractVector{<:Real},
                        col_ind::ColumnIndex)
    insert_multiple_entries!(df, v, row_inds, col_ind)
    return df
end

# df[MultiRowIndex, MultiColumnIndex] = DataFrame
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{Bool})
    setindex!(df, new_df, find(row_inds), find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{<:ColumnIndex})
    setindex!(df, new_df, find(row_inds), col_inds)
end
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_inds::AbstractVector{<:Real},
                        col_inds::AbstractVector{Bool})
    setindex!(df, new_df, row_inds, find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_inds::AbstractVector{<:Real},
                        col_inds::AbstractVector{<:ColumnIndex})
    for j in 1:length(col_inds)
        insert_multiple_entries!(df, new_df[:, j], row_inds, col_inds[j])
    end
    return df
end

# df[MultiRowIndex, MultiColumnIndex] = AbstractVector
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, find(row_inds), find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{<:ColumnIndex})
    setindex!(df, v, find(row_inds), col_inds)
end
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{<:Real},
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, row_inds, find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        v::AbstractVector,
                        row_inds::AbstractVector{<:Real},
                        col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        insert_multiple_entries!(df, v, row_inds, col_ind)
    end
    return df
end

# df[MultiRowIndex, MultiColumnIndex] = Single Item
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, find(row_inds), find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_inds::AbstractVector{Bool},
                        col_inds::AbstractVector{<:ColumnIndex})
    setindex!(df, v, find(row_inds), col_inds)
end
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_inds::AbstractVector{<:Real},
                        col_inds::AbstractVector{Bool})
    setindex!(df, v, row_inds, find(col_inds))
end
function Base.setindex!(df::DataFrame,
                        v::Any,
                        row_inds::AbstractVector{<:Real},
                        col_inds::AbstractVector{<:ColumnIndex})
    for col_ind in col_inds
        insert_multiple_entries!(df, v, row_inds, col_ind)
    end
    return df
end

# df[:] = DataFrame, df[:, :] = DataFrame
function Base.setindex!(df::DataFrame,
                        new_df::DataFrame,
                        row_inds::Colon,
                        col_inds::Colon=Colon())
    df.columns = copy(new_df.columns)
    df.colindex = copy(new_df.colindex)
    df
end

# df[:, :] = ...
Base.setindex!(df::DataFrame, v, ::Colon, ::Colon) =
    (df[1:size(df, 1), 1:size(df, 2)] = v; df)

# df[Any, :] = ...
Base.setindex!(df::DataFrame, v, row_inds, ::Colon) =
    (df[row_inds, 1:size(df, 2)] = v; df)

# df[:, Any] = ...
Base.setindex!(df::DataFrame, v, ::Colon, col_inds) =
    (df[col_inds] = v; df)

# Special deletion assignment
Base.setindex!(df::DataFrame, x::Void, col_ind::Int) = delete!(df, col_ind)

##############################################################################
##
## Mutating Associative methods
##
##############################################################################

Base.empty!(df::DataFrame) = (empty!(df.columns); empty!(index(df)); df)

"""
Insert a column into a data frame in place.


```julia
insert!(df::DataFrame, col_ind::Int, item::AbstractVector, name::Symbol;
        makeunique::Bool=false)
```

### Arguments

* `df` : the DataFrame to which we want to add a column

* `col_ind` : a position at which we want to insert a column

* `item` : a column to be inserted into `df`

* `name` : column name

* `makeunique` : Defines what to do if `name` already exists in `df`;
  if it is `false` an error will be thrown; if it is `true` a new unique name will
  be generated by adding a suffix

### Result

* `::DataFrame` : a `DataFrame` with added column.

### Examples

```jldoctest
julia> d = DataFrame(a=1:3)
3×1 DataFrames.DataFrame
│ Row │ a │
├─────┼───┤
│ 1   │ 1 │
│ 2   │ 2 │
│ 3   │ 3 │

julia> insert!(d, 1, 'a':'c', :b)
3×2 DataFrames.DataFrame
│ Row │ b   │ a │
├─────┼─────┼───┤
│ 1   │ 'a' │ 1 │
│ 2   │ 'b' │ 2 │
│ 3   │ 'c' │ 3 │
```

"""
function Base.insert!(df::DataFrame, col_ind::Int, item::AbstractVector, name::Symbol;
                      makeunique::Bool=false)
    0 < col_ind <= ncol(df) + 1 || throw(BoundsError())
    size(df, 1) == length(item) || size(df, 1) == 0 || error("number of rows does not match")

    if haskey(df, name)
        if makeunique
            k = 1
            while true
                # we only make sure that new column name is unique
                # if df originally had duplicates in names we do not fix it
                nn = Symbol("$(name)_$k")
                if !haskey(df, nn)
                    name = nn
                    break
                end
                k += 1
            end
        else
            # TODO: remove depwarn and uncomment ArgumentError below
            Base.depwarn("Inserting duplicate column name is deprecated, use makeunique=true.", :insert!)
            # msg = """Duplicate variable name $(name).
            #      Pass makeunique=true to make it unique using a suffix automatically."""
            # throw(ArgumentError(msg))
        end
    end
    insert!(index(df), col_ind, name)
    insert!(df.columns, col_ind, item)
    df
end

function Base.insert!(df::DataFrame, col_ind::Int, item, name::Symbol; makeunique::Bool=false)
    insert!(df, col_ind, upgrade_scalar(df, item), name, makeunique=makeunique)
end

"""
Merge data frames.


```julia
merge!(df::DataFrame, others::AbstractDataFrame...)
```

For every column `c` with name `n` in `others` sequentially perform `df[n] = c`.
In particular, if there are duplicate column names present in `df` and `others`
the last encountered column will be retained.
This behavior is identical with how `merge!` works for any `Associative` type.
Use `join` if you want to join two `DataFrame`s.

**Arguments**

* `df` : the DataFrame to merge into
* `others` : `AbstractDataFrame`s to be merged into `df`

**Result**

* `::DataFrame` : the updated result. Columns with duplicate names are overwritten.

**Examples**

```julia
df = DataFrame(id = 1:10, x = rand(10), y = rand(["a", "b", "c"], 10))
df2 = DataFrame(id = 11:20, z = rand(10))
merge!(df, df2)  # column z is added, column id is overwritten
```
"""
function Base.merge!(df::DataFrame, others::AbstractDataFrame...)
    for other in others
        for n in _names(other)
            df[n] = other[n]
        end
    end
    return df
end

##############################################################################
##
## Copying
##
##############################################################################

# A copy of a DataFrame points to the original column vectors but
#   gets its own Index.
Base.copy(df::DataFrame) = DataFrame(copy(columns(df)), copy(index(df)))

# Deepcopy is recursive -- if a column is a vector of DataFrames, each of
#   those DataFrames is deepcopied.
function Base.deepcopy(df::DataFrame)
    DataFrame(deepcopy(columns(df)), deepcopy(index(df)))
end

##############################################################################
##
## Deletion / Subsetting
##
##############################################################################

# delete!() deletes columns; deleterows!() deletes rows
# delete!(df, 1)
# delete!(df, :Old)
function Base.delete!(df::DataFrame, inds::Vector{Int})
    for ind in sort(inds, rev = true)
        if 1 <= ind <= ncol(df)
            splice!(df.columns, ind)
            delete!(index(df), ind)
        else
            throw(ArgumentError("Can't delete a non-existent DataFrame column"))
        end
    end
    return df
end
Base.delete!(df::DataFrame, c::Int) = delete!(df, [c])
Base.delete!(df::DataFrame, c::Any) = delete!(df, index(df)[c])

# deleterows!()
function deleterows!(df::DataFrame, ind::Union{Integer, UnitRange{Int}})
    for i in 1:ncol(df)
        df.columns[i] = deleteat!(df.columns[i], ind)
    end
    df
end

function deleterows!(df::DataFrame, ind::AbstractVector{Int})
    ind2 = sort(ind)
    n = size(df, 1)

    idf = 1
    iind = 1
    ikeep = 1
    keep = Vector{Int}(n-length(ind2))
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

##############################################################################
##
## Hcat specialization
##
##############################################################################

# hcat! for 2 arguments, only a vector or a data frame is allowed
function hcat!(df1::DataFrame, df2::AbstractDataFrame; makeunique::Bool=false)
    u = add_names(index(df1), index(df2), makeunique=makeunique)
    for i in 1:length(u)
        df1[u[i]] = df2[i]
    end
    return df1
end

# definition required to avoid hcat! ambiguity
function hcat!(df1::DataFrame, df2::DataFrame; makeunique::Bool=false)
    invoke(hcat!, Tuple{DataFrame, AbstractDataFrame}, df1, df2, makeunique=makeunique)
end

hcat!(df::DataFrame, x::AbstractVector; makeunique::Bool=false) =
    hcat!(df, DataFrame(Any[x]), makeunique=makeunique)
hcat!(x::AbstractVector, df::DataFrame; makeunique::Bool=false) =
    hcat!(DataFrame(Any[x]), df, makeunique=makeunique)
function hcat!(x, df::DataFrame; makeunique::Bool=false)
    throw(ArgumentError("x must be AbstractVector or AbstractDataFrame"))
end
function hcat!(df::DataFrame, x; makeunique::Bool=false)
    throw(ArgumentError("x must be AbstractVector or AbstractDataFrame"))
end

# hcat! for 1-n arguments
hcat!(df::DataFrame; makeunique::Bool=false) = df
hcat!(a::DataFrame, b, c...; makeunique::Bool=false) =
    hcat!(hcat!(a, b, makeunique=makeunique), c..., makeunique=makeunique)

# hcat
Base.hcat(df::DataFrame, x; makeunique::Bool=false) =
    hcat!(copy(df), x, makeunique=makeunique)
Base.hcat(df1::DataFrame, df2::AbstractDataFrame; makeunique::Bool=false) =
    hcat!(copy(df1), df2, makeunique=makeunique)
Base.hcat(df1::DataFrame, df2::AbstractDataFrame, dfn::AbstractDataFrame...;
          makeunique::Bool=false) =
    hcat!(hcat(df1, df2, makeunique=makeunique), dfn..., makeunique=makeunique)

##############################################################################
##
## Missing values support
##
##############################################################################
"""
    allowmissing!(df::DataFrame)

Convert all columns of a `df` from element type `T` to
`Union{T, Missing}` to support missing values.

    allowmissing!(df::DataFrame, col::Union{Integer, Symbol})

Convert a single column of a `df` from element type `T` to
`Union{T, Missing}` to support missing values.

    allowmissing!(df::DataFrame, cols::AbstractVector{<:Union{Integer, Symbol}})

Convert multiple columns of a `df` from element type `T` to
`Union{T, Missing}` to support missing values.
"""
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

##############################################################################
##
## Pooling
##
##############################################################################

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
        if eltype(df[i]) <: AbstractString
            df[i] = CategoricalVector(df[i])
        end
    end
    df
end

function Base.append!(df1::DataFrame, df2::AbstractDataFrame)
   _names(df1) == _names(df2) || error("Column names do not match")
   eltypes(df1) == eltypes(df2) || error("Column eltypes do not match")
   ncols = size(df1, 2)
   # TODO: This needs to be a sort of transaction to be 100% safe
   for j in 1:ncols
       append!(df1[j], df2[j])
   end
   return df1
end

Base.convert(::Type{DataFrame}, A::AbstractMatrix) = DataFrame(A)

function Base.convert(::Type{DataFrame}, d::Associative)
    colnames = keys(d)
    if isa(d, Dict)
        colnames = sort!(collect(keys(d)))
    else
        colnames = keys(d)
    end
    colindex = Index(Symbol[k for k in colnames])
    columns = Any[d[c] for c in colnames]
    DataFrame(columns, colindex)
end


##############################################################################
##
## push! a row onto a DataFrame
##
##############################################################################

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

# array and tuple like collections
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


#
# expanded from: include("subdataframe/subdataframe.jl")
#

##############################################################################
##
## We use SubDataFrame's to maintain a reference to a subset of a DataFrame
## without making copies.
##
##############################################################################

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

"""
A view of row subsets of an AbstractDataFrame

A `SubDataFrame` is meant to be constructed with `view`.  A
SubDataFrame is used frequently in split/apply sorts of operations.

```julia
view(d::AbstractDataFrame, rows)
```

### Arguments

* `d` : an AbstractDataFrame
* `rows` : any indexing type for rows, typically an Int,
  AbstractVector{Int}, AbstractVector{Bool}, or a Range

### Notes

A `SubDataFrame` is an AbstractDataFrame, so expect that most
DataFrame functions should work. Such methods include `describe`,
`dump`, `nrow`, `size`, `by`, `stack`, and `join`. Indexing is just
like a DataFrame; copies are returned.

To subset along columns, use standard column indexing as that creates
a view to the columns by default. To subset along rows and columns,
use column-based indexing with `view`.

### Examples

```julia
df = DataFrame(a = repeat([1, 2, 3, 4], outer=[2]),
               b = repeat([2, 1], outer=[4]),
               c = randn(8))
sdf1 = view(df, 1:6)
sdf2 = view(df, df[:a] .> 1)
sdf3 = view(df[[1,3]], df[:a] .> 1)  # row and column subsetting
sdf4 = groupby(df, :a)[1]  # indexing a GroupedDataFrame returns a SubDataFrame
sdf5 = view(sdf1, 1:3)
sdf1[:,[:a,:b]]
```

"""
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

##############################################################################
##
## AbstractDataFrame interface
##
##############################################################################

index(sdf::SubDataFrame) = index(sdf.parent)

# TODO: Remove these
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

##############################################################################
##
## Miscellaneous
##
##############################################################################

Base.map(f::Function, sdf::SubDataFrame) = f(sdf) # TODO: deprecate

without(sdf::SubDataFrame, c) = view(without(sdf.parent, c), sdf.rows)


#
# expanded from: include("groupeddataframe/grouping.jl")
#

#
#  Split - Apply - Combine operations
#

##############################################################################
##
## GroupedDataFrame...
##
##############################################################################

"""
The result of a `groupby` operation on an AbstractDataFrame; a
view into the AbstractDataFrame grouped by rows.

Not meant to be constructed directly, see `groupby`.
"""
mutable struct GroupedDataFrame
    parent::AbstractDataFrame
    cols::Vector         # columns used for sorting
    idx::Vector{Int}     # indexing vector when sorted by the given columns
    starts::Vector{Int}  # starts of groups
    ends::Vector{Int}    # ends of groups
end

#
# Split
#

"""
A view of an AbstractDataFrame split into row groups

```julia
groupby(d::AbstractDataFrame, cols; sort = false, skipmissing = false)
groupby(cols; sort = false, skipmissing = false)
```

### Arguments

* `d` : an AbstractDataFrame to split (optional, see [Returns](#returns))
* `cols` : data table columns to group by
* `sort`: whether to sort rows according to the values of the grouping columns `cols`
* `skipmissing`: whether to skip rows with `missing` values in one of the grouping columns `cols`

### Returns

* `::GroupedDataFrame` : a grouped view into `d`
* `::Function`: a function `x -> groupby(x, cols)` (if `d` is not specified)

### Details

An iterator over a `GroupedDataFrame` returns a `SubDataFrame` view
for each grouping into `d`. A `GroupedDataFrame` also supports
indexing by groups and `map`.

See the following for additional split-apply-combine operations:

* `by` : split-apply-combine using functions
* `aggregate` : split-apply-combine; applies functions in the form of a cross product
* `combine` : combine (obviously)
* `colwise` : apply a function to each column in an AbstractDataFrame or GroupedDataFrame

### Examples

```julia
df = DataFrame(a = repeat([1, 2, 3, 4], outer=[2]),
               b = repeat([2, 1], outer=[4]),
               c = randn(8))
gd = groupby(df, :a)
gd[1]
last(gd)
vcat([g[:b] for g in gd]...)
for g in gd
    println(g)
end
map(d -> mean(skipmissing(d[:c])), gd)   # returns a GroupApplied object
combine(map(d -> mean(skipmissing(d[:c])), gd))
```

"""
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

##############################################################################
##
## GroupApplied...
##    the result of a split-apply operation
##    TODOs:
##      - better name?
##      - ref
##      - keys, vals
##      - length
##      - start, next, done -- should this return (k,v) or just v?
##      - make it a real associative type? Is there a need to look up key columns?
##
##############################################################################

"""
The result of a `map` operation on a GroupedDataFrame; mainly for use
with `combine`

Not meant to be constructed directly, see `groupby` abnd
`combine`. Minimal support is provided for this type. `map` is
provided for a GroupApplied object.

"""
struct GroupApplied{T<:AbstractDataFrame}
    gd::GroupedDataFrame
    vals::Vector{T}

    function (::Type{GroupApplied})(gd::GroupedDataFrame, vals::Vector)
        length(gd) == length(vals) ||
            throw(DimensionMismatch("GroupApplied requires keys and vals be of equal length (got $(length(gd)) and $(length(vals)))."))
        new{eltype(vals)}(gd, vals)
    end
end


#
# Apply / map
#

# map() sweeps along groups
function Base.map(f::Function, gd::GroupedDataFrame)
    GroupApplied(gd, [wrap(f(df)) for df in gd])
end
function Base.map(f::Function, ga::GroupApplied)
    GroupApplied(ga.gd, [wrap(f(df)) for df in ga.vals])
end

wrap(df::AbstractDataFrame) = df
wrap(A::Matrix) = convert(DataFrame, A)
wrap(s::Any) = DataFrame(x1 = s)

"""
Combine a GroupApplied object (rudimentary)

```julia
combine(ga::GroupApplied)
```

### Arguments

* `ga` : a GroupApplied

### Returns

* `::DataFrame`

### Examples

```julia
df = DataFrame(a = repeat([1, 2, 3, 4], outer=[2]),
               b = repeat([2, 1], outer=[4]),
               c = randn(8))
gd = groupby(df, :a)
combine(map(d -> mean(skipmissing(d[:c])), gd))
```

"""
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


"""
Apply a function to each column in an AbstractDataFrame or
GroupedDataFrame

```julia
colwise(f::Function, d)
colwise(d)
```

### Arguments

* `f` : a function or vector of functions
* `d` : an AbstractDataFrame of GroupedDataFrame

If `d` is not provided, a curried version of groupby is given.

### Returns

* various, depending on the call

### Examples

```julia
df = DataFrame(a = repeat([1, 2, 3, 4], outer=[2]),
               b = repeat([2, 1], outer=[4]),
               c = randn(8))
colwise(sum, df)
colwise([sum, length], df)
colwise((minimum, maximum), df)
colwise(sum, groupby(df, :a))
```

"""
colwise(f, d::AbstractDataFrame) = [f(d[i]) for i in 1:ncol(d)]

# apply several functions to each column in a DataFrame
colwise(fns::Union{AbstractVector, Tuple}, d::AbstractDataFrame) = [f(d[i]) for f in fns, i in 1:ncol(d)]
colwise(f, gd::GroupedDataFrame) = [colwise(f, g) for g in gd]

"""
Split-apply-combine in one step; apply `f` to each grouping in `d`
based on columns `col`

```julia
by(d::AbstractDataFrame, cols, f::Function; sort::Bool = false)
by(f::Function, d::AbstractDataFrame, cols; sort::Bool = false)
```

### Arguments

* `d` : an AbstractDataFrame
* `cols` : a column indicator (Symbol, Int, Vector{Symbol}, etc.)
* `f` : a function to be applied to groups; expects each argument to
  be an AbstractDataFrame
* `sort`: sort row groups (no sorting by default)

`f` can return a value, a vector, or a DataFrame. For a value or
vector, these are merged into a column along with the `cols` keys. For
a DataFrame, `cols` are combined along columns with the resulting
DataFrame. Returning a DataFrame is the clearest because it allows
column labeling.

A method is defined with `f` as the first argument, so do-block
notation can be used.

`by(d, cols, f)` is equivalent to `combine(map(f, groupby(d, cols)))`.

### Returns

* `::DataFrame`

### Examples

```julia
df = DataFrame(a = repeat([1, 2, 3, 4], outer=[2]),
               b = repeat([2, 1], outer=[4]),
               c = randn(8))
by(df, :a, d -> sum(d[:c]))
by(df, :a, d -> 2 * skipmissing(d[:c]))
by(df, :a, d -> DataFrame(c_sum = sum(d[:c]), c_mean = mean(skipmissing(d[:c]))))
by(df, :a, d -> DataFrame(c = d[:c], c_mean = mean(skipmissing(d[:c]))))
by(df, [:a, :b]) do d
    DataFrame(m = mean(skipmissing(d[:c])), v = var(skipmissing(d[:c])))
end
```

"""
by(d::AbstractDataFrame, cols, f::Function; sort::Bool = false) =
    combine(map(f, groupby(d, cols, sort = sort)))
by(f::Function, d::AbstractDataFrame, cols; sort::Bool = false) =
    by(d, cols, f, sort = sort)

#
# Aggregate convenience functions
#

# Applies a set of functions over a DataFrame, in the from of a cross-product
"""
Split-apply-combine that applies a set of functions over columns of an
AbstractDataFrame or GroupedDataFrame

```julia
aggregate(d::AbstractDataFrame, cols, fs)
aggregate(gd::GroupedDataFrame, fs)
```

### Arguments

* `d` : an AbstractDataFrame
* `gd` : a GroupedDataFrame
* `cols` : a column indicator (Symbol, Int, Vector{Symbol}, etc.)
* `fs` : a function or vector of functions to be applied to vectors
  within groups; expects each argument to be a column vector

Each `fs` should return a value or vector. All returns must be the
same length.

### Returns

* `::DataFrame`

### Examples

```julia
df = DataFrame(a = repeat([1, 2, 3, 4], outer=[2]),
               b = repeat([2, 1], outer=[4]),
               c = randn(8))
aggregate(df, :a, sum)
aggregate(df, :a, [sum, x->mean(skipmissing(x))])
aggregate(groupby(df, :a), [sum, x->mean(skipmissing(x))])
```

"""
aggregate(d::AbstractDataFrame, fs::Function; sort::Bool=false) = aggregate(d, [fs], sort=sort)
function aggregate(d::AbstractDataFrame, fs::Vector{T}; sort::Bool=false) where T<:Function
    headers = _makeheaders(fs, _names(d))
    _aggregate(d, fs, headers, sort)
end

# Applies aggregate to non-key cols of each SubDataFrame of a GroupedDataFrame
aggregate(gd::GroupedDataFrame, f::Function; sort::Bool=false) = aggregate(gd, [f], sort=sort)
function aggregate(gd::GroupedDataFrame, fs::Vector{T}; sort::Bool=false) where T<:Function
    headers = _makeheaders(fs, setdiff(_names(gd), gd.cols))
    res = combine(map(x -> _aggregate(without(x, gd.cols), fs, headers), gd))
    sort && sort!(res, cols=headers)
    res
end

# Groups DataFrame by cols before applying aggregate
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


#
# expanded from: include("dataframerow/dataframerow.jl")
#

# Container for a DataFrame row
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

# hash column element
Base.@propagate_inbounds hash_colel(v::AbstractArray, i, h::UInt = zero(UInt)) = hash(v[i], h)
Base.@propagate_inbounds hash_colel(v::AbstractCategoricalArray, i, h::UInt = zero(UInt)) =
    hash(CategoricalArrays.index(v.pool)[v.refs[i]], h)
Base.@propagate_inbounds function hash_colel(v::AbstractCategoricalArray{>: Missing}, i, h::UInt = zero(UInt))
    ref = v.refs[i]
    ref == 0 ? hash(missing, h) : hash(CategoricalArrays.index(v.pool)[ref], h)
end

# hash of DataFrame rows based on its values
# so that duplicate rows would have the same hash
# table columns are passed as a tuple of vectors to ensure type specialization
rowhash(cols::Tuple{AbstractVector}, r::Int, h::UInt = zero(UInt))::UInt =
    hash_colel(cols[1], r, h)
function rowhash(cols::Tuple{Vararg{AbstractVector}}, r::Int, h::UInt = zero(UInt))::UInt
    h = hash_colel(cols[1], r, h)
    rowhash(Base.tail(cols), r, h)
end

Base.hash(r::DataFrameRow, h::UInt = zero(UInt)) =
    rowhash(ntuple(i -> r.df[i], ncol(r.df)), r.row, h)

# comparison of DataFrame rows
# only the rows of the same DataFrame could be compared
# rows are equal if they have the same values (while the row indices could differ)
# if all non-missing values are equal, but there are missings, returns missing
Base.:(==)(r1::DataFrameRow, r2::DataFrameRow) = isequal(r1, r2)

function Base.isequal(r1::DataFrameRow, r2::DataFrameRow)
    isequal_row(r1.df, r1.row, r2.df, r2.row)
end

# internal method for comparing the elements of the same data table column
isequal_colel(col::AbstractArray, r1::Int, r2::Int) =
    (r1 == r2) || isequal(Base.unsafe_getindex(col, r1), Base.unsafe_getindex(col, r2))

# table columns are passed as a tuple of vectors to ensure type specialization
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

# lexicographic ordering on DataFrame rows, missing > !missing
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


#
# expanded from: include("dataframerow/utils.jl")
#

# Rows grouping.
# Maps row contents to the indices of all the equal rows.
# Used by groupby(), join(), nonunique()
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

# "kernel" functions for hashrows()
# adjust row hashes by the hashes of column elements
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

# should give the same hash as AbstractVector{T}
function hashrows_col!(h::Vector{UInt},
                       n::Vector{Bool},
                       v::AbstractCategoricalVector{T}) where T
    # TODO is it possible to optimize by hashing the pool values once?
    @inbounds for (i, ref) in enumerate(v.refs)
        h[i] = hash(CategoricalArrays.index(v.pool)[ref], h[i])
    end
    h
end

# should give the same hash as AbstractVector{T}
# enables efficient sequential memory access pattern
function hashrows_col!(h::Vector{UInt},
                       n::Vector{Bool},
                       v::AbstractCategoricalVector{>: Missing})
    # TODO is it possible to optimize by hashing the pool values once?
    @inbounds for (i, ref) in enumerate(v.refs)
        if ref == 0
            h[i] = hash(missing, h[i])
            length(n) > 0 && (n[i] = true)
        else
            h[i] = hash(CategoricalArrays.index(v.pool)[ref], h[i])
        end
    end
    h
end

# Calculate the vector of `df` rows hash values.
function hashrows(df::AbstractDataFrame, skipmissing::Bool)
    rhashes = zeros(UInt, nrow(df))
    missings = fill(false, skipmissing ? nrow(df) : 0)
    for col in columns(df)
        hashrows_col!(rhashes, missings, col)
    end
    return (rhashes, missings)
end

# Helper function for RowGroupDict.
# Returns a tuple:
# 1) the number of row groups in a data table
# 2) vector of row hashes
# 3) slot array for a hash map, non-zero values are
#    the indices of the first row in a group
# Optional group vector is set to the group indices of each row
function row_group_slots(df::AbstractDataFrame,
                         groups::Union{Vector{Int}, Void} = nothing,
                         skipmissing::Bool = false)
    rhashes, missings = hashrows(df, skipmissing)
    row_group_slots(ntuple(i -> df[i], ncol(df)), rhashes, missings, groups, skipmissing)
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

# Builds RowGroupDict for a given DataFrame.
# Partly uses the code of Wes McKinney's groupsort_indexer in pandas (file: src/groupby.pyx).
function group_rows(df::AbstractDataFrame, skipmissing::Bool = false)
    groups = Vector{Int}(nrow(df))
    ngroups, rhashes, gslots = row_group_slots(df, groups, skipmissing)

    # count elements in each group
    stops = zeros(Int, ngroups)
    @inbounds for g_ix in groups
        stops[g_ix] += 1
    end

    # group start positions in a sorted table
    starts = Vector{Int}(ngroups)
    if !isempty(starts)
        starts[1] = 1
        @inbounds for i in 1:(ngroups-1)
            starts[i+1] = starts[i] + stops[i]
        end
    end

    # define row permutation that sorts them into groups
    rperm = Vector{Int}(length(groups))
    copy!(stops, starts)
    @inbounds for (i, gix) in enumerate(groups)
        rperm[stops[gix]] = i
        stops[gix] += 1
    end
    stops .-= 1

    # drop group 1 which contains rows with missings in grouping columns
    if skipmissing
        splice!(starts, 1)
        splice!(stops, 1)
        ngroups -= 1
    end

    return RowGroupDict(df, ngroups, rhashes, gslots, groups, rperm, starts, stops)
end

# Find index of a row in gd that matches given row by content, 0 if not found
function findrow(gd::RowGroupDict,
                 df::DataFrame,
                 gd_cols::Tuple{Vararg{AbstractVector}},
                 df_cols::Tuple{Vararg{AbstractVector}},
                 row::Int)
    (gd.df === df) && return row # same table, return itself
    # different tables, content matching required
    rhash = rowhash(df_cols, row)
    szm1 = length(gd.gslots)-1
    slotix = ini_slotix = rhash & szm1 + 1
    while true
        g_row = gd.gslots[slotix]
        if g_row == 0 || # not found
            (rhash == gd.rhashes[g_row] &&
            isequal_row(gd_cols, g_row, df_cols, row)) # found
            return g_row
        end
        slotix = (slotix & szm1) + 1 # miss, try the next slot
        (slotix == ini_slotix) && break
    end
    return 0 # not found
end

# Find indices of rows in 'gd' that match given row by content.
# return empty set if no row matches
function findrows(gd::RowGroupDict,
                  df::DataFrame,
                  gd_cols::Tuple{Vararg{AbstractVector}},
                  df_cols::Tuple{Vararg{AbstractVector}},
                  row::Int)
    g_row = findrow(gd, df, gd_cols, df_cols, row)
    (g_row == 0) && return view(gd.rperm, 0:-1)
    gix = gd.groups[g_row]
    return view(gd.rperm, gd.starts[gix]:gd.stops[gix])
end

function Base.getindex(gd::RowGroupDict, dfr::DataFrameRow)
    g_row = findrow(gd, dfr.df, ntuple(i -> gd.df[i], ncol(gd.df)),
                    ntuple(i -> dfr.df[i], ncol(dfr.df)), dfr.row)
    (g_row == 0) && throw(KeyError(dfr))
    gix = gd.groups[g_row]
    return view(gd.rperm, gd.starts[gix]:gd.stops[gix])
end



#
# expanded from: include("abstractdataframe/iteration.jl")
#

##############################################################################
##
## Iteration: eachrow, eachcol
##
##############################################################################

# TODO: Reconsider/redesign eachrow -- ~100% overhead

# Iteration by rows
struct DFRowIterator{T <: AbstractDataFrame}
    df::T
end
"""
    eachrow(df) => DataFrames.DFRowIterator

Iterate a DataFrame row by row, with each row represented as a `DataFrameRow`,
which is a view that acts like a one-row DataFrame.
"""
eachrow(df::AbstractDataFrame) = DFRowIterator(df)

Base.start(itr::DFRowIterator) = 1
Base.done(itr::DFRowIterator, i::Int) = i > size(itr.df, 1)
Base.next(itr::DFRowIterator, i::Int) = (DataFrameRow(itr.df, i), i + 1)
Base.size(itr::DFRowIterator) = (size(itr.df, 1), )
Base.length(itr::DFRowIterator) = size(itr.df, 1)
Base.getindex(itr::DFRowIterator, i::Any) = DataFrameRow(itr.df, i)
Base.map(f::Function, dfri::DFRowIterator) = [f(row) for row in dfri]

# Iteration by columns
struct DFColumnIterator{T <: AbstractDataFrame}
    df::T
end
eachcol(df::AbstractDataFrame) = DFColumnIterator(df)

Base.start(itr::DFColumnIterator) = 1
Base.done(itr::DFColumnIterator, j::Int) = j > size(itr.df, 2)
Base.next(itr::DFColumnIterator, j::Int) = ((_names(itr.df)[j], itr.df[j]), j + 1)
Base.size(itr::DFColumnIterator) = (size(itr.df, 2), )
Base.length(itr::DFColumnIterator) = size(itr.df, 2)
Base.getindex(itr::DFColumnIterator, j::Any) = itr.df[:, j]
function Base.map(f::Function, dfci::DFColumnIterator)
    # note: `f` must return a consistent length
    res = DataFrame()
    for (n, v) in eachcol(dfci.df)
        res[n] = f(v)
    end
    res
end


#
# expanded from: include("abstractdataframe/join.jl")
#

##
## Join / merge
##

# Like similar, but returns a array that can have missings and is initialized with missings
similar_missing(dv::AbstractArray{T}, dims::Union{Int, Tuple{Vararg{Int}}}) where {T} =
    fill!(similar(dv, Union{T, Missing}, dims), missing)

const OnType = Union{Symbol, NTuple{2,Symbol}, Pair{Symbol,Symbol}}

# helper structure for DataFrames joining
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

# helper map between the row indices in original and joined table
struct RowIndexMap
    "row indices in the original table"
    orig::Vector{Int}
    "row indices in the resulting joined table"
    join::Vector{Int}
end

Base.length(x::RowIndexMap) = length(x.orig)

# composes the joined data table using the maps between the left and right
# table rows and the indices of rows in the result

function compose_joined_table(joiner::DataFrameJoiner, kind::Symbol,
                              left_ixs::RowIndexMap, leftonly_ixs::RowIndexMap,
                              right_ixs::RowIndexMap, rightonly_ixs::RowIndexMap;
                              makeunique::Bool=false)
    @assert length(left_ixs) == length(right_ixs)
    # compose left half of the result taking all left columns
    all_orig_left_ixs = vcat(left_ixs.orig, leftonly_ixs.orig)

    ril = length(right_ixs)
    lil = length(left_ixs)
    loil = length(leftonly_ixs)
    roil = length(rightonly_ixs)

    if loil > 0
        # combine the matched (left_ixs.orig) and non-matched (leftonly_ixs.orig) indices of the left table rows
        # preserving the original rows order
        all_orig_left_ixs = similar(left_ixs.orig, lil + loil)
        @inbounds all_orig_left_ixs[left_ixs.join] = left_ixs.orig
        @inbounds all_orig_left_ixs[leftonly_ixs.join] = leftonly_ixs.orig
    else
        # the result contains only the left rows that are matched to right rows (left_ixs)
        all_orig_left_ixs = left_ixs.orig # no need to copy left_ixs.orig as it's not used elsewhere
    end
    # permutation to swap rightonly and leftonly rows
    right_perm = vcat(1:ril, ril+roil+1:ril+roil+loil, ril+1:ril+roil)
    if length(leftonly_ixs) > 0
        # compose right_perm with the permutation that restores left rows order
        right_perm[vcat(right_ixs.join, leftonly_ixs.join)] = right_perm[1:ril+loil]
    end
    all_orig_right_ixs = vcat(right_ixs.orig, rightonly_ixs.orig)

    # compose right half of the result taking all right columns excluding on
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

# map the indices of the left and right joined tables
# to the indices of the rows in the resulting table
# if `nothing` is given, the corresponding map is not built
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
    left_table_cols = ntuple(i -> left_table[i], ncol(left_table))
    next_join_ix = 1
    for l_ix in 1:nrow(left_table)
        r_ixs = findrows(right_dict, left_table, right_dict_cols, left_table_cols, l_ix)
        if isempty(r_ixs)
            update!(leftonly_ixs, l_ix, next_join_ix)
            next_join_ix += 1
        else
            update!(left_ixs, l_ix, next_join_ix, length(r_ixs))
            update!(right_ixs, r_ixs, next_join_ix)
            update!(rightonly_mask, r_ixs)
            next_join_ix += length(r_ixs)
        end
    end
end

# map the row indices of the left and right joined tables
# to the indices of rows in the resulting table
# returns the 4-tuple of row indices maps for
# - matching left rows
# - non-matching left rows
# - matching right rows
# - non-matching right rows
# if false is provided, the corresponding map is not built and the
# tuple element is empty RowIndexMap
function update_row_maps!(left_table::AbstractDataFrame,
                          right_table::AbstractDataFrame,
                          right_dict::RowGroupDict,
                          map_left::Bool, map_leftonly::Bool,
                          map_right::Bool, map_rightonly::Bool)
    init_map(df::AbstractDataFrame, init::Bool) = init ?
        RowIndexMap(sizehint!(Vector{Int}(), nrow(df)),
                    sizehint!(Vector{Int}(), nrow(df))) : nothing
    to_bimap(x::RowIndexMap) = x
    to_bimap(::Void) = RowIndexMap(Vector{Int}(), Vector{Int}())

    # init maps as requested
    left_ixs = init_map(left_table, map_left)
    leftonly_ixs = init_map(left_table, map_leftonly)
    right_ixs = init_map(right_table, map_right)
    rightonly_mask = map_rightonly ? fill(true, nrow(right_table)) : nothing
    update_row_maps!(left_table, right_table, right_dict, left_ixs, leftonly_ixs, right_ixs, rightonly_mask)
    if map_rightonly
        rightonly_orig_ixs = find(rightonly_mask)
        rightonly_ixs = RowIndexMap(rightonly_orig_ixs,
                                    collect(length(right_ixs.orig) +
                                            (leftonly_ixs === nothing ? 0 : length(leftonly_ixs)) +
                                            (1:length(rightonly_orig_ixs))))
    else
        rightonly_ixs = nothing
    end

    return to_bimap(left_ixs), to_bimap(leftonly_ixs), to_bimap(right_ixs), to_bimap(rightonly_ixs)
end

"""
    join(df1, df2; on = Symbol[], kind = :inner, makeunique = false)

Join two `DataFrame` objects

### Arguments

* `df1`, `df2` : the two AbstractDataFrames to be joined

### Keyword Arguments

* `on` : A column, or vector of columns to join df1 and df2 on. If the column(s)
    that df1 and df2 will be joined on have different names, then the columns
    should be `(left, right)` tuples or `left => right` pairs, or a vector of
    such tuples or pairs. `on` is a required argument for all joins except for
    `kind = :cross`

* `kind` : the type of join, options include:

  - `:inner` : only include rows with keys that match in both `df1`
    and `df2`, the default
  - `:outer` : include all rows from `df1` and `df2`
  - `:left` : include all rows from `df1`
  - `:right` : include all rows from `df2`
  - `:semi` : return rows of `df1` that match with the keys in `df2`
  - `:anti` : return rows of `df1` that do not match with the keys in `df2`
  - `:cross` : a full Cartesian product of the key combinations; every
    row of `df1` is matched with every row of `df2`


* `makeunique` : if `false` (the default), an error will be raised
  if duplicate names are found in columns not joined on;
  if `true`, duplicate names will be suffixed with `_i`
  (`i` starting at 1 for the first duplicate).

For the three join operations that may introduce missing values (`:outer`, `:left`,
and `:right`), all columns of the returned data table will support missing values.

When merging `on` categorical columns that differ in the ordering of their levels, the
ordering of the left `DataFrame` takes precedence over the ordering of the right `DataFrame`

### Result

* `::DataFrame` : the joined DataFrame

### Examples

```julia
name = DataFrame(ID = [1, 2, 3], Name = ["John Doe", "Jane Doe", "Joe Blogs"])
job = DataFrame(ID = [1, 2, 4], Job = ["Lawyer", "Doctor", "Farmer"])

join(name, job, on = :ID)
join(name, job, on = :ID, kind = :outer)
join(name, job, on = :ID, kind = :left)
join(name, job, on = :ID, kind = :right)
join(name, job, on = :ID, kind = :semi)
join(name, job, on = :ID, kind = :anti)
join(name, job, kind = :cross)

job2 = DataFrame(identifier = [1, 2, 4], Job = ["Lawyer", "Doctor", "Farmer"])
join(name, job2, on = (:ID, :identifier))
join(name, job2, on = :ID => :identifier)
```

"""
function Base.join(df1::AbstractDataFrame,
                   df2::AbstractDataFrame;
                   on::Union{<:OnType, AbstractVector{<:OnType}} = Symbol[],
                   kind::Symbol = :inner, makeunique::Bool=false)
    if kind == :cross
        (on == Symbol[]) || throw(ArgumentError("Cross joins don't use argument 'on'."))
        return crossjoin(df1, df2, makeunique=makeunique)
    elseif on == Symbol[]
        throw(ArgumentError("Missing join argument 'on'."))
    end

    joiner = DataFrameJoiner(df1, df2, on)

    if kind == :inner
        compose_joined_table(joiner, kind, update_row_maps!(joiner.dfl_on, joiner.dfr_on,
                                                            group_rows(joiner.dfr_on),
                                                            true, false, true, false)...,
                                                            makeunique=makeunique)
    elseif kind == :left
        compose_joined_table(joiner, kind, update_row_maps!(joiner.dfl_on, joiner.dfr_on,
                                                            group_rows(joiner.dfr_on),
                                                            true, true, true, false)...,
                                                            makeunique=makeunique)
    elseif kind == :right
        compose_joined_table(joiner, kind, update_row_maps!(joiner.dfr_on, joiner.dfl_on,
                                                            group_rows(joiner.dfl_on),
                                                            true, true, true, false)[[3, 4, 1, 2]]...,
                                                            makeunique=makeunique)
    elseif kind == :outer
        compose_joined_table(joiner, kind, update_row_maps!(joiner.dfl_on, joiner.dfr_on,
                                                            group_rows(joiner.dfr_on),
                                                            true, true, true, true)...,
                                                            makeunique=makeunique)
    elseif kind == :semi
        # hash the right rows
        dfr_on_grp = group_rows(joiner.dfr_on)
        # iterate over left rows and leave those found in right
        left_ixs = Vector{Int}()
        sizehint!(left_ixs, nrow(joiner.dfl))
        dfr_on_grp_cols = ntuple(i -> dfr_on_grp.df[i], ncol(dfr_on_grp.df))
        dfl_on_cols = ntuple(i -> joiner.dfl_on[i], ncol(joiner.dfl_on))
        @inbounds for l_ix in 1:nrow(joiner.dfl_on)
            if findrow(dfr_on_grp, joiner.dfl_on, dfr_on_grp_cols, dfl_on_cols, l_ix) != 0
                push!(left_ixs, l_ix)
            end
        end
        return joiner.dfl[left_ixs, :]
    elseif kind == :anti
        # hash the right rows
        dfr_on_grp = group_rows(joiner.dfr_on)
        # iterate over left rows and leave those not found in right
        leftonly_ixs = Vector{Int}()
        sizehint!(leftonly_ixs, nrow(joiner.dfl))
        dfr_on_grp_cols = ntuple(i -> dfr_on_grp.df[i], ncol(dfr_on_grp.df))
        dfl_on_cols = ntuple(i -> joiner.dfl_on[i], ncol(joiner.dfl_on))
        @inbounds for l_ix in 1:nrow(joiner.dfl_on)
            if findrow(dfr_on_grp, joiner.dfl_on, dfr_on_grp_cols, dfl_on_cols, l_ix) == 0
                push!(leftonly_ixs, l_ix)
            end
        end
        return joiner.dfl[leftonly_ixs, :]
    else
        throw(ArgumentError("Unknown kind of join requested: $kind"))
    end
end

function crossjoin(df1::AbstractDataFrame, df2::AbstractDataFrame; makeunique::Bool=false)
    r1, r2 = size(df1, 1), size(df2, 1)
    colindex = merge(index(df1), index(df2), makeunique=makeunique)
    cols = Any[[repeat(c, inner=r2) for c in columns(df1)];
               [repeat(c, outer=r1) for c in columns(df2)]]
    DataFrame(cols, colindex)
end


#
# expanded from: include("abstractdataframe/reshape.jl")
#

##############################################################################
##
## Reshaping
##
## Also, see issue # ??
##
##############################################################################

##############################################################################
##
## stack()
## melt()
##
##############################################################################

"""
Stacks a DataFrame; convert from a wide to long format


```julia
stack(df::AbstractDataFrame, [measure_vars], [id_vars];
      variable_name::Symbol=:variable, value_name::Symbol=:value)
melt(df::AbstractDataFrame, [id_vars], [measure_vars];
     variable_name::Symbol=:variable, value_name::Symbol=:value)
```

### Arguments

* `df` : the AbstractDataFrame to be stacked

* `measure_vars` : the columns to be stacked (the measurement
  variables), a normal column indexing type, like a Symbol,
  Vector{Symbol}, Int, etc.; for `melt`, defaults to all
  variables that are not `id_vars`. If neither `measure_vars`
  or `id_vars` are given, `measure_vars` defaults to all
  floating point columns.

* `id_vars` : the identifier columns that are repeated during
  stacking, a normal column indexing type; for `stack` defaults to all
  variables that are not `measure_vars`

* `variable_name` : the name of the new stacked column that shall hold the names
  of each of `measure_vars`

* `value_name` : the name of the new stacked column containing the values from
  each of `measure_vars`


### Result

* `::DataFrame` : the long-format DataFrame with column `:value`
  holding the values of the stacked columns (`measure_vars`), with
  column `:variable` a Vector of Symbols with the `measure_vars` name,
  and with columns for each of the `id_vars`.

See also `stackdf` and `meltdf` for stacking methods that return a
view into the original DataFrame. See `unstack` for converting from
long to wide format.


### Examples

```julia
d1 = DataFrame(a = repeat([1:3;], inner = [4]),
               b = repeat([1:4;], inner = [3]),
               c = randn(12),
               d = randn(12),
               e = map(string, 'a':'l'))

d1s = stack(d1, [:c, :d])
d1s2 = stack(d1, [:c, :d], [:a])
d1m = melt(d1, [:a, :b, :e])
d1s_name = melt(d1, [:a, :b, :e], variable_name=:somemeasure)
```

"""
function stack(df::AbstractDataFrame, measure_vars::AbstractVector{<:Integer},
               id_vars::AbstractVector{<:Integer}; variable_name::Symbol=:variable,
               value_name::Symbol=:value)
    N = length(measure_vars)
    cnames = names(df)[id_vars]
    insert!(cnames, 1, value_name)
    insert!(cnames, 1, variable_name)
    DataFrame(Any[repeat(_names(df)[measure_vars], inner=nrow(df)),   # variable
                  vcat([df[c] for c in measure_vars]...),             # value
                  [repeat(df[c], outer=N) for c in id_vars]...],      # id_var columns
              cnames)
end
function stack(df::AbstractDataFrame, measure_var::Int, id_var::Int;
               variable_name::Symbol=:variable, value_name::Symbol=:value)
    stack(df, [measure_var], [id_var];
          variable_name=variable_name, value_name=value_name)
end
function stack(df::AbstractDataFrame, measure_vars::AbstractVector{<:Integer}, id_var::Int;
               variable_name::Symbol=:variable, value_name::Symbol=:value)
    stack(df, measure_vars, [id_var];
          variable_name=variable_name, value_name=value_name)
end
function stack(df::AbstractDataFrame, measure_var::Int, id_vars::AbstractVector{<:Integer};
               variable_name::Symbol=:variable, value_name::Symbol=:value)
    stack(df, [measure_var], id_vars;
          variable_name=variable_name, value_name=value_name)
end
function stack(df::AbstractDataFrame, measure_vars, id_vars;
               variable_name::Symbol=:variable, value_name::Symbol=:value)
    stack(df, index(df)[measure_vars], index(df)[id_vars];
          variable_name=variable_name, value_name=value_name)
end
# no vars specified, by default select only numeric columns
numeric_vars(df::AbstractDataFrame) =
    [T <: AbstractFloat || (T >: Missing && Missings.T(T) <: AbstractFloat)
     for T in eltypes(df)]

function stack(df::AbstractDataFrame, measure_vars = numeric_vars(df);
               variable_name::Symbol=:variable, value_name::Symbol=:value)
    mv_inds = index(df)[measure_vars]
    stack(df, mv_inds, setdiff(1:ncol(df), mv_inds);
          variable_name=variable_name, value_name=value_name)
end

"""
Stacks a DataFrame; convert from a wide to long format; see
`stack`.
"""
function melt(df::AbstractDataFrame, id_vars::Union{Int,Symbol};
              variable_name::Symbol=:variable, value_name::Symbol=:value)
    melt(df, [id_vars]; variable_name=variable_name, value_name=value_name)
end
function melt(df::AbstractDataFrame, id_vars;
              variable_name::Symbol=:variable, value_name::Symbol=:value)
    id_inds = index(df)[id_vars]
    stack(df, setdiff(1:ncol(df), id_inds), id_inds;
          variable_name=variable_name, value_name=value_name)
end
function melt(df::AbstractDataFrame, id_vars, measure_vars;
              variable_name::Symbol=:variable, value_name::Symbol=:value)
    stack(df, measure_vars, id_vars; variable_name=variable_name,
          value_name=value_name)
end
melt(df::AbstractDataFrame; variable_name::Symbol=:variable, value_name::Symbol=:value) =
    stack(df; variable_name=variable_name, value_name=value_name)

##############################################################################
##
## unstack()
##
##############################################################################

"""
Unstacks a DataFrame; convert from a long to wide format

```julia
unstack(df::AbstractDataFrame, rowkeys::Union{Symbol, Integer},
        colkey::Union{Symbol, Integer}, value::Union{Symbol, Integer})
unstack(df::AbstractDataFrame, rowkeys::AbstractVector{<:Union{Symbol, Integer}},
        colkey::Union{Symbol, Integer}, value::Union{Symbol, Integer})
unstack(df::AbstractDataFrame, colkey::Union{Symbol, Integer},
        value::Union{Symbol, Integer})
unstack(df::AbstractDataFrame)
```

### Arguments

* `df` : the AbstractDataFrame to be unstacked

* `rowkeys` : the column(s) with a unique key for each row, if not given,
  find a key by grouping on anything not a `colkey` or `value`

* `colkey` : the column holding the column names in wide format,
  defaults to `:variable`

* `value` : the value column, defaults to `:value`

### Result

* `::DataFrame` : the wide-format DataFrame

If `colkey` contains `missing` values then they will be skipped and a warning will be printed.

If combination of `rowkeys` and `colkey` contains duplicate entries then last `value` will
be retained and a warning will be printed.

### Examples

```julia
wide = DataFrame(id = 1:12,
                 a  = repeat([1:3;], inner = [4]),
                 b  = repeat([1:4;], inner = [3]),
                 c  = randn(12),
                 d  = randn(12))

long = stack(wide)
wide0 = unstack(long)
wide1 = unstack(long, :variable, :value)
wide2 = unstack(long, :id, :variable, :value)
wide3 = unstack(long, [:id, :a], :variable, :value)
```
Note that there are some differences between the widened results above.
"""
function unstack(df::AbstractDataFrame, rowkey::Int, colkey::Int, value::Int)
    refkeycol = categorical(df[rowkey])
    droplevels!(refkeycol)
    keycol = categorical(df[colkey])
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
    levs = levels(refkeycol)
    # we have to handle a case with missings in refkeycol as levs will skip missing
    col = similar(df[rowkey], length(levs) + hadmissing)
    copy!(col, levs)
    hadmissing && (col[end] = missing)
    df2 = DataFrame(unstacked_val, map(Symbol, levels(keycol)))
    insert!(df2, 1, col, _names(df)[rowkey])
end

unstack(df::AbstractDataFrame, rowkey::ColumnIndex,
        colkey::ColumnIndex, value::ColumnIndex) =
    unstack(df, index(df)[rowkey], index(df)[colkey], index(df)[value])

# Version of unstack with just the colkey and value columns provided
unstack(df::AbstractDataFrame, colkey::ColumnIndex, value::ColumnIndex) =
    unstack(df, index(df)[colkey], index(df)[value])

# group on anything not a key or value
unstack(df::AbstractDataFrame, colkey::Int, value::Int) =
    unstack(df, setdiff(_names(df), _names(df)[[colkey, value]]), colkey, value)

unstack(df::AbstractDataFrame, rowkeys, colkey::ColumnIndex, value::ColumnIndex) =
    unstack(df, rowkeys, index(df)[colkey], index(df)[value])

unstack(df::AbstractDataFrame, rowkeys::AbstractVector{<:Real}, colkey::Int, value::Int) =
    unstack(df, names(df)[rowkeys], colkey, value)

function unstack(df::AbstractDataFrame, rowkeys::AbstractVector{Symbol}, colkey::Int, value::Int)
    length(rowkeys) == 0 && throw(ArgumentError("No key column found"))
    length(rowkeys) == 1 && return unstack(df, rowkeys[1], colkey, value)
    g = groupby(df, rowkeys, sort=true)
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


##############################################################################
##
## Reshaping using referencing (issue #145)
## New AbstractVector types (all read only):
##     StackedVector
##     RepeatedVector
##
##############################################################################

"""
An AbstractVector{Any} that is a linear, concatenated view into
another set of AbstractVectors

NOTE: Not exported.

### Constructor

```julia
StackedVector(d::AbstractVector...)
```

### Arguments

* `d...` : one or more AbstractVectors

### Examples

```julia
StackedVector(Any[[1,2], [9,10], [11,12]])  # [1,2,9,10,11,12]
```

"""
mutable struct StackedVector <: AbstractVector{Any}
    components::Vector{Any}
end

function Base.getindex(v::StackedVector,i::Real)
    lengths = [length(x)::Int for x in v.components]
    cumlengths = [0; cumsum(lengths)]
    j = searchsortedlast(cumlengths .+ 1, i)
    if j > length(cumlengths)
        error("indexing bounds error")
    end
    k = i - cumlengths[j]
    if k < 1 || k > length(v.components[j])
        error("indexing bounds error")
    end
    v.components[j][k]
end

function Base.getindex(v::StackedVector,i::AbstractVector{I}) where I<:Real
    result = similar(v.components[1], length(i))
    for idx in 1:length(i)
        result[idx] = v[i[idx]]
    end
    result
end

Base.size(v::StackedVector) = (length(v),)
Base.length(v::StackedVector) = sum(map(length, v.components))
Base.ndims(v::StackedVector) = 1
Base.eltype(v::StackedVector) = promote_type(map(eltype, v.components)...)
Base.similar(v::StackedVector, T::Type, dims::Union{Integer, AbstractUnitRange}...) =
    similar(v.components[1], T, dims...)

CategoricalArrays.CategoricalArray(v::StackedVector) = CategoricalArray(v[:]) # could be more efficient


"""
An AbstractVector that is a view into another AbstractVector with
repeated elements

NOTE: Not exported.

### Constructor

```julia
RepeatedVector(parent::AbstractVector, inner::Int, outer::Int)
```

### Arguments

* `parent` : the AbstractVector that's repeated
* `inner` : the numer of times each element is repeated
* `outer` : the numer of times the whole vector is repeated after
  expanded by `inner`

`inner` and `outer` have the same meaning as similarly named arguments
to `repeat`.

### Examples

```julia
RepeatedVector([1,2], 3, 1)   # [1,1,1,2,2,2]
RepeatedVector([1,2], 1, 3)   # [1,2,1,2,1,2]
RepeatedVector([1,2], 2, 2)   # [1,2,1,2,1,2,1,2]
```

"""
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

##############################################################################
##
## stackdf()
## meltdf()
## Reshaping using referencing (issue #145), using the above vector types
##
##############################################################################

"""
A stacked view of a DataFrame (long format)

Like `stack` and `melt`, but a view is returned rather than data
copies.

```julia
stackdf(df::AbstractDataFrame, [measure_vars], [id_vars];
        variable_name::Symbol=:variable, value_name::Symbol=:value)
meltdf(df::AbstractDataFrame, [id_vars], [measure_vars];
       variable_name::Symbol=:variable, value_name::Symbol=:value)
```

### Arguments

* `df` : the wide AbstractDataFrame

* `measure_vars` : the columns to be stacked (the measurement
  variables), a normal column indexing type, like a Symbol,
  Vector{Symbol}, Int, etc.; for `melt`, defaults to all
  variables that are not `id_vars`

* `id_vars` : the identifier columns that are repeated during
  stacking, a normal column indexing type; for `stack` defaults to all
  variables that are not `measure_vars`

### Result

* `::DataFrame` : the long-format DataFrame with column `:value`
  holding the values of the stacked columns (`measure_vars`), with
  column `:variable` a Vector of Symbols with the `measure_vars` name,
  and with columns for each of the `id_vars`.

The result is a view because the columns are special AbstractVectors
that return indexed views into the original DataFrame.

### Examples

```julia
d1 = DataFrame(a = repeat([1:3;], inner = [4]),
               b = repeat([1:4;], inner = [3]),
               c = randn(12),
               d = randn(12),
               e = map(string, 'a':'l'))

d1s = stackdf(d1, [:c, :d])
d1s2 = stackdf(d1, [:c, :d], [:a])
d1m = meltdf(d1, [:a, :b, :e])
```

"""
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
    stackdf(df, [measure_var], [id_var]; variable_name=variable_name,
            value_name=value_name)
end
function stackdf(df::AbstractDataFrame, measure_vars, id_var::Int;
                 variable_name::Symbol=:variable, value_name::Symbol=:value)
    stackdf(df, measure_vars, [id_var]; variable_name=variable_name,
            value_name=value_name)
end
function stackdf(df::AbstractDataFrame, measure_var::Int, id_vars;
                 variable_name::Symbol=:variable, value_name::Symbol=:value)
    stackdf(df, [measure_var], id_vars; variable_name=variable_name,
            value_name=value_name)
end
function stackdf(df::AbstractDataFrame, measure_vars, id_vars;
                 variable_name::Symbol=:variable, value_name::Symbol=:value)
    stackdf(df, index(df)[measure_vars], index(df)[id_vars];
            variable_name=variable_name, value_name=value_name)
end
function stackdf(df::AbstractDataFrame, measure_vars = numeric_vars(df);
                 variable_name::Symbol=:variable, value_name::Symbol=:value)
    m_inds = index(df)[measure_vars]
    stackdf(df, m_inds, setdiff(1:ncol(df), m_inds);
            variable_name=variable_name, value_name=value_name)
end

"""
A stacked view of a DataFrame (long format); see `stackdf`
"""
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



#
# expanded from: include("abstractdataframe/io.jl")
#

##############################################################################
#
# Text output
#
##############################################################################

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
    printtable(STDOUT,
               df,
               header = header,
               separator = separator,
               quotemark = quotemark,
               nastring = nastring)
    return
end
##############################################################################
#
# HTML output
#
##############################################################################

function html_escape(cell::AbstractString)
    cell = replace(cell, "&", "&amp;")
    cell = replace(cell, "<", "&lt;")
    cell = replace(cell, ">", "&gt;")
    return cell
end

function Base.show(io::IO, ::MIME"text/html", df::AbstractDataFrame)
    cnames = _names(df)
    write(io, "<table class=\"data-frame\">")
    write(io, "<thead>")
    write(io, "<tr>")
    write(io, "<th></th>")
    for column_name in cnames
        write(io, "<th>$column_name</th>")
    end
    write(io, "</tr>")
    write(io, "</thead>")
    write(io, "<tbody>")
    haslimit = get(io, :limit, true)
    n = size(df, 1)
    if haslimit
        tty_rows, tty_cols = displaysize(io)
        mxrow = min(n,tty_rows)
    else
        mxrow = n
    end
    for row in 1:mxrow
        write(io, "<tr>")
        write(io, "<th>$row</th>")
        for column_name in cnames
            cell = sprint(ourshowcompact, df[row, column_name])
            write(io, "<td>$(html_escape(cell))</td>")
        end
        write(io, "</tr>")
    end
    if n > mxrow
        write(io, "<tr>")
        write(io, "<th>&vellip;</th>")
        for column_name in cnames
            write(io, "<td>&vellip;</td>")
        end
        write(io, "</tr>")
    end
    write(io, "</tbody>")
    write(io, "</table>")
end

##############################################################################
#
# LaTeX output
#
##############################################################################

function latex_char_escape(char::AbstractString)
    if char == "\\"
        return "\\textbackslash{}"
    elseif char == "~"
        return "\\textasciitilde{}"
    else
        return string("\\", char)
    end
end

function latex_escape(cell::AbstractString)
    cell = replace(cell, ['\\','~','#','$','%','&','_','^','{','}'], latex_char_escape)
    return cell
end

function Base.show(io::IO, ::MIME"text/latex", df::AbstractDataFrame)
    nrows = size(df, 1)
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

##############################################################################
#
# MIME
#
##############################################################################

function Base.show(io::IO, ::MIME"text/csv", df::AbstractDataFrame)
    printtable(io, df, true, ',')
end

function Base.show(io::IO, ::MIME"text/tab-separated-values", df::AbstractDataFrame)
    printtable(io, df, true, '\t')
end

##############################################################################
#
# DataStreams-based IO
#
##############################################################################

using DataStreams, WeakRefStrings

struct DataFrameStream{T}
    columns::T
    header::Vector{String}
end
DataFrameStream(df::DataFrame) = DataFrameStream(Tuple(df.columns), string.(names(df)))

# DataFrame Data.Source implementation
function Data.schema(df::DataFrame)
    return Data.Schema(Type[eltype(A) for A in df.columns],
                       string.(names(df)), length(df) == 0 ? 0 : length(df.columns[1]))
end

Data.isdone(source::DataFrame, row, col, rows, cols) = row > rows || col > cols
function Data.isdone(source::DataFrame, row, col)
    cols = length(source)
    return Data.isdone(source, row, col, cols == 0 ? 0 : length(source.columns[1]), cols)
end

Data.streamtype(::Type{DataFrame}, ::Type{Data.Column}) = true
Data.streamtype(::Type{DataFrame}, ::Type{Data.Field}) = true

Data.streamfrom(source::DataFrame, ::Type{Data.Column}, ::Type{T}, row, col) where {T} =
    source[col]
Data.streamfrom(source::DataFrame, ::Type{Data.Field}, ::Type{T}, row, col) where {T} =
    source[col][row]

# DataFrame Data.Sink implementation
Data.streamtypes(::Type{DataFrame}) = [Data.Column, Data.Field]
Data.weakrefstrings(::Type{DataFrame}) = true

allocate(::Type{T}, rows, ref) where {T} = Vector{T}(uninitialized, rows)
allocate(::Type{CategoricalString{R}}, rows, ref) where {R} = CategoricalArray{String, 1, R}(rows)
allocate(::Type{Union{CategoricalString{R}, Missing}}, rows, ref) where {R} = CategoricalArray{Union{String, Missing}, 1, R}(rows)
allocate(::Type{CategoricalValue{T, R}}, rows, ref) where {T, R} =
    CategoricalArray{T, 1, R}(rows)
allocate(::Type{Union{Missing, CategoricalValue{T, R}}}, rows, ref) where {T, R} =
    CategoricalArray{Union{Missing, T}, 1, R}(rows)
allocate(::Type{WeakRefString{T}}, rows, ref) where {T} =
    WeakRefStringArray(ref, WeakRefString{T}, rows)
allocate(::Type{Union{Missing, WeakRefString{T}}}, rows, ref) where {T} =
    WeakRefStringArray(ref, Union{Missing, WeakRefString{T}}, rows)
allocate(::Type{Missing}, rows, ref) = missings(rows)

# Construct or modify a DataFrame to be ready to stream data from a source with `sch`
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
    append!(sink.columns[col], column)
end
    
Data.close!(df::DataFrameStream) =
    DataFrame(collect(Any, df.columns), Symbol.(df.header), makeunique=true)




#
# expanded from: include("abstractdataframe/show.jl")
#

#' @exported
#' @description
#'
#' Returns a string summary of an AbstractDataFrame in a standardized
#' form. For example, a standard DataFrame with 10 rows and 5 columns
#' will be summarized as "10×5 DataFrame".
#'
#' @param df::AbstractDataFrame The AbstractDataFrame to be summarized.
#'
#' @returns res::String The summary of `df`.
#'
#' @examples
#'
#' summary(DataFrame(A = 1:10))
function Base.summary(df::AbstractDataFrame) # -> String
    nrows, ncols = size(df)
    return @sprintf("%d×%d %s", nrows, ncols, typeof(df))
end

#' @description
#'
#' Determine the number of characters that would be used to print a value.
#'
#' @param x::Any A value whose string width will be computed.
#'
#' @returns w::Int The width of the string.
#'
#' @examples
#'
#' ourstrwidth("abc")
#' ourstrwidth(10000)
let
    local io = IOBuffer(Vector{UInt8}(80), true, true)
    global ourstrwidth
    function ourstrwidth(x::Any) # -> Int
        truncate(io, 0)
        ourshowcompact(io, x)
        textwidth(String(take!(io)))
    end
end

#' @description
#'
#' Render a value to an IO object in a compact format. Unlike
#' Base.showcompact, we render strings without surrounding quote
#' marks.
#'
#' @param io::IO An IO object to be printed to.
#' @param x::Any A value to be printed.
#'
#' @returns x::Void A `nothing` value.
#'
#' @examples
#'
#' ourshowcompact(STDOUT, "abc")
#' ourshowcompact(STDOUT, 10000)
ourshowcompact(io::IO, x::Any) = showcompact(io, x) # -> Void
ourshowcompact(io::IO, x::AbstractString) = escape_string(io, x, "") # -> Void
ourshowcompact(io::IO, x::Symbol) = ourshowcompact(io, string(x)) # -> Void

#' @description
#'
#' Calculates, for each column of an AbstractDataFrame, the maximum
#' string width used to render either the name of that column or the
#' longest entry in that column -- among the rows of the AbstractDataFrame
#' will be rendered to IO. The widths for all columns are returned as a
#' vector.
#'
#' NOTE: The last entry of the result vector is the string width of the
#'       implicit row ID column contained in every AbstractDataFrame.
#'
#' @param df::AbstractDataFrame The AbstractDataFrame whose columns will be
#'        printed.
#' @param rowindices1::AbstractVector{Int} A set of indices of the first
#'        chunk of the AbstractDataFrame that would be rendered to IO.
#' @param rowindices2::AbstractVector{Int} A set of indices of the second
#'        chunk of the AbstractDataFrame that would be rendered to IO. Can
#'        be empty if the AbstractDataFrame would be printed without any
#'        ellipses.
#' @param rowlabel::AbstractString The label that will be used when rendered the
#'        numeric ID's of each row. Typically, this will be set to "Row".
#'
#' @returns widths::Vector{Int} The maximum string widths required to render
#'          each column, including that column's name.
#'
#' @examples
#'
#' df = DataFrame(A = 1:3, B = ["x", "yy", "z"])
#' maxwidths = getmaxwidths(df, 1:1, 3:3, :Row)
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
                maxwidth = max(maxwidth, ourstrwidth(col[i]))
            catch
                maxwidth = max(maxwidth, undefstrwidth)
            end
        end
        maxwidths[j] = maxwidth
        j += 1
    end

    rowmaxwidth1 = isempty(rowindices1) ? 0 : ndigits(maximum(rowindices1))
    rowmaxwidth2 = isempty(rowindices2) ? 0 : ndigits(maximum(rowindices2))

    maxwidths[j] = max(max(rowmaxwidth1, rowmaxwidth2), ourstrwidth(rowlabel))

    return maxwidths
end

#' @description
#'
#' Given the maximum widths required to render each column of an
#' AbstractDataFrame, this returns the total number of characters
#' that would be required to render an entire row to an IO system.
#'
#' NOTE: This width includes the whitespace and special characters used to
#'       pretty print the AbstractDataFrame.
#'
#' @param maxwidths::Vector{Int} The maximum width needed to render each
#'        column of an AbstractDataFrame.
#'
#' @returns totalwidth::Int The total width required to render a complete row
#'          of the AbstractDataFrame for which `maxwidths` was computed.
#'
#' @examples
#'
#' df = DataFrame(A = 1:3, B = ["x", "yy", "z"])
#' maxwidths = getmaxwidths(df, 1:1, 3:3, "Row")
#' totalwidth = getprintedwidth(maxwidths))
function getprintedwidth(maxwidths::Vector{Int}) # -> Int
    # Include length of line-initial |
    totalwidth = 1
    for i in 1:length(maxwidths)
        # Include length of field + 2 spaces + trailing |
        totalwidth += maxwidths[i] + 3
    end
    return totalwidth
end

#' @description
#'
#' When rendering an AbstractDataFrame to a REPL window in chunks, each of
#' which will fit within the width of the REPL window, this function will
#' return the indices of the columns that should be included in each chunk.
#'
#' NOTE: The resulting bounds should be interpreted as follows: the
#'       i-th chunk bound is the index MINUS 1 of the first column in the
#'       i-th chunk. The (i + 1)-th chunk bound is the EXACT index of the
#'       last column in the i-th chunk. For example, the bounds [0, 3, 5]
#'       imply that the first chunk contains columns 1-3 and the second chunk
#'       contains columns 4-5.
#'
#' @param maxwidths::Vector{Int} The maximum width needed to render each
#'        column of an AbstractDataFrame.
#' @param splitchunks::Bool Should the output be split into chunks at all or
#'        should only one chunk be constructed for the entire
#'        AbstractDataFrame?
#' @param availablewidth::Int The available width in the REPL.
#'
#' @returns chunkbounds::Vector{Int} The bounds of each chunk of columns.
#'
#' @examples
#'
#' df = DataFrame(A = 1:3, B = ["x", "yy", "z"])
#' maxwidths = getmaxwidths(df, 1:1, 3:3, "Row")
#' chunkbounds = getchunkbounds(maxwidths, true)
function getchunkbounds(maxwidths::Vector{Int},
                        splitchunks::Bool,
                        availablewidth::Int=displaysize()[2]) # -> Vector{Int}
    ncols = length(maxwidths) - 1
    rowmaxwidth = maxwidths[ncols + 1]
    if splitchunks
        chunkbounds = [0]
        # Include 2 spaces + 2 | characters for row/col label
        totalwidth = rowmaxwidth + 4
        for j in 1:ncols
            # Include 2 spaces + | character in per-column character count
            totalwidth += maxwidths[j] + 3
            if totalwidth > availablewidth
                push!(chunkbounds, j - 1)
                totalwidth = rowmaxwidth + 4 + maxwidths[j] + 3
            end
        end
        push!(chunkbounds, ncols)
    else
        chunkbounds = [0, ncols]
    end
    return chunkbounds
end

#' @description
#'
#' Render a subset of rows and columns of an AbstractDataFrame to an
#' IO system. For chunked printing, this function is used to print a
#' single chunk, starting from the first indicated column and ending with
#' the last indicated column. Assumes that the maximum string widths
#' required for printing have been precomputed.
#'
#' @param io::IO The IO system to which `df` will be printed.
#' @param df::AbstractDataFrame An AbstractDataFrame.
#' @param rowindices::AbstractVector{Int} The indices of the subset of rows
#'        that will be rendered to `io`.
#' @param maxwidths::Vector{Int} The pre-computed maximum string width
#'        required to render each column.
#' @param leftcol::Int The index of the first column in a chunk to be
#'        rendered.
#' @param rightcol::Int The index of the last column in a chunk to be
#'        rendered.
#'
#' @returns o::Void A `nothing` value.
#'
#' @examples
#'
#' df = DataFrame(A = 1:3, B = ["x", "y", "z"])
#' showrowindices(STDOUT, df, 1:2, [1, 1, 5], 1, 2)
function showrowindices(io::IO,
                        df::AbstractDataFrame,
                        rowindices::AbstractVector{Int},
                        maxwidths::Vector{Int},
                        leftcol::Int,
                        rightcol::Int) # -> Void
    rowmaxwidth = maxwidths[end]

    for i in rowindices
        # Print row ID
        @printf io "│ %d" i
        padding = rowmaxwidth - ndigits(i)
        for _ in 1:padding
            write(io, ' ')
        end
        print(io, " │ ")
        # Print DataFrame entry
        for j in leftcol:rightcol
            strlen = 0
            try
                s = df[i, j]
                strlen = ourstrwidth(s)
                if ismissing(s)
                    print_with_color(:light_black, io, s)
                else
                    ourshowcompact(io, s)
                end
            catch
                strlen = ourstrwidth(Base.undef_ref_str)
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

#' @description
#'
#' Render a subset of rows (possibly in chunks) of an AbstractDataFrame to an
#' IO system. Users can control
#'
#' NOTE: The value of `maxwidths[end]` must be the string width of
#' `rowlabel`.
#'
#' @param io::IO The IO system to which `df` will be printed.
#' @param df::AbstractDataFrame An AbstractDataFrame.
#' @param rowindices1::AbstractVector{Int} The indices of the first subset
#'        of rows to be rendered.
#' @param rowindices2::AbstractVector{Int} The indices of the second subset
#'        of rows to be rendered. An ellipsis will be printed before
#'        rendering this second subset of rows.
#' @param maxwidths::Vector{Int} The pre-computed maximum string width
#'        required to render each column.
#' @param splitchunks::Bool Should the printing of the AbstractDataFrame
#'        be done in chunks? Defaults to `false`.
#' @param allcols::Bool Should only one chunk be printed if printing in
#'        chunks? Defaults to `true`.
#' @param rowlabel::Symbol What label should be printed when rendering the
#'        numeric ID's of each row? Defaults to `"Row"`.
#' @param displaysummary::Bool Should a brief string summary of the
#'        AbstractDataFrame be rendered to the IO system before printing the
#'        contents of the renderable rows? Defaults to `true`.
#'
#' @returns o::Void A `nothing` value.
#'
#' @examples
#'
#' df = DataFrame(A = 1:3, B = ["x", "y", "z"])
#' showrows(STDOUT, df, 1:2, 3:3, [1, 1, 5], false, :Row, true)
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
        return
    end

    rowmaxwidth = maxwidths[ncols + 1]
    chunkbounds = getchunkbounds(maxwidths, splitchunks, displaysize(io)[2])
    nchunks = allcols ? length(chunkbounds) - 1 : min(length(chunkbounds) - 1, 1)

    header = displaysummary ? summary(df) : ""
    if !allcols && length(chunkbounds) > 2
        header *= ". Omitted printing of $(chunkbounds[end] - chunkbounds[2]) columns"
    end
    println(io, header)

    for chunkindex in 1:nchunks
        leftcol = chunkbounds[chunkindex] + 1
        rightcol = chunkbounds[chunkindex + 1]

        # Print column names
        @printf io "│ %s" rowlabel
        padding = rowmaxwidth - ourstrwidth(rowlabel)
        for itr in 1:padding
            write(io, ' ')
        end
        @printf io " │ "
        for j in leftcol:rightcol
            s = _names(df)[j]
            ourshowcompact(io, s)
            padding = maxwidths[j] - ourstrwidth(s)
            for itr in 1:padding
                write(io, ' ')
            end
            if j == rightcol
                print(io, " │\n")
            else
                print(io, " │ ")
            end
        end

        # Print table bounding line
        write(io, '├')
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

#' @exported
#' @description
#'
#' Render an AbstractDataFrame to an IO system. The specific visual
#' representation chosen depends on the width of the REPL window
#' from which the call to `show` derives. The dynamic response 
#' to screen width can be configured using the `allcols` argument.
#'
#' @param io::IO The IO system to which `df` will be printed.
#' @param df::AbstractDataFrame An AbstractDataFrame.
#' @param allcols::Bool Should only a subset of columns that fits
#'        the device width be printed? Defaults to `false`.
#' @param rowlabel::Symbol What label should be printed when rendering the
#'        numeric ID's of each row? Defaults to `"Row"`.
#' @param displaysummary::Bool Should a brief string summary of the
#'        AbstractDataFrame be rendered to the IO system before printing the
#'        contents of the renderable rows? Defaults to `true`.
#'
#' @returns o::Void A `nothing` value.
#'
#' @examples
#'
#' df = DataFrame(A = 1:3, B = ["x", "y", "z"])
#' show(STDOUT, df, false, :Row, true)
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

#' @exported
#' @description
#'
#' Render an AbstractDataFrame to STDOUT with or without chunking. See
#' other `show` documentation for details. This is mainly used to force
#' showing the AbstractDataFrame in chunks.
#'
#' @param df::AbstractDataFrame An AbstractDataFrame.
#' @param allcols::Bool Should only a subset of columns that fits
#'        the device width be printed? Defaults to `false`.
#'
#' @returns o::Void A `nothing` value.
#'
#' @examples
#'
#' df = DataFrame(A = 1:3, B = ["x", "y", "z"])
#' show(df, true)
function Base.show(df::AbstractDataFrame,
                   allcols::Bool = false) # -> Void
    return show(STDOUT, df, allcols)
end

#' @exported
#' @description
#'
#' Render all of the rows of an AbstractDataFrame to an IO system. See
#' `show` documentation for details.
#'
#' @param io::IO The IO system to which `df` will be printed.
#' @param df::AbstractDataFrame An AbstractDataFrame.
#' @param allcols::Bool Should only a subset of columns that fits
#'        the device width be printed? Defaults to `true`.
#' @param rowlabel::Symbol What label should be printed when rendering the
#'        numeric ID's of each row? Defaults to `"Row"`.
#' @param displaysummary::Bool Should a brief string summary of the
#'        AbstractDataFrame be rendered to the IO system before printing the
#'        contents of the renderable rows? Defaults to `true`.
#'
#' @returns o::Void A `nothing` value.
#'
#' @examples
#'
#' df = DataFrame(A = 1:3, B = ["x", "y", "z"])
#' showall(STDOUT, df, false, :Row, true)
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

#' @exported
#' @description
#'
#' Render all of the rows of an AbstractDataFrame to STDOUT. See
#' `showall` documentation for details.
#'
#' @param df::AbstractDataFrame An AbstractDataFrame.
#' @param allcols::Bool Should only a subset of columns that fits
#'        the device width be printed? Defaults to `true`.
#'
#' @returns o::Void A `nothing` value.
#'
#' @examples
#'
#' df = DataFrame(A = 1:3, B = ["x", "y", "z"])
#' showall(df, true)
function Base.showall(df::AbstractDataFrame,
                      allcols::Bool = true) # -> Void
    showall(STDOUT, df, allcols)
    return
end

#' @exported
#' @description
#'
#' Render a summary of the column names, column types and column missingness
#' count.
#'
#' @param io::IO The `io` to be rendered to.
#' @param df::AbstractDataFrame An AbstractDataFrame.
#' @param all::Bool If `false` (default), only a subset of columns
#'        fitting on the screen is printed.
#' @param values::Bool If `true` (default), the first and the last value of
#'        each column are printed.
#'
#' @returns o::Void A `nothing` value.
#'
#' @examples
#'
#' df = DataFrame(A = 1:3, B = ["x", "y", "z"])
#' showcols(df)
function showcols(io::IO, df::AbstractDataFrame, all::Bool = false,
                  values::Bool = true) # -> Void
    print(io, summary(df))
    metadata = DataFrame(Name = _names(df),
                         Eltype = eltypes(df),
                         Missing = colmissing(df))
    nrows, ncols = size(df)
    if values && nrows > 0
        if nrows == 1
            metadata[:Values] = [sprint(ourshowcompact, df[1, i]) for i in 1:ncols]
        else
            metadata[:Values] = [sprint(ourshowcompact, df[1, i]) * "  …  " *
                                 sprint(ourshowcompact, df[end, i]) for i in 1:ncols]
        end
    end
    (all ? showall : show)(io, metadata, true, Symbol("Col #"), false)
    return
end

#' @exported
#' @description
#'
#' Render a summary of the column names, column types and column missingness
#' count.
#'
#' @param df::AbstractDataFrame An AbstractDataFrame.
#' @param all::Bool If `false` (default), only a subset of columns
#'        fitting on the screen is printed.
#' @param values::Bool If `true` (default), first and last value of
#'        each column is printed.
#'
#' @returns o::Void A `nothing` value.
#'
#' @examples
#'
#' df = DataFrame(A = 1:3, B = ["x", "y", "z"])
#' showcols(df)
function showcols(df::AbstractDataFrame, all::Bool=false, values::Bool=true)
    showcols(STDOUT, df, all, values) # -> Void
end


#
# expanded from: include("groupeddataframe/show.jl")
#

function Base.show(io::IO, gd::GroupedDataFrame)
    N = length(gd)
    println(io, "$(typeof(gd))  $N groups with keys: $(gd.cols)")
    if N > 0
        println(io, "First Group:")
        show(io, gd[1])
    end
    if N > 1
        print(io, "\n⋮\n")
        println(io, "Last Group:")
        show(io, gd[N])
    end
end

function Base.showall(io::IO, gd::GroupedDataFrame)
    N = length(gd)
    println(io, "$(typeof(gd))  $N groups with keys: $(gd.cols)")
    for i = 1:N
        println(io, "gd[$i]:")
        show(io, gd[i])
    end
end


#
# expanded from: include("dataframerow/show.jl")
#

#' @exported
#' @description
#'
#' Render a DataFrameRow to an IO system. Each column of the DataFrameRow
#' is printed on a separate line.
#'
#' @param io::IO The IO system where rendering will take place.
#' @param r::DataFrameRow The DataFrameRow to be rendered to `io`.
#'
#' @returns o::Void A `nothing` value.
#'
#' @examples
#'
#' df = DataFrame(A = 1:3, B = ["x", "y", "z"])
#' for r in eachrow(df)
#'     show(STDOUT, r)
#' end
function Base.show(io::IO, r::DataFrameRow)
    labelwidth = mapreduce(n -> length(string(n)), max, _names(r)) + 2
    @printf(io, "DataFrameRow (row %d)\n", r.row)
    for (label, value) in r
        println(io, rpad(label, labelwidth, ' '), value)
    end
end



#
# expanded from: include("abstractdataframe/sort.jl")
#

##############################################################################
##
## Sorting
##
##############################################################################

#########################################
## Permutation & Ordering types/functions
#########################################
# Sorting in Julia works through Orderings, where each ordering is a type
# which defines a comparison function lt(o::Ord, a, b).

# UserColOrdering: user ordering of a column; this is just a convenience container
#                  which allows a user to specify column specific orderings
#                  with "order(column, rev=true,...)"

mutable struct UserColOrdering{T<:ColumnIndex}
    col::T
    kwargs
end

# This is exported, and lets a user define orderings for a particular column
order(col::T; kwargs...) where {T<:ColumnIndex} = UserColOrdering{T}(col, kwargs)

# Allow getting the column even if it is not wrapped in a UserColOrdering
_getcol(o::UserColOrdering) = o.col
_getcol(x) = x

###
# Get an Ordering for a single column
###
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

# DFPerm: defines a permutation on a particular DataFrame, using
#         a single ordering (O<:Ordering) or a list of column orderings
#         (O<:AbstractVector{Ordering}), one per DataFrame column
#
#         If a user only specifies a few columns, the DataFrame
#         contained in the DFPerm only contains those columns, and
#         the permutation induced by this ordering is used to
#         sort the original (presumably larger) DataFrame

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

# get ordering function for the i-th column used for ordering
col_ordering(o::DFPerm{O}, i::Int) where {O<:Ordering} = o.ord
col_ordering(o::DFPerm{V}, i::Int) where {V<:AbstractVector} = o.ord[i]

Base.@propagate_inbounds Base.getindex(o::DFPerm, i::Int, j::Int) = o.df[i, j]
Base.@propagate_inbounds Base.getindex(o::DFPerm, a::DataFrameRow, j::Int) = a[j]

function Sort.lt(o::DFPerm, a, b)
    @inbounds for i = 1:ncol(o.df)
        ord = col_ordering(o, i)
        va = o[a, i]
        vb = o[b, i]
        lt(ord, va, vb) && return true
        lt(ord, vb, va) && return false
    end
    false # a and b are equal
end

###
# Get an Ordering for a DataFrame
###

################
## Case 1: no columns requested, so sort by all columns using requested order
################
## Case 1a: single order
######
ordering(df::AbstractDataFrame, lt::Function, by::Function, rev::Bool, order::Ordering) =
    DFPerm(Order.ord(lt, by, rev, order), df)

######
## Case 1b: lt, by, rev, and order are Arrays
######
function ordering(df::AbstractDataFrame,
                  lt::AbstractVector{S}, by::AbstractVector{T},
                  rev::AbstractVector{Bool}, order::AbstractVector) where {S<:Function, T<:Function}
    if !(length(lt) == length(by) == length(rev) == length(order) == size(df,2))
        throw(ArgumentError("Orderings must be specified for all DataFrame columns"))
    end
    DFPerm([Order.ord(_lt, _by, _rev, _order) for (_lt, _by, _rev, _order) in zip(lt, by, rev, order)], df)
end

################
## Case 2:  Return a regular permutation when there's only one column
################
## Case 2a: The column is given directly
######
ordering(df::AbstractDataFrame, col::ColumnIndex, lt::Function, by::Function, rev::Bool, order::Ordering) =
    Perm(Order.ord(lt, by, rev, order), df[col])

######
## Case 2b: The column is given as a UserColOrdering
######
ordering(df::AbstractDataFrame, col_ord::UserColOrdering, lt::Function, by::Function, rev::Bool, order::Ordering) =
    Perm(ordering(col_ord, lt, by, rev, order), df[col_ord.col])

################
## Case 3:  General case: cols is an iterable of a combination of ColumnIndexes and UserColOrderings
################
## Case 3a: None of lt, by, rev, or order is an Array
######
function ordering(df::AbstractDataFrame, cols::AbstractVector, lt::Function, by::Function, rev::Bool, order::Ordering)

    if length(cols) == 0
        return ordering(df, lt, by, rev, order)
    end

    if length(cols) == 1
        return ordering(df, cols[1], lt, by, rev, order)
    end

    # Collect per-column ordering info

    ords = Ordering[]
    newcols = Int[]

    for col in cols
        push!(ords, ordering(col, lt, by, rev, order))
        push!(newcols, index(df)[(_getcol(col))])
    end

    # Simplify ordering when all orderings are the same
    if all([ords[i] == ords[1] for i = 2:length(ords)])
        return DFPerm(ords[1], df[newcols])
    end

    return DFPerm(ords, df[newcols])
end

######
# Case 3b: cols, lt, by, rev, and order are all arrays
######
function ordering(df::AbstractDataFrame, cols::AbstractVector,
                  lt::AbstractVector{S}, by::AbstractVector{T},
                  rev::AbstractVector{Bool}, order::AbstractVector) where {S<:Function, T<:Function}

    if !(length(lt) == length(by) == length(rev) == length(order))
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

    for i in 1:length(cols)
        push!(ords, ordering(cols[i], lt[i], by[i], rev[i], order[i]))
        push!(newcols, index(df)[(_getcol(cols[i]))])
    end

    # Simplify ordering when all orderings are the same
    if all([ords[i] == ords[1] for i = 2:length(ords)])
        return DFPerm(ords[1], df[newcols])
    end

    return DFPerm(ords, df[newcols])
end

######
## At least one of lt, by, rev, or order is an array or tuple, so expand all to arrays
######
function ordering(df::AbstractDataFrame, cols::AbstractVector, lt, by, rev, order)
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

#### Convert cols from tuple to Array, if necessary
ordering(df::AbstractDataFrame, cols::Tuple, args...) = ordering(df, [cols...], args...)


###########################
# Default sorting algorithm
###########################

# TimSort is fast for data with structure, but only if the DataFrame is large enough
# TODO: 8192 is informed but somewhat arbitrary

Sort.defalg(df::AbstractDataFrame) = size(df, 1) < 8192 ? Sort.MergeSort : SortingAlgorithms.TimSort

# For DataFrames, we can choose the algorithm based on the column type and requested ordering
function Sort.defalg(df::AbstractDataFrame, ::Type{T}, o::Ordering) where T<:Real
    # If we're sorting a single numerical column in forward or reverse,
    # RadixSort will generally be the fastest stable sort
    if isbits(T) && sizeof(T) <= 8 && (o==Order.Forward || o==Order.Reverse)
        SortingAlgorithms.RadixSort
    else
        Sort.defalg(df)
    end
end
Sort.defalg(df::AbstractDataFrame,        ::Type,            o::Ordering) = Sort.defalg(df)
Sort.defalg(df::AbstractDataFrame, col    ::ColumnIndex,     o::Ordering) = Sort.defalg(df, eltype(df[col]), o)
Sort.defalg(df::AbstractDataFrame, col_ord::UserColOrdering, o::Ordering) = Sort.defalg(df, col_ord.col, o)
Sort.defalg(df::AbstractDataFrame, cols,                     o::Ordering) = Sort.defalg(df)

function Sort.defalg(df::AbstractDataFrame, o::Ordering; alg=nothing, cols=[])
    alg != nothing && return alg
    Sort.defalg(df, cols, o)
end

########################
## Actual sort functions
########################

Base.issorted(df::AbstractDataFrame; cols=Any[], lt=isless, by=identity, rev=false, order=Forward) =
    issorted(eachrow(df), ordering(df, cols, lt, by, rev, order))

# sort and sortperm functions

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


#
# expanded from: include("dataframe/sort.jl")
#

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

        copy!(pp,p)
        Base.permute!!(col, pp)
    end
    df
end



#
# expanded from: include("deprecated.jl")
#

import Base: @deprecate

function DataFrame(columns::AbstractVector)
    Base.depwarn("calling vector of vectors constructor without passing column names is deprecated", :DataFrame)
    DataFrame(columns, gennames(length(columns)))
end

@deprecate by(d::AbstractDataFrame, cols, s::Vector{Symbol}) aggregate(d, cols, map(eval, s))
@deprecate by(d::AbstractDataFrame, cols, s::Symbol) aggregate(d, cols, eval(s))

@deprecate nullable!(df::AbstractDataFrame, col::ColumnIndex) allowmissing!(df, col)
@deprecate nullable!(df::AbstractDataFrame, cols::Vector{<:ColumnIndex}) allowmissing!(df, cols)
@deprecate nullable!(colnames::Array{Symbol,1}, df::AbstractDataFrame) allowmissing!(df, colnames)
@deprecate nullable!(colnums::Array{Int,1}, df::AbstractDataFrame) allowmissing!(df, colnums)

import Base: keys, values, insert!
@deprecate keys(df::AbstractDataFrame) names(df)
@deprecate values(df::AbstractDataFrame) DataFrames.columns(df)
@deprecate insert!(df::DataFrame, df2::AbstractDataFrame) merge!(df, df2)

@deprecate pool categorical
@deprecate pool! categorical!

@deprecate complete_cases! dropmissing!
@deprecate complete_cases completecases

@deprecate sub(df::AbstractDataFrame, rows) view(df, rows)


## write.table
# using CodecZlib, TranscodingStreams

export writetable
"""
Write data to a tabular-file format (CSV, TSV, ...)

```julia
writetable(filename, df, [keyword options])
```

### Arguments

* `filename::AbstractString` : the filename to be created
* `df::AbstractDataFrame` : the AbstractDataFrame to be written

### Keyword Arguments

* `separator::Char` -- The separator character that you would like to use. Defaults to the output of `getseparator(filename)`, which uses commas for files that end in `.csv`, tabs for files that end in `.tsv` and a single space for files that end in `.wsv`.
* `quotemark::Char` -- The character used to delimit string fields. Defaults to `'"'`.
* `header::Bool` -- Should the file contain a header that specifies the column names from `df`. Defaults to `true`.
* `nastring::AbstractString` -- What to write in place of missing data. Defaults to `"NA"`.

### Result

* `::DataFrame`

### Examples

```julia
df = DataFrame(A = 1:10)
writetable("output.csv", df)
writetable("output.dat", df, separator = ',', header = false)
writetable("output.dat", df, quotemark = '\'', separator = ',')
writetable("output.dat", df, header = false)
```
"""
function writetable(filename::AbstractString,
                    df::AbstractDataFrame;
                    header::Bool = true,
                    separator::Char = getseparator(filename),
                    quotemark::Char = '"',
                    nastring::AbstractString = "NA",
                    append::Bool = false)
    Base.depwarn("writetable is deprecated, use CSV.write from the CSV package instead",
                 :writetable)

    if endswith(filename, ".bz") || endswith(filename, ".bz2")
        throw(ArgumentError("BZip2 compression not yet implemented"))
    end

    if append && isfile(filename) && filesize(filename) > 0
        file_df = readtable(filename, header = false, nrows = 1)

        # Check if number of columns matches
        if size(file_df, 2) != size(df, 2)
            throw(DimensionMismatch("Number of columns differ between file and DataFrame"))
        end

        # When 'append'-ing to a nonempty file,
        # 'header' triggers a check for matching colnames
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


## read.table

struct ParsedCSV
    bytes::Vector{UInt8} # Raw bytes from CSV file
    bounds::Vector{Int}  # Right field boundary indices
    lines::Vector{Int}   # Line break indices
    quoted::BitVector    # Was field quoted in text
end

struct ParseOptions{S <: String, T <: String}
    header::Bool
    separator::Char
    quotemarks::Vector{Char}
    decimal::Char
    nastrings::Vector{S}
    truestrings::Vector{T}
    falsestrings::Vector{T}
    makefactors::Bool
    names::Vector{Symbol}
    eltypes::Vector
    allowcomments::Bool
    commentmark::Char
    ignorepadding::Bool
    skipstart::Int
    skiprows::AbstractVector{Int}
    skipblanks::Bool
    encoding::Symbol
    allowescapes::Bool
    normalizenames::Bool
end

# Dispatch on values of ParseOptions to avoid running
#   unused checks for every byte read
struct ParseType{ALLOWCOMMENTS, SKIPBLANKS, ALLOWESCAPES, SPC_SEP} end
ParseType(o::ParseOptions) = ParseType{o.allowcomments, o.skipblanks, o.allowescapes, o.separator == ' '}()

macro read_peek_eof(io, nextchr)
    io = esc(io)
    nextchr = esc(nextchr)
    quote
        nextnext = eof($io) ? 0xff : read($io, UInt8)
        $nextchr, nextnext, nextnext == 0xff
    end
end

macro skip_within_eol(io, chr, nextchr, endf)
    io = esc(io)
    chr = esc(chr)
    nextchr = esc(nextchr)
    endf = esc(endf)
    quote
        if $chr == UInt32('\r') && $nextchr == UInt32('\n')
            $chr, $nextchr, $endf = @read_peek_eof($io, $nextchr)
        end
    end
end

macro skip_to_eol(io, chr, nextchr, endf)
    io = esc(io)
    chr = esc(chr)
    nextchr = esc(nextchr)
    endf = esc(endf)
    quote
        while !$endf && !@atnewline($chr, $nextchr)
            $chr, $nextchr, $endf = @read_peek_eof($io, $nextchr)
        end
        @skip_within_eol($io, $chr, $nextchr, $endf)
    end
end

macro atnewline(chr, nextchr)
    chr = esc(chr)
    nextchr = esc(nextchr)
    quote
        $chr == UInt32('\n') || $chr == UInt32('\r')
    end
end

macro atblankline(chr, nextchr)
    chr = esc(chr)
    nextchr = esc(nextchr)
    quote
        ($chr == UInt32('\n') || $chr == UInt32('\r')) &&
        ($nextchr == UInt32('\n') || $nextchr == UInt32('\r'))
    end
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
            elseif $nextchr == UInt32('b')
                '\b'
            elseif $nextchr == UInt32('f')
                '\f'
            elseif $nextchr == UInt32('v')
                '\v'
            elseif $nextchr == UInt32('\\')
                '\\'
            else
                msg = @sprintf("Invalid escape character '%s%s' encountered",
                               $chr,
                               $nextchr)
                error(msg)
            end
        else
            msg = @sprintf("Invalid escape character '%s%s' encountered",
                           $chr,
                           $nextchr)
            error(msg)
        end
    end
end

macro isspace(byte)
    byte = esc(byte)
    quote
        0x09 <= $byte <= 0x0d || $byte == 0x20
    end
end

# This trick is ugly, but is ~33% faster than push!() for large arrays
macro push(count, a, val, l)
    count = esc(count) # Number of items in array
    a = esc(a)         # Array to update
    val = esc(val)     # Value to insert
    l = esc(l)         # Length of array
    quote
        $count += 1
        if $l < $count
            $l *= 2
            resize!($a, $l)
        end
        $a[$count] = $val
    end
end

function getseparator(filename::AbstractString)
    m = match(r"\.(\w+)(\.(gz|bz|bz2))?$", filename)
    ext = isa(m, RegexMatch) ? m.captures[1] : ""
    if ext == "csv"
        return ','
    elseif ext == "tsv"
        return '\t'
    elseif ext == "wsv"
        return ' '
    else
        return ','
    end
end

tf = (true, false)
for allowcomments in tf, skipblanks in tf, allowescapes in tf, wsv in tf
    dtype = ParseType{allowcomments, skipblanks, allowescapes, wsv}
    @eval begin
        # Read CSV file's rows into buffer while storing field boundary information
        # TODO: Experiment with mmaping input
        function readnrows!(p::ParsedCSV,
                            io::IO,
                            nrows::Integer,
                            o::ParseOptions,
                            dispatcher::$(dtype),
                            firstchr::UInt8=0xff)
            # TODO: Use better variable names
            # Information about parse results
            n_bytes = 0
            n_bounds = 0
            n_lines = 0
            n_fields = 1
            l_bytes = length(p.bytes)
            l_lines = length(p.lines)
            l_bounds = length(p.bounds)
            l_quoted = length(p.quoted)

            # Current state of the parser
            in_quotes = false
            in_escape = false
            $(if allowcomments quote at_start = true end end)
            $(if wsv quote skip_white = true end end)
            chr = 0xff
            nextchr = (firstchr == 0xff && !eof(io)) ? read(io, UInt8) : firstchr
            endf = nextchr == 0xff

            # 'in' does not work if passed UInt8 and Vector{Char}
            quotemarks = convert(Vector{UInt8}, o.quotemarks)

            # Insert a dummy field bound at position 0
            @push(n_bounds, p.bounds, 0, l_bounds)
            @push(n_bytes, p.bytes, '\n', l_bytes)
            @push(n_lines, p.lines, 0, l_lines)

            # Loop over bytes from the input until we've read requested rows
            while !endf && ((nrows == -1) || (n_lines < nrows + 1))

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
                        if !in_quotes && chr == UInt32(o.commentmark)
                            @skip_to_eol(io, chr, nextchr, endf)

                            # Skip the linebreak if the comment began at the start of a line
                            if at_start
                                continue
                            end
                        end
                    end
                end)

                $(if skipblanks
                    quote
                        # Skip blank lines
                        if !in_quotes
                            while !endf && @atblankline(chr, nextchr)
                                chr, nextchr, endf = @read_peek_eof(io, nextchr)
                                @skip_within_eol(io, chr, nextchr, endf)
                            end
                        end
                    end
                end)

                $(if allowescapes
                    quote
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
                    elseif $(if wsv
                                quote chr == UInt32(' ') || chr == UInt32('\t') end
                            else
                                quote chr == UInt32(o.separator) end
                            end)
                        $(if wsv
                            quote
                                if !(nextchr in UInt32[' ', '\t', '\n', '\r']) && !skip_white
                                    @push(n_bounds, p.bounds, n_bytes, l_bounds)
                                    @push(n_bytes, p.bytes, '\n', l_bytes)
                                    @push(n_fields, p.quoted, false, l_quoted)
                                    skip_white = false
                                end
                            end
                        else
                            quote
                                @push(n_bounds, p.bounds, n_bytes, l_bounds)
                                @push(n_bytes, p.bytes, '\n', l_bytes)
                                @push(n_fields, p.quoted, false, l_quoted)
                            end
                        end)
                    # Finished reading a row
                    elseif @atnewline(chr, nextchr)
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
end

function bytestotype(::Type{N},
                     bytes::Vector{UInt8},
                     left::Integer,
                     right::Integer,
                     nastrings::Vector{T},
                     wasquoted::Bool = false,
                     truestrings::Vector{P} = P[],
                     falsestrings::Vector{P} = P[]) where {N <: Integer,
                                                           T <: String,
                                                           P <: String}
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
        if UInt32('0') <= byte <= UInt32('9')
            value += (byte - UInt8('0')) * power
            power *= 10
        else
            return value, false, false
        end
        index -= 1
        byte = bytes[index]
    end

    if byte == UInt32('-')
        return -value, left < right, false
    elseif byte == UInt32('+')
        return value, left < right, false
    elseif UInt32('0') <= byte <= UInt32('9')
        value += (byte - UInt8('0')) * power
        return value, true, false
    else
        return value, false, false
    end
end

let out = Vector{Float64}(1)
    global bytestotype
    function bytestotype(::Type{N},
                         bytes::Vector{UInt8},
                         left::Integer,
                         right::Integer,
                         nastrings::Vector{T},
                         wasquoted::Bool = false,
                         truestrings::Vector{P} = P[],
                         falsestrings::Vector{P} = P[]) where {N <: AbstractFloat,
                                                               T <: String,
                                                               P <: String}
        if left > right
            return 0.0, true, true
        end

        if bytematch(bytes, left, right, nastrings)
            return 0.0, true, true
        end

        wasparsed = ccall(:jl_substrtod,
                          Int32,
                          (Ptr{UInt8}, Csize_t, Int, Ptr{Float64}),
                          bytes,
                          convert(Csize_t, left - 1),
                          right - left + 1,
                          out) == 0

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

    if bytematch(bytes, left, right, nastrings)
        return "", true, true
    end

    return String(bytes[left:right]), true, false
end

function builddf(rows::Integer,
                 cols::Integer,
                 bytes::Integer,
                 fields::Integer,
                 p::ParsedCSV,
                 o::ParseOptions)
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
        is_bool = true

        i = 0
        while i < rows
            i += 1

            # Determine left and right boundaries of field
            left = p.bounds[(i - 1) * cols + j] + 2
            right = p.bounds[(i - 1) * cols + j + 1]
            wasquoted = p.quoted[(i - 1) * cols + j]

            # Ignore left-and-right whitespace padding
            # TODO: Debate moving this into readnrows()
            # TODO: Modify readnrows() so that '\r' and '\n'
            #       don't occur near edges
            if o.ignorepadding && !wasquoted
                while left < right && @isspace(p.bytes[left])
                    left += 1
                end
                while left <= right && @isspace(p.bytes[right])
                    right -= 1
                end
            end

            # If eltypes has been defined, use it
            if !isempty(o.eltypes)
                values[i], wasparsed, msng[i] =
                    bytestotype(o.eltypes[j],
                                p.bytes,
                                left,
                                right,
                                o.nastrings,
                                wasquoted,
                                o.truestrings,
                                o.falsestrings)

                # Don't go to guess type zone
                if wasparsed
                    continue
                else
                    error(@sprintf("Failed to parse '%s' using type '%s'",
                                   String(p.bytes[left:right]),
                                   o.eltypes[j]))
                end
            end

            # (1) Try to parse values as Int's
            if is_int
                values[i], wasparsed, msng[i] =
                  bytestotype(Int64,
                              p.bytes,
                              left,
                              right,
                              o.nastrings,
                              wasquoted,
                              o.truestrings,
                              o.falsestrings)
                if wasparsed
                    continue
                else
                    is_int = false
                    values = convert(Array{Float64}, values)
                end
            end

            # (2) Try to parse as Float64's
            if is_float
                values[i], wasparsed, msng[i] =
                  bytestotype(Float64,
                              p.bytes,
                              left,
                              right,
                              o.nastrings,
                              wasquoted,
                              o.truestrings,
                              o.falsestrings)
                if wasparsed
                    continue
                else
                    is_float = false
                    values = Vector{Bool}(rows)
                    i = 0
                    continue
                end
            end

            # (3) Try to parse as Bool's
            if is_bool
                values[i], wasparsed, msng[i] =
                  bytestotype(Bool,
                              p.bytes,
                              left,
                              right,
                              o.nastrings,
                              wasquoted,
                              o.truestrings,
                              o.falsestrings)
                if wasparsed
                    continue
                else
                    is_bool = false
                    values = Vector{String}(rows)
                    i = 0
                    continue
                end
            end

            # (4) Fallback to String
            values[i], wasparsed, msng[i] =
              bytestotype(String,
                          p.bytes,
                          left,
                          right,
                          o.nastrings,
                          wasquoted,
                          o.truestrings,
                          o.falsestrings)
        end

        vals = similar(values, Union{eltype(values), Missing})
        @inbounds for i in eachindex(vals)
            vals[i] = msng[i] ? missing : values[i]
        end
        if o.makefactors && !(is_int || is_float || is_bool)
            columns[j] = CategoricalArray{Union{eltype(values), Missing}}(vals)
        else
            columns[j] = vals
        end
    end

    if isempty(o.names)
        return DataFrame(columns, gennames(cols))
    else
        return DataFrame(columns, o.names)
    end
end

function parsenames!(names::Vector{Symbol},
                     ignorepadding::Bool,
                     bytes::Vector{UInt8},
                     bounds::Vector{Int},
                     quoted::BitVector,
                     fields::Int,
                     normalizenames::Bool)
    if fields == 0
        error("Header line was empty")
    end

    resize!(names, fields)

    for j in 1:fields
        left = bounds[j] + 2
        right = bounds[j + 1]

        if ignorepadding && !quoted[j]
            while left < right && @isspace(bytes[left])
                left += 1
            end
            while left <= right && @isspace(bytes[right])
                right -= 1
            end
        end

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
        f = 0
        while t <= n && p.bounds[t] < bound
            f += 1
            t += 1
        end
        lengths[i] = f
    end
    m = median(lengths)
    corruptrows = find(lengths .!= m)
    l = corruptrows[1]
    error(@sprintf("Saw %d rows, %d columns and %d fields\n * Line %d has %d columns\n",
                   rows,
                   cols,
                   fields,
                   l,
                   lengths[l] + 1))
end

function readtable!(p::ParsedCSV,
                    io::IO,
                    nrows::Integer,
                    o::ParseOptions)

    chr, nextchr = 0xff, 0xff

    skipped_lines = 0

    # Skip lines at the start
    if o.skipstart != 0
        while skipped_lines < o.skipstart
            chr, nextchr, endf = @read_peek_eof(io, nextchr)
            @skip_to_eol(io, chr, nextchr, endf)
            skipped_lines += 1
        end
    else
        chr, nextchr, endf = @read_peek_eof(io, nextchr)
    end

    if o.allowcomments || o.skipblanks
        while true
            if o.allowcomments && nextchr == UInt32(o.commentmark)
                chr, nextchr, endf = @read_peek_eof(io, nextchr)
                @skip_to_eol(io, chr, nextchr, endf)
            elseif o.skipblanks && @atnewline(nextchr, nextchr)
                chr, nextchr, endf = @read_peek_eof(io, nextchr)
                @skip_within_eol(io, chr, nextchr, endf)
            else
                break
            end
            skipped_lines += 1
        end
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

    # Sanity checks
    bytes != 0 || error("Failed to read any bytes.")
    rows != 0 || error("Failed to read any rows.")
    fields != 0 || error("Failed to read any fields.")

    # Determine the number of columns
    cols = fld(fields, rows)

    # if the file is empty but has a header then fields, cols and rows will not be computed correctly
    if length(o.names) != cols && cols == 1 && rows == 1 && fields == 1 && bytes == 2
        fields = 0
        rows = 0
        cols = length(o.names)
    end

    # Confirm that the number of columns is consistent across rows
    if fields != rows * cols
        findcorruption(rows, cols, fields, p)
    end

    # Parse contents of a buffer into a DataFrame
    df = builddf(rows, cols, bytes, fields, p, o)

    # Return the final DataFrame
    return df
end

function readtable(io::IO,
                   nbytes::Integer = 1;
                   header::Bool = true,
                   separator::Char = ',',
                   quotemark::Vector{Char} = ['"'],
                   decimal::Char = '.',
                   nastrings::Vector = ["", "NA"],
                   truestrings::Vector = ["T", "t", "TRUE", "true"],
                   falsestrings::Vector = ["F", "f", "FALSE", "false"],
                   makefactors::Bool = false,
                   nrows::Integer = -1,
                   names::Vector = Symbol[],
                   eltypes::Vector = [],
                   allowcomments::Bool = false,
                   commentmark::Char = '#',
                   ignorepadding::Bool = true,
                   skipstart::Integer = 0,
                   skiprows::AbstractVector{Int} = Int[],
                   skipblanks::Bool = true,
                   encoding::Symbol = :utf8,
                   allowescapes::Bool = false,
                   normalizenames::Bool = true)
    if encoding != :utf8
        throw(ArgumentError("Argument 'encoding' only supports ':utf8' currently."))
    elseif !isempty(skiprows)
        throw(ArgumentError("Argument 'skiprows' is not yet supported."))
    elseif decimal != '.'
        throw(ArgumentError("Argument 'decimal' is not yet supported."))
    end

    if !isempty(eltypes)
        for j in 1:length(eltypes)
            if !(eltypes[j] in [String, Bool, Float64, Int64])
                throw(ArgumentError("Invalid eltype $(eltypes[j]) encountered.\nValid eltypes: $(String), Bool, Float64 or Int64"))
            end
        end
    end

    # Allocate buffers for storing metadata
    p = ParsedCSV(Vector{UInt8}(nbytes),
                   Vector{Int}(1),
                   Vector{Int}(1),
                  BitArray(1))

    # Set parsing options
    o = ParseOptions(header, separator, quotemark, decimal,
                     nastrings, truestrings, falsestrings,
                     makefactors, names, eltypes,
                     allowcomments, commentmark, ignorepadding,
                     skipstart, skiprows, skipblanks, encoding,
                     allowescapes, normalizenames)

    # Use the IO stream method for readtable()
    df = readtable!(p, io, nrows, o)

    # Close the IO stream
    close(io)

    # Return the resulting DataFrame
    return df
end

export readtable

"""
Read data from a tabular-file format (CSV, TSV, ...)

```julia
readtable(filename, [keyword options])
```

### Arguments

* `filename::AbstractString` : the filename to be read

### Keyword Arguments

*   `header::Bool` -- Use the information from the file's header line to determine column names. Defaults to `true`.
*   `separator::Char` -- Assume that fields are split by the `separator` character. If not specified, it will be guessed from the filename: `.csv` defaults to `','`, `.tsv` defaults to `'\t'`, `.wsv` defaults to `' '`.
*   `quotemark::Vector{Char}` -- Assume that fields contained inside of two `quotemark` characters are quoted, which disables processing of separators and linebreaks. Set to `Char[]` to disable this feature and slightly improve performance. Defaults to `['"']`.
*   `decimal::Char` -- Assume that the decimal place in numbers is written using the `decimal` character. Defaults to `'.'`.
*   `nastrings::Vector{String}` -- Translate any of the strings into this vector into a `missing`. Defaults to `["", "NA"]`.
*   `truestrings::Vector{String}` -- Translate any of the strings into this vector into a Boolean `true`. Defaults to `["T", "t", "TRUE", "true"]`.
*   `falsestrings::Vector{String}` -- Translate any of the strings into this vector into a Boolean `false`. Defaults to `["F", "f", "FALSE", "false"]`.
*   `makefactors::Bool` -- Convert string columns into `PooledDataVector`'s for use as factors. Defaults to `false`.
*   `nrows::Int` -- Read only `nrows` from the file. Defaults to `-1`, which indicates that the entire file should be read.
*   `names::Vector{Symbol}` -- Use the values in this array as the names for all columns instead of or in lieu of the names in the file's header. Defaults to `[]`, which indicates that the header should be used if present or that numeric names should be invented if there is no header.
*   `eltypes::Vector` -- Specify the types of all columns. Defaults to `[]`.
*   `allowcomments::Bool` -- Ignore all text inside comments. Defaults to `false`.
*   `commentmark::Char` -- Specify the character that starts comments. Defaults to `'#'`.
*   `ignorepadding::Bool` -- Ignore all whitespace on left and right sides of a field. Defaults to `true`.
*   `skipstart::Int` -- Specify the number of initial rows to skip. Defaults to `0`.
*   `skiprows::Vector{Int}` -- Specify the indices of lines in the input to ignore. Defaults to `[]`.
*   `skipblanks::Bool` -- Skip any blank lines in input. Defaults to `true`.
*   `encoding::Symbol` -- Specify the file's encoding as either `:utf8` or `:latin1`. Defaults to `:utf8`.
*   `normalizenames::Bool` -- Ensure that column names are valid Julia identifiers. For instance this renames a column named `"a b"` to `"a_b"` which can then be accessed with `:a_b` instead of `Symbol("a b")`. Defaults to `true`.

### Result

* `::DataFrame`

### Examples

```julia
df = readtable("data.csv")
df = readtable("data.tsv")
df = readtable("data.wsv")
df = readtable("data.txt", separator = '\t')
df = readtable("data.txt", header = false)
```
"""
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
                   eltypes::Vector = [],
                   allowcomments::Bool = false,
                   commentmark::Char = '#',
                   ignorepadding::Bool = true,
                   skipstart::Integer = 0,
                   skiprows::AbstractVector{Int} = Int[],
                   skipblanks::Bool = true,
                   encoding::Symbol = :utf8,
                   allowescapes::Bool = false,
                   normalizenames::Bool = true)
    Base.depwarn("readtable is deprecated, use CSV.read from the CSV package instead",
                 :readtable)

    _r(io) = readtable(io,
                       nbytes,
                       header = header,
                       separator = separator,
                       quotemark = quotemark,
                       decimal = decimal,
                       nastrings = nastrings,
                       truestrings = truestrings,
                       falsestrings = falsestrings,
                       makefactors = makefactors,
                       nrows = nrows,
                       names = names,
                       eltypes = eltypes,
                       allowcomments = allowcomments,
                       commentmark = commentmark,
                       ignorepadding = ignorepadding,
                       skipstart = skipstart,
                       skiprows = skiprows,
                       skipblanks = skipblanks,
                       encoding = encoding,
                       allowescapes = allowescapes,
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

"""
    inlinetable(s[, flags]; args...)

A helper function to process strings as tabular data for non-standard string
literals. Parses the string `s` containing delimiter-separated tabular data
(by default, comma-separated values) using `readtable`. The optional `flags`
argument contains a list of flag characters, which, if present, are equivalent
to supplying named arguments to `readtable` as follows:

- `f`: `makefactors=true`, convert string columns to `PooledData` columns
- `c`: `allowcomments=true`, ignore lines beginning with `#`
- `H`: `header=false`, do not interpret the first line as column names
"""
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

"""
    @csv_str(s[, flags])
    csv"[data]"fcH

Construct a `DataFrame` from a non-standard string literal containing comma-
separated values (CSV) using `readtable`, just as if it were being loaded from
an external file. The suffix flags `f`, `c`, and `H` are optional. If present,
they are equivalent to supplying named arguments to `readtable` as follows:

* `f`: `makefactors=true`, convert string columns to `CategoricalArray` columns
* `c`: `allowcomments=true`, ignore lines beginning with `#`
* `H`: `header=false`, do not interpret the first line as column names

# Example
```jldoctest
julia> df = csv\"""
           name,  age, squidPerWeek
           Alice,  36,         3.14
           Bob,    24,         0
           Carol,  58,         2.71
           Eve,    49,         7.77
           \"""
4×3 DataFrames.DataFrame
│ Row │ name    │ age │ squidPerWeek │
├─────┼─────────┼─────┼──────────────┤
│ 1   │ "Alice" │ 36  │ 3.14         │
│ 2   │ "Bob"   │ 24  │ 0.0          │
│ 3   │ "Carol" │ 58  │ 2.71         │
│ 4   │ "Eve"   │ 49  │ 7.77         │
```
"""
macro csv_str(s, flags...)
    Base.depwarn("@csv_str and the csv\"\"\" syntax are deprecated. " *
                 "Use CSV.read(IOBuffer(...)) from the CSV package instead.",
                 :csv_str)
    inlinetable(s, flags...; separator=',')
end

"""
    @csv2_str(s[, flags])
    csv2"[data]"fcH

Construct a `DataFrame` from a non-standard string literal containing
semicolon-separated values using `readtable`, with comma acting as the decimal
character, just as if it were being loaded from an external file. The suffix
flags `f`, `c`, and `H` are optional. If present, they are equivalent to
supplying named arguments to `readtable` as follows:

* `f`: `makefactors=true`, convert string columns to `CategoricalArray` columns
* `c`: `allowcomments=true`, ignore lines beginning with `#`
* `H`: `header=false`, do not interpret the first line as column names

# Example
```jldoctest
julia> df = csv2\"""
           name;  age; squidPerWeek
           Alice;  36;         3,14
           Bob;    24;         0
           Carol;  58;         2,71
           Eve;    49;         7,77
           \"""
4×3 DataFrames.DataFrame
│ Row │ name    │ age │ squidPerWeek │
├─────┼─────────┼─────┼──────────────┤
│ 1   │ "Alice" │ 36  │ 3.14         │
│ 2   │ "Bob"   │ 24  │ 0.0          │
│ 3   │ "Carol" │ 58  │ 2.71         │
│ 4   │ "Eve"   │ 49  │ 7.77         │
```
"""
macro csv2_str(s, flags...)
    Base.depwarn("@csv2_str and the csv2\"\"\" syntax are deprecated. " *
                 "Use CSV.read(IOBuffer(...)) from the CSV package instead.",
                 :csv2_str)
    inlinetable(s, flags...; separator=';', decimal=',')
end

"""
    @wsv_str(s[, flags])
    wsv"[data]"fcH

Construct a `DataFrame` from a non-standard string literal containing
whitespace-separated values (WSV) using `readtable`, just as if it were being
loaded from an external file. The suffix flags `f`, `c`, and `H` are optional.
If present, they are equivalent to supplying named arguments to `readtable` as
follows:

* `f`: `makefactors=true`, convert string columns to `CategoricalArray` columns
* `c`: `allowcomments=true`, ignore lines beginning with `#`
* `H`: `header=false`, do not interpret the first line as column names

# Example
```jldoctest
julia> df = wsv\"""
           name  age squidPerWeek
           Alice  36         3.14
           Bob    24         0
           Carol  58         2.71
           Eve    49         7.77
           \"""
4×3 DataFrames.DataFrame
│ Row │ name    │ age │ squidPerWeek │
├─────┼─────────┼─────┼──────────────┤
│ 1   │ "Alice" │ 36  │ 3.14         │
│ 2   │ "Bob"   │ 24  │ 0.0          │
│ 3   │ "Carol" │ 58  │ 2.71         │
│ 4   │ "Eve"   │ 49  │ 7.77         │
```
"""
macro wsv_str(s, flags...)
    Base.depwarn("@wsv_str and the wsv\"\"\" syntax are deprecated. " *
                 "Use CSV.read(IOBuffer(...)) from the CSV package instead.",
                 :wsv_str)
    inlinetable(s, flags...; separator=' ')
end

"""
    @tsv_str(s[, flags])
    tsv"[data]"fcH

Construct a `DataFrame` from a non-standard string literal containing tab-
separated values (TSV) using `readtable`, just as if it were being loaded from
an external file. The suffix flags `f`, `c`, and `H` are optional. If present,
they are equivalent to supplying named arguments to `readtable` as follows:

* `f`: `makefactors=true`, convert string columns to `CategoricalArray` columns
* `c`: `allowcomments=true`, ignore lines beginning with `#`
* `H`: `header=false`, do not interpret the first line as column names

# Example
```jldoctest
julia> df = tsv\"""
           name\tage\tsquidPerWeek
           Alice\t36\t3.14
           Bob\t24\t0
           Carol\t58\t2.71
           Eve\t49\t7.77
           \"""
4×3 DataFrames.DataFrame
│ Row │ name    │ age │ squidPerWeek │
├─────┼─────────┼─────┼──────────────┤
│ 1   │ "Alice" │ 36  │ 3.14         │
│ 2   │ "Bob"   │ 24  │ 0.0          │
│ 3   │ "Carol" │ 58  │ 2.71         │
│ 4   │ "Eve"   │ 49  │ 7.77         │
```
"""
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

# Pipeline
import Base: |>
@deprecate (|>)(gd::GroupedDataFrame, fs::Function) aggregate(gd, fs)
@deprecate (|>)(gd::GroupedDataFrame, fs::Vector{T}) where {T<:Function} aggregate(gd, fs)
@deprecate colwise(f) x -> colwise(f, x)
@deprecate groupby(cols::Vector{T}; sort::Bool = false, skipmissing::Bool = false) where {T} x -> groupby(x, cols, sort = sort, skipmissing = skipmissing)
@deprecate groupby(cols; sort::Bool = false, skipmissing::Bool = false) x -> groupby(x, cols, sort = sort, skipmissing = skipmissing)

function Base.getindex(x::AbstractIndex, idx::Bool)
    Base.depwarn("Indexing with Bool values is deprecated except for Vector{Bool}", :getindex)
    1
end

function Base.getindex(x::AbstractIndex, idx::Real)
    Base.depwarn("Indexing with values that are not Integer is deprecated", :getindex)
    Int(idx)
end

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
