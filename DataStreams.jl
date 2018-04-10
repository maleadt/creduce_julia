__precompile__(true)
module DataStreams

export Data

module Data

using Missings, WeakRefStrings

if !isdefined(Base, :AbstractRange)
    const AbstractRange = Range
end
macro uninit(expr)
    if !isdefined(Base, :uninitialized)
        splice!(expr.args, 2)
    end
    return esc(expr)
end

# Data.Schema
"""
A `Data.Schema` describes a tabular dataset, i.e. a set of named, typed columns with records as rows

`Data.Schema` allow `Data.Source` and `Data.Sink` to talk to each other and prepare to provide/receive data through streaming.
`Data.Schema` provides the following accessible properties:

 * `Data.header(schema)` to return the header/column names in a `Data.Schema`
 * `Data.types(schema)` to return the column types in a `Data.Schema`; `Union{T, Missing}` indicates columns that may contain missing data (`missing` values)
 * `Data.size(schema)` to return the (# of rows, # of columns) in a `Data.Schema`; note that # of rows may be `nothing`, meaning unknown

`Data.Schema` has the following constructors:

 * `Data.Schema()`: create an "emtpy" schema with no rows, no columns, and no column names
 * `Data.Schema(types[, header, rows, meta::Dict])`: column element types are provided as a tuple or vector; column names provided as an iterable; # of rows can be an Int or `missing` to indicate unknown # of rows

`Data.Schema` are indexable via column names to get the number of that column in the `Data.Schema`

```julia
julia> sch = Data.Schema([Int], ["column1"], 10)
Data.Schema:
rows: 10	cols: 1
Columns:
 "column1"  Int64

julia> sch["column1"]
1
```

**Developer note**: the full type definition is `Data.Schema{R, T}` where the `R` type parameter will be `true` or `false`, indicating
whether the # of rows are known (i.e not `missing`), respectively. The `T` type parameter is a `Tuple{A, B, ...}` representing the column element types
in the `Data.Schema`. Both of these type parameters provide valuable information that may be useful when constructing `Sink`s or streaming data.
"""
mutable struct Schema{R, T}
    # types::T               # Julia types of columns
    header::Vector{String}   # column names
    rows::Union{Int, Missing}# number of rows in the dataset
    cols::Int                # number of columns in a dataset
    metadata::Dict           # for any other metadata we'd like to keep around (not used for '==' operation)
    index::Dict{String, Int} # maps column names as Strings to their index # in `header` and `types`
end

function Schema(types=(), header=["Column$i" for i = 1:length(types)], rows::Union{Integer,Missing}=0, metadata::Dict=Dict())
    !ismissing(rows) && rows < 0 && throw(ArgumentError("Invalid # of rows for Data.Schema; use `nothing` to indicate an unknown # of rows"))
    types2 = Tuple(types)
    header2 = String[string(x) for x in header]
    cols = length(header2)
    cols != length(types2) && throw(ArgumentError("length(header): $(length(header2)) must == length(types): $(length(types2))"))
    return Schema{!ismissing(rows), Tuple{types2...}}(header2, rows, cols, metadata, Dict(n=>i for (i, n) in enumerate(header2)))
end
Schema(types, rows::Union{Integer,Missing}, metadata::Dict=Dict()) = Schema(types, ["Column$i" for i = 1:length(types)], rows, metadata)

header(sch::Schema) = sch.header
types(sch::Schema{R, T}) where {R, T} = Tuple(T.parameters)
metadata(sch::Schema) = sch.metadata
Base.size(sch::Schema) = (sch.rows, sch.cols)
Base.size(sch::Schema, i::Int) = ifelse(i == 1, sch.rows, ifelse(i == 2, sch.cols, 0))

setrows!(source, rows) = isdefined(source, :schema) ? (source.schema.rows = rows; nothing) : nothing
setrows!(source::Array, rows) = nothing
Base.getindex(sch::Schema, col::String) = sch.index[col]

function Base.show(io::IO, schema::Schema)
    println(io, "Data.Schema:")
    println(io, "rows: $(schema.rows)  cols: $(schema.cols)")
    if schema.cols > 0
        println(io, "Columns:")
        Base.print_matrix(io, hcat(schema.header, collect(types(schema))))
    end
end

function transform(sch::Data.Schema{R, T}, transforms::Dict{Int, <:Base.Callable}, weakref) where {R, T}
    types = Data.types(sch)
    transforms2 = ((get(transforms, x, identity) for x = 1:length(types))...,)
    newtypes = ((Core.Inference.return_type(transforms2[x], (types[x],)) for x = 1:length(types))...,)
    if !weakref
        newtypes = map(x->x >: Missing ? ifelse(Missings.T(x) <: WeakRefString, Union{String, Missing}, x) : ifelse(x <: WeakRefString, String, x), newtypes)
    end
    return Schema(newtypes, Data.header(sch), size(sch, 1), sch.metadata), transforms2
end
transform(sch::Data.Schema, transforms::Dict{String, <:Base.Callable}, s) = transform(sch, Dict{Int, Base.Callable}(sch[x]=>f for (x, f) in transforms), s)

# Data.StreamTypes
abstract type StreamType end
struct Field  <: StreamType end
struct Row    <: StreamType end
struct Column <: StreamType end

# Data.Source Interface
abstract type Source end

# Required methods
"""
`Data.schema(s::Source) => Data.Schema`

Return the `Data.Schema` of a source, which describes the # of rows & columns, as well as the column types of a dataset.
Some sources like `CSV.Source` or `SQLite.Source` store the `Data.Schema` directly in the type, whereas
others like `DataFrame` compute the schema on the fly.

The `Data.Schema` of a source is used in various ways during the streaming process:

- The # of rows and if they are known are used to generate the inner streaming loop
- The # of columns determine if the innermost streaming loop can be unrolled automatically or not
- The types of the columns are used in loop unrolling to generate efficient and type-stable streaming

See `?Data.Schema` for more details on how to work with the type.
"""
function schema end
"""
`Data.isdone(source, row, col) => Bool`

Checks whether a source can stream additional fields/columns for a desired
`row` and `col` intersection. Used during the streaming process, especially for sources
that have an unknown # of rows, to detect when a source has been exhausted of data.

Data.Source types must at least implement:

`Data.isdone(source::S, row::Int, col::Int)`

If more convenient/performant, they can also implement:

`Data.isdone(source::S, row::Int, col::Int, rows::Union{Int, Missing}, cols::Int)`

where `rows` and `cols` are the size of the `source`'s schema when streaming.

A simple example of how a DataFrame implements this is:
```julia
Data.isdone(df::DataFrame, row, col, rows, cols) = row > rows || col > cols
```
"""
function isdone end
isdone(source, row, col, rows, cols) = isdone(source, row, col)

"""
`Data.streamtype{T<:Data.Source, S<:Data.StreamType}(::Type{T}, ::Type{S}) => Bool`

Indicates whether the source `T` supports streaming of type `S`. To be overloaded by individual sources according to supported `Data.StreamType`s.
This is used in the streaming process to determine the compatability of streaming from a specific source to a specific sink.
It also helps in determining the preferred streaming method, when matched up with the results of `Data.streamtypes(s::Sink)`.

For example, if `MyPkg.Source` supported `Data.Field` streaming, I would define:

```julia
Data.streamtype(::Type{MyPkg.Source}, ::Type{Data.Field}) = true
```

and/or for `Data.Column` streaming:

```julia
Data.streamtype(::Type{MyPkg.Source}, ::Type{Data.Column}) = true
```
"""
function streamtype end

"""
`Data.reset!(source)`

Resets a source into a state that allows streaming its data again.
For example, for `CSV.Source`, the internal buffer is "seek"ed back to the start position of
the csv data (after the column name headers). For `SQLite.Source`, the source SQL query is re-executed.
"""
function reset! end

"""
Data.Source types must implement one of the following:

`Data.streamfrom(source, ::Type{Data.Field}, ::Type{T}, row::Int, col::Int) where {T}`

`Data.streamfrom(source, ::Type{Data.Column}, ::Type{T}, col::Int) where {T}`

Performs the actually streaming of data "out" of a data source. For `Data.Field` streaming, the single field value of type `T`
at the intersection of `row` and `col` is returned. For `Data.Column` streaming, the column # `col` with element type `T` is returned.

For `Data.Column`, a source can also implement:
```julia
Data.streamfrom(source, ::Type{Data.Field}, ::Type{T}, row::Int, col::Int) where {T}
```
where `row` indicates the # of rows that have already been streamed from the source.
"""
function streamfrom end
Data.streamfrom(source, ::Type{Data.Column}, T, row, col) = Data.streamfrom(source, Data.Column, T, col)
Data.streamfrom(source, ::Type{Data.Column}, T, r::AbstractRange, col) = Data.streamfrom(source, Data.Column, T, first(r), col)[r]

# Generic fallbacks
Data.streamtype(source, ::Type{Data.Row}) = Data.streamtype(source, Data.Field)
Data.streamtype(source, ::Type{<:StreamType}) = false
Data.reset!(source) = nothing

struct RandomAccess end
struct Sequential end

"""
`Data.accesspattern(source) => Data.RandomAccess | Data.Sequential`

returns the data access pattern for a Data.Source.

`RandomAccess` indicates that a source supports streaming data (via calls to `Data.streamfrom`) with arbitrary row/column values in any particular order.

`Sequential` indicates that the source only supports streaming data sequentially, starting w/ row 1, then accessing each column from 1:N, then row 2, and each column from 1:N again, etc.

For example, a `DataFrame` holds all data in-memory, and thus supports easy random access in any order.
A `CSV.Source` however, is streaming the contents of a file, where rows must be read sequentially, and each column sequentially within each rows.

By default, sources are assumed to have a `Sequential` access pattern.
"""
function accesspattern end

accesspattern(x) = Sequential()

const EMPTY_REFERENCE = UInt8[]
"""
Data.Source types can optionally implement

`Data.reference(x::Source) => Vector{UInt8}`

where the type retruns a `Vector{UInt8}` that represents a memory block that should be kept in reference for WeakRefStringArrays.

In many streaming situations, the minimizing of data copying/movement is ideal. Some sources can provide in-memory access to their data
in the form of a `Vector{UInt8}`, i.e. a single byte vector, that sinks can "point to" when streaming, instead of needing to always
copy all the data. In particular, the `WeakRefStrings` package provides utilities for creating "string types" that don't actually hold their
own data, but instead just "point" to data that lives elsewhere. As Strings can be some of the most expensive data structures to copy
and move around, this provides excellent performance gains in some cases when the sink is able to leverage this alternative structure.
"""
function reference end
reference(x) = EMPTY_REFERENCE

# Data.Sink Interface
"""
Represents a type that can have data streamed to it from `Data.Source`s.

To satisfy the `Data.Sink` interface, it must provide two constructors with the following signatures:

```
[Sink](sch::Data.Schema, S::Type{StreamType}, append::Bool, args...; reference::Vector{UInt8}=UInt8[], kwargs...) => [Sink]
[Sink](sink, sch::Data.Schema, S::Type{StreamType}, append::Bool; reference::Vector{UInt8}=UInt8[]) => [Sink]
```

Let's break these down, piece by piece:

- `[Sink]`: this is your sink type, i.e. `CSV.Sink`, `DataFrame`, etc. You're defining a constructor for your sink type.
- `sch::Data.Schema`: in the streaming process, the schema of a `Data.Source` is provided to the sink in order to allow the sink to "initialize" properly in order to receive data according to the format in `sch`. This might mean pre-allocating space according to the # of rows/columns in the source, managing the sink's own schema to match `sch`, etc.
- `S::Type{StreamType}`: `S` represents the type of streaming that will occur from the `Data.Source`, either `Data.Field` or `Data.Column`
- `append::Bool`: a boolean flag indicating whether the data should be appended to a sink's existing data store, or, if `false`, if the sink's data should be fully replaced by the incoming `Data.Source`'s data
- `args...`: In the 1st constructor form, `args...` represents a catchall for any additional arguments your sink may need to construct. For example, `SQLite.jl` defines `Sink(sch, S, append, db, table_name)`, meaning that the `db` and `table_name` are additional required arguments in order to properly create an `SQLite.Sink`.
- `reference::Vector{UInt8}`: if your sink defined `Data.weakrefstrings(sink::MySink) = true`, then it also needs to be able to accept the `reference` keyword argument, where a source's memory block will be passed, to be held onto appropriately by the sink when streaming WeakRefStrings. If a sink does not support streaming WeakRefStrings (the default), the sink constructor doesn't need to support any keyword arguments.
- `kwargs...`: Similar to `args...`, `kwargs...` is a catchall for any additional keyword arguments you'd like to expose for your sink constructor, typically matching supported keyword arguments that are provided through the normal sink constructor
- `sink`: in the 2nd form, an already-constructed sink is passed in as the 1st argument. This allows efficient sink re-use. This constructor needs to ensure the existing sink is modified (enlarged, shrunk, schema changes, etc) to be ready to accept the incoming source data as described by `sch`.

Now let's look at an example implementation from CSV.jl:

```julia
function CSV.Sink(fullpath::AbstractString; append::Bool=false, headers::Bool=true, colnames::Vector{String}=String[], kwargs...)
    io = IOBuffer()
    options = CSV.Options(kwargs...)
    !append && header && !isempty(colnames) && writeheaders(io, colnames, options)
    return CSV.Sink(options, io, fullpath, position(io), !append && header && !isempty(colnames), colnames, length(colnames), append)
end

function CSV.Sink(sch::Data.Schema, T, append, file::AbstractString; reference::Vector{UInt8}=UInt8[], kwargs...)
    sink = CSV.Sink(file; append=append, colnames=Data.header(sch), kwargs...)
    return sink
end

function CSV.Sink(sink, sch::Data.Schema, T, append; reference::Vector{UInt8}=UInt8[])
    sink.append = append
    sink.cols = size(sch, 2)
    !sink.header && !append && writeheaders(sink.io, Data.header(sch), sink.options, sink.quotefields)
    return sink
end
```

In this case, CSV.jl defined an initial constructor that just takes the filename with a few keyword arguments.
The two required Data.Sink constructors are then defined. The first constructs a new Sink, requiring a `file::AbstractString` argument.
We also see that `CSV.Sink` supports WeakRefString streaming by accepting a `reference` keyword argument (which is trivially implemented for CSV, since all data is simply written out to disk as text).

For the 2nd (last) constructor in the definitions above, we see the case where an existing `sink` is passed to `CSV.Sink`.
The sink updates a few of its fields (`sink.append = append`), and some logic is computed to determine if
the column headers should be written.
"""
abstract type Sink end

"""
`Data.streamtypes(::Type{[Sink]}) => Vector{StreamType}`

Returns a vector of `Data.StreamType`s that the sink is able to receive; the order of elements indicates the sink's streaming preference

For example, if my sink only supports `Data.Field` streaming, I would simply define:
```julia
Data.streamtypes(::Type{MyPkg.Sink}) = [Data.Field]
```

If, on the other hand, my sink also supported `Data.Column` streaming, and `Data.Column` streaming happend to be more efficient, I could define:
```julia
Data.streamtypes(::Type{MyPkg.Sink}) = [Data.Column, Data.Field] # put Data.Column first to indicate preference
```

A third option is a sink that operates on entire rows at a time, in which case I could define:
```julia
Data.streamtypes(::Type{MyPkg.Sink}) = [Data.Row]
```
The subsequent `Data.streamto!` method would then require the signature `Data.streamto!(sink::MyPkg.Sink, ::Type{Data.Row}, vals::NamedTuple, row, col, knownrows`
"""
function streamtypes end

"""
`Data.streamto!(sink, S::Type{StreamType}, val, row, col)`

`Data.streamto!(sink, S::Type{StreamType}, val, row, col, knownrows)`

Streams data to a sink. `S` is the type of streaming (`Data.Field`, `Data.Row`, or `Data.Column`). `val` is the value or values (single field, row as a NamedTuple, or column, respectively)
to be streamed to the sink. `row` and `col` indicate where the data should be streamed/stored.

A sink may optionally define the method that also accepts the `knownrows` argument, which will be `Val{true}` or `Val{false}`,
indicating whether the source streaming has a known # of rows or not. This can be useful for sinks that
may know how to pre-allocate space in the cases where the source can tell the # of rows, or in the case
of unknown # of rows, may need to stream the data in differently.
"""
function streamto! end
Data.streamto!(sink, S, val, row, col, knownrows) = Data.streamto!(sink, S, val, row, col)
Data.streamto!(sink, S, val, row, col) = Data.streamto!(sink, S, val, col)

# Optional methods
"""
`Data.cleanup!(sink)`

Sometimes errors occur during the streaming of data from source to sink. Some sinks may be left in
an undesired state if an error were to occur mid-streaming. `Data.cleanup!` allows a sink to "clean up"
any necessary resources in the case of a streaming error. `SQLite.jl`, for example, defines:
```julia
function Data.cleanup!(sink::SQLite.Sink)
    rollback(sink.db, sink.transaction)
    return
end
```
Since a database transaction is initiated at the start of streaming, it must be rolled back in the case of streaming error.

The default definition is: `Data.cleanup!(sink) = nothing`
"""
function cleanup! end

"""
`Data.close!(sink) => sink`

A function to "close" a sink to streaming. Some sinks require a definitive time where data can be "committed",
`Data.close!` allows a sink to perform any necessary resource management or commits to ensure all data that has been
streamed is stored appropriately. For example, the `SQLite` package defines:
```julia
function Data.close!(sink::SQLite.Sink)
    commit(sink.db, sink.transaction)
    return sink
end
```
Which commits a database transaction that was started when the sink was initially "opened".
"""
function close! end

# Generic fallbacks
cleanup!(sink) = nothing
close!(sink) = sink

"""
`Data.weakrefstrings(sink) => Bool`

If a sink is able to appropriately handle `WeakRefString` objects, it can define:
```julia
Data.weakrefstrings(::Type{[Sink]}) = true
```
to indicate that a source may stream those kinds of values to it. By default, sinks do
not support WeakRefString streaming. Supporting WeakRefStrings corresponds to accepting the
`reference` keyword argument in the required sink constructor method, see `?Data.Sink`.
"""
function weakrefstrings end
weakrefstrings(x) = false

# Data.stream!
"""
`Data.stream!(source, sink; append::Bool=false, transforms=Dict())`

`Data.stream!(source, ::Type{Sink}, args...; append::Bool=false, transforms=Dict(), kwargs...)`

Stream data from source to sink. The 1st definition assumes already constructed source & sink and takes two optional keyword arguments:

- `append::Bool=false`: whether the data from `source` should be appended to `sink`
- `transforms::Dict`: A dict with mappings between column # or name (Int or String) to a "transform" function. For `Data.Field` streaming, the transform function should be of the form `f(x::T) => y::S`, i.e. takes a single input of type `T` and returns a single value of type `S`. For `Data.Column` streaming, it should be of the form `f(x::AbstractVector{T}) => y::AbstractVector{S}`, i.e. take an AbstractVector with eltype `T` and return another AbstractVector with eltype `S`.

For the 2nd definition, the Sink type itself is passed as the 2nd argument (`::Type{Sink}`) and is constructed "on-the-fly", being passed `args...` and `kwargs...` like `Sink(args...; kwargs...)`.

While users are free to call `Data.stream!` themselves, oftentimes, packages want to provide even higher-level convenience functions.

An example of of these higher-level convenience functions are from CSV.jl:

```julia
function CSV.read(fullpath::Union{AbstractString,IO}, sink::Type=DataFrame, args...; append::Bool=false, transforms::Dict=Dict{Int,Function}(), kwargs...)
    source = CSV.Source(fullpath; kwargs...)
    sink = Data.stream!(source, sink, args...; append=append, transforms=transforms, kwargs...)
    return Data.close!(sink)
end

function CSV.read(fullpath::Union{AbstractString,IO}, sink::T; append::Bool=false, transforms::Dict=Dict{Int,Function}(), kwargs...) where T
    source = CSV.Source(fullpath; kwargs...)
    sink = Data.stream!(source, sink; append=append, transforms=transforms)
    return Data.close!(sink)
end
```
In this example, CSV.jl defines it's own high-level function for reading from a `CSV.Source`. In these examples, a `CSV.Source` is constructed using the `fullpath` argument, along w/ any extra `kwargs...`.
The sink can be provided as a type with `args...` and `kwargs...` that will be passed to its DataStreams constructor, like `Sink(sch, streamtype, append, args...; kwargs...)`; otherwise, an already-constructed
Sink can be provided directly (2nd example).

Once the `source` is constructed, the data is streamed via the call to `Data.stream(source, sink; append=append, transforms=transforms)`, with the sink being returned.

And finally, to "finish" the streaming process, `Data.close!(sink)` is closed, which returns the finalized sink. Note that `Data.stream!(source, sink)` could be called multiple times with different sources and the same sink,
most likely with `append=true` being passed, to enable the accumulation of several sources into a single sink. A single `Data.close!(sink)` method should be called to officially close or commit the final sink.

Two "builtin" Source/Sink types that are included with the DataStreams package are the `Data.Table` and `Data.RowTable` types. `Data.Table` is a NamedTuple of AbstractVectors, with column names as NamedTuple fieldnames.
This type supports both `Data.Field` and `Data.Column` streaming. `Data.RowTable` is just a Vector of NamedTuples, and as such, only supports `Data.Field` streaming.

In addition, any `Data.Source` can be iterated via the `Data.rows(source)` function, which returns a NamedTuple-iterator over the rows of a source. 
"""
function stream! end

# skipfield! and skiprow! only apply to Data.Field/Data.Row streaming
skipfield!(source, S, T, row, col) = Data.accesspattern(source) == Data.RandomAccess() ? nothing : Data.streamfrom(source, S, T, row, col)
function skiprow!(source, S, row, col)
    Data.accesspattern(source) == Data.RandomAccess() && return
    sch = Data.schema(source)
    cols = size(sch, 2)
    types = Data.types(sch)
    for i = col:cols
        Data.streamfrom(source, S, types[i], row, i)
    end
    return
end
function skiprows!(source, S, from, to)
    Data.accesspattern(source) == Data.RandomAccess() && return
    sch = Data.schema(source)
    cols = size(sch, 2)
    types = Data.types(sch)
    for row = from:to
        for col = 1:cols
            Data.streamfrom(source, S, types[col], row, col)
        end
    end
end

datatype(T) = eval(Base.datatype_module(Base.unwrap_unionall(T)), Base.datatype_name(T))

# generic public definitions
const TRUE = x->true
# the 2 methods below are safe and expected to be called from higher-level package convenience functions (e.g. CSV.read)
function Data.stream!(source::So, ::Type{Si}, args...;
                        append::Bool=false,
                        transforms::Dict=Dict{Int, Function}(),
                        filter::Function=TRUE,
                        columns::Vector=[],
                        kwargs...) where {So, Si}
    S = datatype(Si)
    sinkstreamtypes = Data.streamtypes(S)
    for sinkstreamtype in sinkstreamtypes
        if Data.streamtype(datatype(So), sinkstreamtype)
            source_schema = Data.schema(source)
            wk = weakrefstrings(S)
            sink_schema, transforms2 = Data.transform(source_schema, transforms, wk)
            if wk
                sink = S(sink_schema, sinkstreamtype, append, args...; reference=Data.reference(source), kwargs...)
            else
                sink = S(sink_schema, sinkstreamtype, append, args...; kwargs...)
            end
            sourcerows = size(source_schema, 1)
            sinkrows = size(sink_schema, 1)
            sinkrowoffset = ifelse(append, ifelse(ismissing(sourcerows), sinkrows, max(0, sinkrows - sourcerows)), 0)
            return Data.stream!(source, sinkstreamtype, sink, source_schema, sinkrowoffset, transforms2, filter, columns, Ref{Tuple(map(Symbol, Data.header(source_schema)))})
        end
    end
    throw(ArgumentError("`source` doesn't support the supported streaming types of `sink`: $sinkstreamtypes"))
end

function Data.stream!(source::So, sink::Si;
                        append::Bool=false,
                        transforms::Dict=Dict{Int, Function}(),
                        filter::Function=TRUE,
                        columns::Vector=[]) where {So, Si}
    S = datatype(Si)
    sinkstreamtypes = Data.streamtypes(S)
    for sinkstreamtype in sinkstreamtypes
        if Data.streamtype(datatype(So), sinkstreamtype)
            source_schema = Data.schema(source)
            wk = weakrefstrings(S)
            sink_schema, transforms2 = transform(source_schema, transforms, wk)
            if wk
                sink = S(sink, sink_schema, sinkstreamtype, append; reference=Data.reference(source))
            else
                sink = S(sink, sink_schema, sinkstreamtype, append)
            end
            sourcerows = size(source_schema, 1)
            sinkrows = size(sink_schema, 1)
            sinkrowoffset = ifelse(append, ifelse(ismissing(sourcerows), sinkrows, max(0, sinkrows - sourcerows)), 0)
            return Data.stream!(source, sinkstreamtype, sink, source_schema, sinkrowoffset, transforms2, filter, columns, Ref{Tuple(map(Symbol, Data.header(source_schema)))})
        end
    end
    throw(ArgumentError("`source` doesn't support the supported streaming types of `sink`: $sinkstreamtypes"))
end

function inner_loop(::Type{Val{N}}, ::Type{S}, ::Type{Val{homogeneous}}, ::Type{T}, knownrows::Type{Val{R}}, names, sourcetypes) where {N, S <: StreamType, homogeneous, T, R}
    if S == Data.Row
        vals = Tuple(:($(Symbol("val_$col"))) for col = 1:N)
        out = @static if isdefined(Core, :NamedTuple)
                :(vals = NamedTuple{$names, $(Tuple{sourcetypes...})}(($(vals...),)))
            else
                exprs = [:($nm::$typ) for (nm, typ) in zip(names, sourcetypes)]
                nt = NamedTuples.make_tuple(exprs)
                :(vals = $nt($(vals...)))
            end
        loop = quote 
            $((:($(Symbol("val_$col")) = Data.streamfrom(source, Data.Field, sourcetypes[$col], row, $col)) for col = 1:N)...)
            $out
            Data.streamto!(sink, Data.Row, vals, sinkrowoffset + row, 0, $knownrows)
        end
    elseif N < 500
        # println("generating inner_loop w/ @nexprs...")
        incr = S == Data.Column ? :(cur_row = length($(Symbol(string("val_", N))))) : :(nothing)
        loop = quote
            Base.@nexprs $N col->begin
                val_col = Data.streamfrom(source, $S, sourcetypes[col], row, col)
                # hack to improve codegen due to inability of inference to inline Union{T, Missing} val_col here
                if val_col isa Missing
                    Data.streamto!(sink, $S, transforms[col](val_col), sinkrowoffset + row, col, $knownrows)
                else
                    Data.streamto!(sink, $S, transforms[col](val_col), sinkrowoffset + row, col, $knownrows)
                end
            end
            $incr
        end
    elseif homogeneous
        # println("generating inner_loop w/ homogeneous types...")
        loop = quote
            for col = 1:$N
                val = Data.streamfrom(source, $S, $T, row, col)
                if val isa Missing
                    Data.streamto!(sink, $S, transforms[col](val), sinkrowoffset + row, col, $knownrows)
                else
                    Data.streamto!(sink, $S, transforms[col](val), sinkrowoffset + row, col, $knownrows)
                end
                $(S == Data.Column && :(cur_row = length(val)))
            end
        end
    else
        # println("generating inner_loop w/ > 500 columns...")
        loop = quote
            for col = 1:cols
                @inbounds cur_row = Data.streamto!(sink, $S, source, sourcetypes[col], row, sinkrowoffset, col, transforms[col], $knownrows)
            end
        end
    end
    # println(macroexpand(loop))
    return loop
end

@inline function streamto!(sink, ::Type{S}, source, ::Type{T}, row, sinkrowoffset, col::Int, f::Base.Callable, knownrows) where {S, T}
    val = Data.streamfrom(source, S, T, row, col)
    if val isa Missing
        Data.streamto!(sink, S, f(val), sinkrowoffset + row, col, knownrows)
    else
        Data.streamto!(sink, S, f(val), sinkrowoffset + row, col, knownrows)
    end
    return length(val)
end

function generate_loop(::Type{Val{knownrows}}, ::Type{S}, inner_loop) where {knownrows, S <: StreamType}
    if knownrows && S == Data.Field
        # println("generating loop w/ known rows...")
        loop = quote
            for row = 1:rows
                $inner_loop
            end
        end
    else
        # println("generating loop w/ unknown rows...")
        loop = quote
            row = cur_row = 1
            while true
                $inner_loop
                row += cur_row # will be 1 for Data.Field, length(val) for Data.Column
                Data.isdone(source, row, cols, rows, cols) && break
            end
            Data.setrows!(source, row)
        end
    end
    # println(macroexpand(loop))
    return loop
end

@generated function Data.stream!(source::So, ::Type{S}, sink::Si,
                        source_schema::Schema{R, T1}, sinkrowoffset,
                        transforms, filter, columns, ::Type{Ref{names}}) where {So, S <: StreamType, Si, R, T1, names}
    types = T1.parameters
    sourcetypes = Tuple(types)
    homogeneous = Val{all(i -> (types[1] === i), types)}
    T = isempty(types) ? Any : types[1]
    N = Val{length(types)}
    knownrows = R ? Val{true} : Val{false}
    RR = R ? Int : Missing
    r = quote
        rows, cols = size(source_schema)::Tuple{$RR, Int}
        Data.isdone(source, 1, 1, rows, cols) && return sink
        sourcetypes = $sourcetypes
        N = $N
        try
            $(generate_loop(knownrows, S, inner_loop(N, S, homogeneous, T, knownrows, names, sourcetypes)))
        catch e
            Data.cleanup!(sink)
            rethrow(e)
        end
        return sink
    end
    # println(r)
    return r
end



#
# expanded from: include("namedtuples.jl")
#

if !isdefined(Core, :NamedTuple)
    using NamedTuples
    function Base.get(f::Function, nt::NamedTuple, k)
        return haskey(nt, k) ? nt[k] : f()
    end
end

# Source/Sink with NamedTuple, both row and column oriented
"A default row-oriented \"table\" that supports both the `Data.Source` and `Data.Sink` interfaces. Can be used like `Data.stream!(source, Data.RowTable)`. It is represented as a Vector of NamedTuples."
const RowTable{T} = Vector{T} where {T <: NamedTuple}
Data.isdone(source::RowTable, row, col, rows, cols) = row > rows || col > cols
function Data.isdone(source::RowTable, row, col)
    rows = length(source)
    return Data.isdone(source, row, col, rows, rows > 0 ? length(rows[1]) : 0)
end
Data.streamtype(::Type{Array}, ::Type{Data.Field}) = true
@inline Data.streamfrom(source::RowTable, ::Type{Data.Field}, ::Type{T}, row, col) where {T} = source[row][col]
Data.streamtypes(::Type{Array}) = [Data.Row]
Data.accesspattern(::RowTable) = Data.RandomAccess()

@inline Data.streamto!(sink::RowTable, ::Type{Data.Row}, val, row, col) =
    row > length(sink) ? push!(sink, val) : setindex!(sink, val, row)
@inline Data.streamto!(sink::RowTable, ::Type{Data.Row}, val, row, col::Int, ::Type{Val{false}}) =
    push!(sink, val)
@inline Data.streamto!(sink::RowTable, ::Type{Data.Row}, val, row, col::Int, ::Type{Val{true}}) =
    setindex!(sink, val, row)

if isdefined(Core, :NamedTuple)

# Basically, a NamedTuple with any # of AbstractVector elements, accessed by column name
"A default column-oriented \"table\" that supports both the `Data.Source` and `Data.Sink` interfaces. Can be used like `Data.stream!(source, Data.Table). It is represented as a NamedTuple of AbstractVectors.`"
const Table = NamedTuple{names, T} where {names, T <: NTuple{N, AbstractVector{S} where S}} where {N}

function Data.schema(rt::RowTable{NamedTuple{names, T}}) where {names, T}
    return Data.Schema(Type[A for A in T.parameters],
                        collect(map(string, names)), length(rt))
end
# NamedTuple Data.Source implementation
# compute Data.Schema on the fly
function Data.schema(df::NamedTuple{names, T}) where {names, T}
    return Data.Schema(Type[eltype(A) for A in T.parameters],
                        collect(map(string, names)), length(df) == 0 ? 0 : length(getfield(df, 1)))
end

else # if isdefined(Core, :NamedTuple)

# Constraint relaxed for compatability; NamedTuples.NamedTuple does not have parameters
"A default column-oriented \"table\" that supports both the `Data.Source` and `Data.Sink` interfaces. Can be used like `Data.stream!(source, Data.Table). It is represented as a NamedTuple of AbstractVectors.`"
const Table = NamedTuple

function Data.schema(rt::RowTable{T}) where {T}
    return Data.Schema(Type[fieldtype(T, i) for i = 1:nfields(T)],
                        collect(map(string, fieldnames(T))), length(rt))
end
# NamedTuple Data.Source implementation
# compute Data.Schema on the fly
function Data.schema(df::NamedTuple)
    return Data.Schema(Type[eltype(A) for A in values(df)],
                        collect(map(string, keys(df))), length(df) == 0 ? 0 : length(getfield(df, 1)))
end

end # if isdefined(Core, :NamedTuple)

Data.isdone(source::Table, row, col, rows, cols) = row > rows || col > cols
function Data.isdone(source::Table, row, col)
    cols = length(source)
    return Data.isdone(source, row, col, cols == 0 ? 0 : length(getfield(source, 1)), cols)
end

# We support both kinds of streaming
Data.streamtype(::Type{<:NamedTuple}, ::Type{Data.Column}) = true
Data.streamtype(::Type{<:NamedTuple}, ::Type{Data.Field}) = true
Data.accesspattern(::Table) = Data.RandomAccess()

# Data.streamfrom is pretty simple, just return the cell or column
@inline Data.streamfrom(source::Table, ::Type{Data.Column}, T, row::Integer, col::Integer) = source[col]
@inline Data.streamfrom(source::Table, ::Type{Data.Field}, T, row::Integer, col::Integer) = source[col][row]

# NamedTuple Data.Sink implementation
# we support both kinds of streaming to our type
Data.streamtypes(::Type{<:NamedTuple}) = [Data.Column, Data.Field]
# we support streaming WeakRefStrings
Data.weakrefstrings(::Type{<:NamedTuple}) = true

# convenience methods for "allocating" a single column for streaming
allocate(::Type{T}, rows, ref) where {T} = @uninit Vector{T}(uninitialized, rows)
# allocate(::Type{T}, rows, ref) where {T <: Union{CategoricalValue, Missing}} =
#     CategoricalArray{CategoricalArrays.unwrap_catvalue_type(T)}(rows)
# special case for WeakRefStrings
allocate(::Type{WeakRefString{T}}, rows, ref) where {T} = WeakRefStringArray(ref, WeakRefString{T}, rows)
allocate(::Type{Union{WeakRefString{T}, Missing}}, rows, ref) where {T} = WeakRefStringArray(ref, Union{WeakRefString{T}, Missing}, rows)

# NamedTuple doesn't allow duplicate names, so make sure there are no duplicates in our column names
function makeunique(names::Vector{String})
    nms = [Symbol(nm) for nm in names]
    seen = Set{Symbol}()
    for (i, x) in enumerate(nms)
        x in seen ? setindex!(nms, Symbol("$(x)_$i"), i) : push!(seen, x)
    end
    return (nms...,)
end

function Array(sch::Data.Schema{R}, ::Type{Data.Row}, append::Bool=false, args...) where {R}
    types = Data.types(sch)
    # check if we're dealing with an existing NamedTuple sink or not
    if !isempty(args) && args[1] isa RowTable && types == Data.types(Data.schema(args[1]))
        sink = args[1]
        sinkrows = size(Data.schema(sink), 1)
        if append && !R
            sch.rows = sinkrows
        else
            newsize = ifelse(!R, 0, ifelse(append, sinkrows + sch.rows, sch.rows))
            resize!(sink, newsize)
            sch.rows = newsize
        end
    else
        rows = ifelse(!R, 0, sch.rows)
        names = makeunique(Data.header(sch))
        # @show rows, names, types
        sink = @static if isdefined(Core, :NamedTuple)
                @uninit Vector{NamedTuple{names, Tuple{types...}}}(uninitialized, rows)
            else
                exprs = [:($nm::$typ) for (nm, typ) in zip(names, types)]
                @uninit Vector{eval(NamedTuples.make_tuple(exprs))}(uninitialized, rows)
            end
        sch.rows = rows
    end
    return sink
end

function Array(sink::RowTable, sch::Data.Schema, ::Type{Data.Row}, append::Bool)
    return Array(sch, Data.Row, append, sink)
end

# Construct or modify a NamedTuple to be ready to stream data from a source with a schema of `sch`
# We support WeakRefString streaming, so we include the `reference` keyword
function NamedTuple(sch::Data.Schema{R}, ::Type{S}=Data.Field,
                    append::Bool=false, args...; reference::Vector{UInt8}=UInt8[]) where {R, S <: StreamType}
    types = Data.types(sch)
    # check if we're dealing with an existing NamedTuple sink or not
    if !isempty(args) && args[1] isa Table && types == Data.types(Data.schema(args[1]))
        # passing in an existing NamedTuple Sink w/ same types as source (as indicated by `sch`)
        sink = args[1]
        sinkrows = size(Data.schema(sink), 1)
        if append && (S == Data.Column || !R) # are we appending and either column-streaming or there are an unknown # of rows
            sch.rows = sinkrows
            # dont' need to do anything because:
              # for Data.Column, we just append columns anyway (see Data.streamto! below)
              # for Data.Field, the # of rows in the source are unknown (ismissing(rows)), so we'll just push! in streamto!
        else
            # need to adjust the existing sink
            # similar to above, for Data.Column or unknown # of rows for Data.Field, we'll append!/push!, so we empty! the columns
            # if appending, we want to grow our columns to be able to include every row in source (sinkrows + sch.rows)
            # if not appending, we're just "re-using" a sink, so we just resize it to the # of rows in the source
            newsize = ifelse(S == Data.Column || !R, 0, ifelse(append, sinkrows + sch.rows, sch.rows))
            foreach(col->resize!(col, newsize), sink)
            sch.rows = newsize
        end
        # take care of a possible reference from source by letting WeakRefStringArrays hold on to them
        if !isempty(reference)
            foreach(col-> col isa WeakRefStringArray && push!(col.data, reference), sink)
        end
    else
        # allocating a fresh NamedTuple Sink; append is irrelevant
        # for Data.Column or unknown # of rows in Data.Field, we only ever append!, so just allocate empty columns
        rows = ifelse(S == Data.Column, 0, ifelse(!R, 0, sch.rows))
        names = makeunique(Data.header(sch))

        sink = @static if isdefined(Core, :NamedTuple)
                NamedTuple{names}(Tuple(allocate(types[i], rows, reference) for i = 1:length(types)))
            else
                NamedTuples.make_tuple(collect(names))((allocate(types[i], rows, reference) for i = 1:length(types))...)
            end
        sch.rows = rows
    end
    return sink
end

# Constructor that takes an existing NamedTuple sink, just pass it to our mega-constructor above
function NamedTuple(sink::Table, sch::Data.Schema, ::Type{S}, append::Bool; reference::Vector{UInt8}=UInt8[]) where {S}
    return NamedTuple(sch, S, append, sink; reference=reference)
end

# Data.streamto! is easy-peasy, if there are known # of rows from source, we pre-allocated
# so we can just set the value; otherwise (didn't pre-allocate), we push!/append! the values
@inline Data.streamto!(sink::Table, ::Type{Data.Field}, val, row, col::Int) =
    (A = getfield(sink, col); row > length(A) ? push!(A, val) : setindex!(A, val, row))
@inline Data.streamto!(sink::Table, ::Type{Data.Field}, val, row, col::Int, ::Type{Val{false}}) =
    push!(getfield(sink, col), val)
@inline Data.streamto!(sink::Table, ::Type{Data.Field}, val, row, col::Int, ::Type{Val{true}}) =
    getfield(sink, col)[row] = val
@inline Data.streamto!(sink::Table, ::Type{Data.Column}, column, row, col::Int, knownrows) =
    append!(getfield(sink, col), column)


# Row iteration for Data.Sources
struct Rows{S, NT}
    source::S
end

Base.eltype(rows::Rows{S, NT}) where {S, NT} = NT
function Base.length(rows::Rows)
    sch = Data.schema(rows.source)
    return size(sch, 1)
end

"Returns a NamedTuple-iterator of any `Data.Source`"
function rows(source::S) where {S}
    sch = Data.schema(source)
    names = makeunique(Data.header(sch))
    types = Data.types(sch)
    NT = @static if isdefined(Core, :NamedTuple)
            NamedTuple{names, Tuple{types...}}
        else
            exprs = [:($nm::$typ) for (nm, typ) in zip(names, types)]
            eval(NamedTuples.make_tuple(exprs))
        end
    return Rows{S, NT}(source)
end

Base.start(rows::Rows) = 1
@static if isdefined(Core, :NamedTuple)

@generated function Base.next(rows::Rows{S, NamedTuple{names, types}}, row::Int) where {S, names, types}
    vals = Tuple(:(Data.streamfrom(rows.source, Data.Field, $typ, row, $col)) for (col, typ) in zip(1:length(names), types.parameters) )
    r = :(($(NamedTuple{names, types})(($(vals...),)), row + 1))
    # println(r)
    return r
end
@generated function Base.done(rows::Rows{S, NamedTuple{names, types}}, row::Int) where {S, names, types}
    cols = length(names)
    return :(Data.isdone(rows.source, row, $cols))
end
else

@generated function Base.next(rows::Rows{S, NT}, row::Int) where {S, NT}
    names = fieldnames(NT)
    types = Tuple(fieldtype(NT, i) for i = 1:nfields(NT))
    vals = Tuple(:(Data.streamfrom(rows.source, Data.Field, $typ, row, $col)) for (col, typ) in zip(1:length(names), types) )
    r = :(($NT($(vals...)), row + 1))
    # println(r)
    return r
end
@generated function Base.done(rows::Rows{S, NT}, row::Int) where {S, NT}
    cols = length(fieldnames(NT))
    return :(Data.isdone(rows.source, row, $cols))
end

end


#
# expanded from: include("query.jl")
#

# function rle(t::Tuple)
#     typs = Pair{Type, Int}[]
#     len = 1
#     T = t[1]
#     for i = 2:length(t)
#         TT = t[i]
#         if T == TT
#             len += 1
#         else
#             push!(typs, T=>len)
#             T = TT
#             len = 1
#         end
#     end
#     push!(typs, T=>len)
#     return typs
# end
tuplesubset(tup, ::Tuple{}) = ()
tuplesubset(tup, inds) = (tup[inds[1]], tuplesubset(tup, Base.tail(inds))...)

import Base.|
|(::Type{A}, ::Type{B}) where {A, B} = Union{A, B}
have(x) = x !== nothing

const QueryCodeType = UInt8
const UNUSED         = 0x00
const SELECTED       = 0x01
const SCALARFILTERED = 0x02
const AGGFILTERED    = 0x04
const SCALARCOMPUTED = 0x08
const AGGCOMPUTED    = 0x10
const SORTED         = 0x20
const GROUPED        = 0x40
const AAA = 0x80

unused(code::QueryCodeType) = code === UNUSED

concat!(A, val) = push!(A, val)
concat!(A, a::AbstractArray) = append!(A, a)

filter(func, val) = func(val)
function filter(filtered, func, val)
    @inbounds for i = 1:length(val)
        !filtered[i] && continue
        filtered[i] = func(val[i])
    end
end

calculate(func, vals...) = func(vals...)
calculate(func, vals::AbstractArray...) = func.(vals...)

# Data.Field/Data.Row aggregating
@generated function aggregate(aggregates, aggkeys, aggvalues)
    # if @generated
        default = Tuple(:($T[]) for T in aggvalues.parameters)
        q = quote
            entry = get!(aggregates, aggkeys, tuple($(default...)))
            $((:(push!(entry[$i], aggvalues[$i]);) for i = 1:length(aggvalues.parameters))...)
        end
        # println(q)
        return q
    # else
    #     entry = get!(aggregates, aggkeys, Tuple(typeof(val)[] for val in aggvalues))
    #     for (A, val) in zip(entry, aggvalues)
    #         push!(A, val)
    #     end
    # end
end
# Data.Column aggregating
@generated function aggregate(aggregates::Dict{K}, aggkeys::T, aggvalues) where {K, T <: NTuple{N, Vector{TT} where TT}} where {N}
    # if @generated
        len = length(aggkeys.parameters)
        vallen = length(aggvalues.parameters)
        inds = Tuple(:(aggkeys[$i][i]) for i = 1:len)
        valueinds = Tuple(:(aggvalues[$i][sortinds]) for i = 1:vallen)
        default = Tuple(:($T[]) for T in aggvalues.parameters)
        q = quote
            # SoA => AoS
            len = length(aggkeys[1])
            aos = @uninit Vector{$K}(uninitialized, len)
            for i = 1:len
                aos[i] = tuple($(inds...))
            end
            sortinds = sortperm(aos)
            aos = aos[sortinds]
            sortedvalues = tuple($(valueinds...))
            key = aos[1]
            n = 1
            for i = 2:len
                key2 = aos[i]
                if isequal(key, key2)
                    continue
                else
                    entry = get!(aggregates, key, tuple($(default...)))
                    $((:(append!(entry[$i], sortedvalues[$i][n:i-1]);) for i = 1:vallen)...)
                    n = i
                    key = key2
                end
            end
            entry = get!(aggregates, key, tuple($(default...)))
            $((:(append!(entry[$i], sortedvalues[$i][n:end]);) for i = 1:vallen)...)
        end
        # println(q)
        return q
    # else
    #     println("slow path")
    # end
end

# Nested binary search tree for multi-column sorting
mutable struct Node{T}
    inds::Vector{Int}
    value::T
    left::Union{Node{T}, Void}
    right::Union{Node{T}, Void}
    node::Union{Node, Void}
end
Node(rowind, ::Tuple{}) = nothing
Node(rowind, t::Tuple) = Node([rowind], t[1], nothing, nothing, Node(rowind, Base.tail(t)))

insert!(node, rowind, dir, ::Tuple{}) = nothing
function insert!(node, rowind, dir, tup)
    key = tup[1]
    if key == node.value
        push!(node.inds, rowind)
        insert!(node.node, rowind, Base.tail(dir), Base.tail(tup))
    elseif (key < node.value) == dir[1]
        if have(node.left)
            insert!(node.left, rowind, dir, tup)
        else
            node.left = Node(rowind, tup)
        end
    else
        if have(node.right)
            insert!(node.right, rowind, dir, tup)
        else
            node.right = Node(rowind, tup)
        end
    end
    return
end
function inds(n::Node, ind, A::Vector{Int})
    if have(n.left)
        inds(n.left, ind, A)
    end
    if have(n.node)
        inds(n.node, ind, A)
    else
        for i in n.inds
            A[ind[]] = i
            ind[] += 1
        end
    end
    if have(n.right)
        inds(n.right, ind, A)
    end
end

function sort(sortinds, sortkeys::Tuple)
    dirs = Tuple(p.second for p in sortkeys)
    root = Node(1, Tuple(p.first[1] for p in sortkeys))
    for i = 2:length(sortkeys[1].first)
        @inbounds insert!(root, i, dirs, Tuple(p.first[i] for p in sortkeys))
    end
    inds(root, Ref(1), sortinds)
    return sortinds
end

struct Sort{ind, asc} end
sortind(::Type{Sort{ind, asc}}) where {ind, asc} = ind
sortasc(::Type{Sort{ind, asc}}) where {ind, asc} = asc

"""
Represents a column used in a Data.Query for querying a Data.Source

Passed as the `actions` argument as an array of NamedTuples to `Data.query(source, actions, sink)`

Options include:

  * `col::Integer`: reference to a source column index
  * `name`: the name the column should have in the resulting query, if none is provided, it will be auto-generated
  * `T`: the type of the column, if not provided, it will be inferred from the source
  * `hide::Bool`: whether the column should be shown in the query resultset; `hide=false` is useful for columns used only for filtering and not needed in the final resultset
  * `filter::Function`: a function to apply to this column to filter out rows where the result is `false`
  * `having::Function`: a function to apply to an aggregated column to filter out rows after applying an aggregation function
  * `compute::Function`: a function to generate a new column, requires a tuple of column indexes `computeargs` that correspond to the function inputs
  * `computeaggregate::Function`: a function to generate a new aggregated column, requires a tuple of column indexes `computeargs` that correspond to the function inputs
  * `computeargs::NTuple{N, Int}`: tuple of column indexes to indicate which columns should be used as inputs to a `compute` or `computeaggregate` function
  * `sort::Bool`: whether this column should be sorted; default `false`
  * `sortindex::Intger`: by default, a resultset will be sorted by sorted columns in the order they appear in the resultset; `sortindex` allows overriding to indicate a custom sorting order
  * `sortasc::Bool`: if a column is `sort=true`, whether it should be sorted in ascending order; default `true`
  * `group::Bool`: whether this column should be grouped, causing other columns to be aggregated
  * `aggregate::Function`: a function to reduce a columns values based on grouping keys, should be of the form `f(A::AbstractArray) => scalar`
"""
struct QueryColumn{code, T, sourceindex, sinkindex, name, sort, args}
    filter::(Function|Void)
    having::(Function|Void)
    compute::(Function|Void)
    aggregate::(Function|Void)
end

function QueryColumn(sourceindex, types=(), header=[];
                name=Symbol(""),
                T::Type=Any,
                sinkindex::(Integer|Void)=sourceindex,
                hide::Bool=false,
                filter::(Function|Void)=nothing,
                having::(Function|Void)=nothing,
                compute::(Function|Void)=nothing,
                computeaggregate::(Function|Void)=nothing,
                computeargs=nothing,
                sort::Bool=false,
                sortindex::(Integer|Void)=nothing,
                sortasc::Bool=true,
                group::Bool=false,
                aggregate::(Function|Void)=nothing,
                kwargs...)
    # validate
    have(compute) && have(computeaggregate) && throw(ArgumentError("column can't be computed as scalar & aggregate"))
    (have(compute) || have(computeaggregate)) && !have(computeargs) && throw(ArgumentError("must provide computeargs=(x, y, z) to specify column index arguments for compute function"))
    have(filter) && have(computeaggregate) && throw(ArgumentError("column can't apply scalar filter & be aggregate computed"))
    group && have(having) && throw(ArgumentError("can't apply having filter on grouping column, use scalar `filter=func` instead"))
    group && have(computeaggregate) && throw(ArgumentError("column can't be part of grouping and aggregate computed"))
    group && have(aggregate) && throw(ArgumentError("column can't be part of grouping and aggregated"))
    group && hide && throw(ArgumentError("grouped column must be included in resultset"))
    sort && !have(sortindex) && throw(ArgumentError("must provide sortindex if column is sorted"))
    sort && hide && throw(ArgumentError("sorted column must be included in resultset"))

    args = ()
    code = UNUSED
    for (arg, c) in (!hide=>SELECTED, sort=>SORTED, group=>GROUPED)
        arg && (code |= c)
    end
    for (arg, c) in ((filter, SCALARFILTERED),
                     (having, AGGFILTERED),
                     (compute, SCALARCOMPUTED),
                     (computeaggregate, AGGCOMPUTED))
        if have(arg)
            code |= c
        end
    end
    if have(compute) || have(computeaggregate)
        args = computeargs
        compute = have(compute) ? compute : computeaggregate
        T = Core.Inference.return_type(compute, have(compute) ? tuplesubset(types, args) : Tuple(Vector{T} for T in tuplesubset(types, args)))
        name = name == Symbol("") ? Symbol("Column$sinkindex") : Symbol(name)
    elseif have(aggregate)
        T = Any
        name = name == Symbol("") && length(header) >= sourceindex ? Symbol(header[sourceindex]) : Symbol(name)
    else
        T = (T == Any && length(types) >= sourceindex) ? types[sourceindex] : T
        name = name == Symbol("") && length(header) >= sourceindex ? Symbol(header[sourceindex]) : Symbol(name)
    end
    S = sort ? Sort{sortindex, sortasc} : nothing
    return QueryColumn{code, T, sourceindex, sinkindex, name, S, args}(filter, having, compute, aggregate)
end

for (f, c) in (:selected=>SELECTED,
               :scalarfiltered=>SCALARFILTERED,
               :aggfiltered=>AGGFILTERED,
               :scalarcomputed=>SCALARCOMPUTED,
               :aggcomputed=>AGGCOMPUTED,
               :sorted=>SORTED,
               :grouped=>GROUPED)
    @eval $f(code::QueryCodeType) = (code & $c) > 0
    @eval $f(x) = $f(code(x))
    @eval $f(x::QueryColumn{code}) where {code} = $f(code)
end

for (f, arg) in (:code=>:c, :T=>:t, :sourceindex=>:so, :sinkindex=>:si, :name=>:n, :sort=>:s, :args=>:a)
    @eval $f(::Type{<:QueryColumn{c, t, so, si, n, s, a}}) where {c, t, so, si, n, s, a} = $arg
    @eval $f(::QueryColumn{c, t, so, si, n, s, a}) where {c, t, so, si, n, s, a} = $arg
end

# E type parameter is for a tuple of integers corresponding to
# column index inputs for aggcomputed columns
struct Query{code, S, T, E, L, O}
    source::S
    columns::T # Tuple{QueryColumn...}, columns are in *output* order (i.e. monotonically increasing by sinkindex)
end

function Query(source::S, actions, limit=nothing, offset=nothing) where {S}
    sch = Data.schema(source)
    types = Data.types(sch)
    header = Data.header(sch)
    len = length(types)
    outlen = length(types)
    columns = []
    cols = Set()
    extras = Set()
    aggcompute_extras = Set()
    si = 0
    outcol = 1
    for x in actions
        # if not provided, set sort index order according to order columns are given
        sortindex = get(x, :sortindex) do
            sorted = get(x, :sort, false)
            if sorted
                si += 1
                return si
            else
                return nothing
            end
        end
        if get(x, :hide, false)
            sinkindex = outlen + 1
            outlen += 1
        else
            sinkindex = outcol
            outcol += 1
        end
        foreach(i->i in cols || push!(extras, i), get(x, :computeargs, ()))
        push!(columns, QueryColumn(
                        get(()->(len += 1; return len), x, :col), 
                        types, header; 
                        sinkindex=sinkindex,
                        sortindex=sortindex,
                        ((k, getfield(x, k)) for k in keys(x))...)
        )
        push!(cols, get(x, :col, 0))
        if aggcomputed(typeof(columns[end]))
            foreach(i->push!(aggcompute_extras, i), args(columns[end]))
        end
    end
    querycode = UNUSED
    for col in columns
        querycode |= code(typeof(col))
    end
    if grouped(querycode)
        for col in columns
            c = code(typeof(col))
            (grouped(c) || have(col.aggregate) || aggcomputed(c) || scalarfiltered(c)) ||
                throw(ArgumentError("in query with grouped columns, each column must be grouped or aggregated: " * string(col)))
        end
    end
    append!(columns, QueryColumn(x, types, header; hide=true, sinkindex=outlen+i) for (i, x) in enumerate(Base.sort(collect(extras))))
    columns = Tuple(columns)

    return Query{querycode, S, typeof(columns), Tuple(aggcompute_extras), limit, offset}(source, columns)
end

"""
    Data.query(source, actions, sink=Data.Table, args...; append::Bool=false, limit=nothing, offset=nothing)

Query a valid DataStreams `Data.Source` according to query `actions` and stream the result into `sink`.
`limit` restricts the total number of rows streamed out, while `offset` will skip initial N rows.
`append=true` will cause the `sink` to _accumulate_ the additional query resultset rows instead of replacing any existing rows in the sink.

`actions` is an array of NamedTuples, w/ each NamedTuple including one or more of the following query arguments:

  * `col::Integer`: reference to a source column index
  * `name`: the name the column should have in the resulting query, if none is provided, it will be auto-generated
  * `T`: the type of the column, if not provided, it will be inferred from the source
  * `hide::Bool`: whether the column should be shown in the query resultset; `hide=false` is useful for columns used only for filtering and not needed in the final resultset
  * `filter::Function`: a function to apply to this column to filter out rows where the result is `false`
  * `having::Function`: a function to apply to an aggregated column to filter out rows after applying an aggregation function
  * `compute::Function`: a function to generate a new column, requires a tuple of column indexes `computeargs` that correspond to the function inputs
  * `computeaggregate::Function`: a function to generate a new aggregated column, requires a tuple of column indexes `computeargs` that correspond to the function inputs
  * `computeargs::NTuple{N, Int}`: tuple of column indexes to indicate which columns should be used as inputs to a `compute` or `computeaggregate` function
  * `sort::Bool`: whether this column should be sorted; default `false`
  * `sortindex::Intger`: by default, a resultset will be sorted by sorted columns in the order they appear in the resultset; `sortindex` allows overriding to indicate a custom sorting order
  * `sortasc::Bool`: if a column is `sort=true`, whether it should be sorted in ascending order; default `true`
  * `group::Bool`: whether this column should be grouped, causing other columns to be aggregated
  * `aggregate::Function`: a function to reduce a columns values based on grouping keys, should be of the form `f(A::AbstractArray) => scalar`
"""
function query(source, actions, sink=Table, args...; append::Bool=false, limit::(Integer|Void)=nothing, offset::(Integer|Void)=nothing)
    q = Query(source, actions, limit, offset)
    sink = Data.stream!(q, sink, args...; append=append)
    return Data.close!(sink)
end

unwk(T, wk) = T
unwk(::Type{WeakRefString{T}}, wk) where {T} = wk ? WeakRefString{T} : String

"Compute the Data.Schema of the resultset of executing Data.Query `q` against its source"
function schema(q::Query{code, S, columns, e, limit, offset}, wk=true) where {code, S, columns, e, limit, offset}
    types = Tuple(unwk(T(col), wk) for col in columns.parameters if selected(col))
    header = Tuple(String(name(col)) for col in columns.parameters if selected(col))
    off = have(offset) ? offset : 0
    rows = size(Data.schema(q.source), 1)
    rows = have(limit) ? min(limit, rows - off) : rows - off
    rows = (scalarfiltered(code) | grouped(code)) ? missing : rows
    return Schema(types, header, rows)
end

# function subset(T, I, i)
#     if Base.tuple_type_head(I) == i
#         head = Base.tuple_type_head(T)
#         tail = Base.tuple_type_tail(I)
#         return tail == Tuple{} ? Tuple{head} : Base.tuple_type_cons(head, subset(Base.tuple_type_tail(T), tail, i + 1))
#     else
#         return subset(T, I, i + 1)
#     end
# end

# function tupletypesubset(::Type{T}, ::Type{I}) where {T, I}
#     if @generated
#         TT = subset(T, I, 1)
#         return :($TT)
#     else
#         Tuple{(T.parameters[i] for i in I.parameters)...}
#     end
# end

codeblock() = Expr(:block)
macro vals(ex)
    return esc(:(Symbol(string("vals", $ex))))
end
macro val(ex)
    return esc(:(Symbol(string("val", $ex))))
end

# generate the entire streaming loop, according to any QueryColumns passed by the user
function generate_loop(knownrows, S, code, columns, extras, sourcetypes, limit, offset)
    streamfrom_inner_loop = codeblock()
    streamto_inner_loop = codeblock()
    pre_outer_loop = codeblock()
    post_outer_loop = codeblock()
    post_outer_loop_streaming = codeblock()
    post_outer_loop_row_streaming_inner_loop = codeblock()
    aggregation_loop = codeblock()
    pre_aggregation_loop = codeblock()
    aggregation_inner_loop = codeblock()
    post_aggregation_loop = codeblock()
    aggregationkeys = []
    aggregationvalues = []
    aggregationcomputed = []
    aggregationfiltered = []
    sortcols = []
    sortbuffers = []
    selectedcols = []
    firstcol = nothing
    firstfilter = true
    colind = 1
    cols = collect(columns.parameters)
    sourceinds = sortperm(cols, by=x->sourceindex(x))
    sourcecolumns = [ind=>cols[ind] for ind in sourceinds]

    starting_row = 1
    if have(offset)
        starting_row = offset + 1
        push!(pre_outer_loop.args, :(Data.skiprows!(source, $S, 1, $offset)))
    end
    rows = have(limit) ? :(min(rows, $(starting_row + limit - 1))) : :rows

    # loop thru sourcecolumns first, to ensure we stream everything we need from the Data.Source
    for (ind, col) in sourcecolumns
        si = sourceindex(col)
        out = sinkindex(col)
        if out == 1
            # keeping track of the first streamed column is handy later
            firstcol = col
        end
        SF = S == Data.Row ? Data.Field : S
        # streamfrom_inner_loop
        # we can skip any columns that aren't needed in the resultset; this works because the `sourcecolumns` are in sourceindex order
        while colind < sourceindex(col)
            push!(streamfrom_inner_loop.args, :(Data.skipfield!(source, $SF, $(sourcetypes[colind]), sourcerow, $colind)))
            colind += 1
        end
        colind += 1
        if scalarcomputed(col)
            # if the column is scalarcomputed, there's no streamfrom, we calculate from previously streamed values and the columns' `args`
            # this works because scalarcomputed columns are sorted last in `columns`
            computeargs = Tuple((@val c) for c in args(col))
            push!(streamfrom_inner_loop.args, :($(@val si) = calculate(q.columns[$ind].compute, $(computeargs...))))
        elseif !aggcomputed(col)
            # otherwise, if the column isn't aggcomputed, we just streamfrom
            r = (S == Data.Column && (have(offset) || have(limit))) ? :(sourcerow:$rows) : :sourcerow
            push!(streamfrom_inner_loop.args, :($(@val si) = Data.streamfrom(source, $SF, $(T(col)), $r, $(sourceindex(col)))))
        end
        if scalarfiltered(col)
            if S != Data.Column
                push!(streamfrom_inner_loop.args, quote
                    # in the scalar filtering case, we check this value immediately and if false,
                    # we can skip streaming the rest of the row
                    ff = filter(q.columns[$ind].filter, $(@val si))
                    if !ff
                        Data.skiprow!(source, $SF, sourcerow, $(sourceindex(col) + 1))
                        @goto end_of_loop
                    end
                end)
            else
                # Data.Column streaming means we need to accumulate row filters in a `filtered`
                # Bool array and column values will be indexed by this Bool array later
                if firstfilter
                    push!(streamfrom_inner_loop.args, :(filtered = fill(true, length($(@val si)))))
                    firstfilter = false
                end
                push!(streamfrom_inner_loop.args, :(filter(filtered, q.columns[$ind].filter, $(@val si))))
            end
        end
    end
    # now we loop through query result columns, to build up code blocks for streaming to Data.Sink
    for (ind, col) in enumerate(columns.parameters)
        si = sourceindex(col)
        out = sinkindex(col)
        # streamto_inner_loop
        if S == Data.Row
            selected(col) && push!(selectedcols, col)
        end
        if !grouped(code)
            if sorted(code)
                # if we're sorted, then we temporarily buffer all values while streaming in
                if selected(col)
                    if S == Data.Column && scalarfiltered(code)
                        push!(streamto_inner_loop.args, :(concat!($(@vals out), $(@val si)[filtered])))
                    else
                        push!(streamto_inner_loop.args, :(concat!($(@vals out), $(@val si))))
                    end
                end
            else
                # if we're not sorting or grouping, we can just stream out in the inner loop
                if selected(col)
                    if S != Data.Row
                        if S == Data.Column && scalarfiltered(code)
                            push!(streamto_inner_loop.args, :(Data.streamto!(sink, $S, $(@val si)[filtered], sinkrowoffset + sinkrow, $out, Val{$knownrows})))
                        else
                            push!(streamto_inner_loop.args, :(Data.streamto!(sink, $S, $(@val si), sinkrowoffset + sinkrow, $out, Val{$knownrows})))
                        end
                    end
                end
            end
        end
        # aggregation_loop
        if grouped(col)
            push!(aggregationkeys, col)
            push!(pre_aggregation_loop.args, :($(@vals out) = @uninit Vector{$(T(col))}(uninitialized, length(aggregates))))
            push!(aggregation_inner_loop.args, :($(@vals out)[i] = k[$(length(aggregationkeys))]))
        elseif !aggcomputed(col) && (selected(col) || sourceindex(col) in extras)
            push!(aggregationvalues, col)
            push!(pre_aggregation_loop.args, :($(@vals out) = @uninit Vector{Any}(uninitialized, length(aggregates))))
            if selected(col)
                push!(aggregation_inner_loop.args, :($(@vals out)[i] = q.columns[$ind].aggregate(v[$(length(aggregationvalues))])))
            end
        elseif aggcomputed(col)
            push!(aggregationcomputed, ind=>col)
            push!(pre_aggregation_loop.args, :($(@vals out) = @uninit Vector{$(T(col))}(uninitialized, length(aggregates))))
        end
        if aggfiltered(col)
            push!(aggregationfiltered, col)
            push!(post_aggregation_loop.args, :(filter(filtered, q.columns[$ind].having, $(@vals out))))
        end
        if sorted(code)
            selected(col) && !aggcomputed(col) && push!(sortbuffers, col)
        end
        if sorted(col)
            push!(sortcols, col)
        end
        # post_outer_loop_streaming
        if sorted(code) || grouped(code)
            if selected(col)
                if sorted(code) && aggfiltered(code)
                    push!(post_outer_loop_streaming.args, :($(@vals out) = $(@vals out)[filtered][sortinds]))
                elseif sorted(code)
                    push!(post_outer_loop_streaming.args, :($(@vals out) = $(@vals out)[sortinds]))
                elseif aggfiltered(code)
                    push!(post_outer_loop_streaming.args, :($(@vals out) = $(@vals out)[filtered]))
                end
                if S == Data.Column
                    push!(post_outer_loop_streaming.args, :(Data.streamto!(sink, $S, $(@vals out), sinkrowoffset + sinkrow, $out, Val{$knownrows})))
                elseif S == Data.Field
                    push!(post_outer_loop_row_streaming_inner_loop.args, :(Data.streamto!(sink, $S, $(@vals out)[row], sinkrowoffset + row, $out, Val{$knownrows})))
                end
            end
        end
    end
    # pre_outer_loop
    if grouped(code)
        K = Tuple{(T(x) for x in aggregationkeys)...}
        V = Tuple{(Vector{T(x)} for x in aggregationvalues)...}
        push!(pre_outer_loop.args, :(aggregates = Dict{$K, $V}()))
        if S == Data.Column && scalarfiltered(code)
            aggkeys =   Tuple(:($(@val sourceindex(col))[filtered]) for col in aggregationkeys)
            aggvalues = Tuple(:($(@val sourceindex(col))[filtered]) for col in aggregationvalues)
        else
            aggkeys =   Tuple(:($(@val sourceindex(col))) for col in aggregationkeys)
            aggvalues = Tuple(:($(@val sourceindex(col))) for col in aggregationvalues)
        end
        # collect aggregate key value(s) and add entry(s) to aggregates dict
        push!(streamto_inner_loop.args, :(aggregate(aggregates, ($(aggkeys...),), ($(aggvalues...),))))
        # push!(streamto_inner_loop.args, :(@show aggregates))
    elseif sorted(code)
        append!(pre_outer_loop.args, :($(@vals sinkindex(col)) = $(T(col))[]) for col in sortbuffers)
    end
    # streamfrom_inner_loop
    if S == Data.Column
        push!(streamfrom_inner_loop.args, :(cur_row = length($(@val sourceindex(firstcol)))))
    end
    # aggregation_loop
    if grouped(code)
        for (ind, col) in aggregationcomputed
            valueargs = Tuple(:(v[$(findfirst(x->sourceindex(x) == i, aggregationvalues))]) for i in args(col))
            push!(aggregation_inner_loop.args, :($(@vals sinkindex(col))[i] = q.columns[$ind].compute($(valueargs...))))
        end
        if aggfiltered(code)
            unshift!(post_aggregation_loop.args, :(filtered = fill(true, length(aggregates))))
        end
        aggregation_loop = quote
            $pre_aggregation_loop
            for (i, (k, v)) in enumerate(aggregates)
                $aggregation_inner_loop
            end
            $post_aggregation_loop
        end
    end
    # post_outer_loop
    push!(post_outer_loop.args, aggregation_loop)
    if sorted(code)
        sort!(sortcols, by=x->sortind(sort(x)))
        if aggfiltered(code)
            push!(post_outer_loop.args, :(sortinds = fill(0, sum(filtered))))
            sortkeys = Tuple(:($(@vals sinkindex(x))[filtered]=>$(sortasc(sort(x)))) for x in sortcols)
        else
            push!(post_outer_loop.args, :(sortinds = fill(0, length($(@vals sinkindex(firstcol))))))
            sortkeys = Tuple(:($(@vals sinkindex(x))=>$(sortasc(sort(x)))) for x in sortcols)
        end
        push!(post_outer_loop.args, :(sort(sortinds, ($(sortkeys...),))))
    end
    push!(post_outer_loop.args, post_outer_loop_streaming)
    # Data.Row streaming out
    if sorted(code) || grouped(code)
        if S == Data.Field || S == Data.Row
            if S == Data.Row
                # post_outer_loop_row_streaming_inner_loop
                names = Tuple(name(x) for x in selectedcols)
                types = Tuple{(T(x) for x in selectedcols)...}
                inds = Tuple(:($(@vals sinkindex(x))[row]) for x in selectedcols)
                vals = @static if isdefined(Core, :NamedTuple)
                        :(vals = NamedTuple{$names, $types}(($(inds...),)))
                    else
                        exprs = [:($nm::$typ) for (nm, typ) in zip(names, types.parameters)]
                        nt = NamedTuples.make_tuple(exprs)
                        :(vals = $nt($(inds...)))
                    end
                push!(post_outer_loop_row_streaming_inner_loop.args,
                    :(Data.streamto!(sink, Data.Row, $vals, sinkrowoffset + row, 0, Val{$knownrows})))
            end
            push!(post_outer_loop.args, quote
                for row = 1:length($(@vals sinkindex(firstcol)))
                    $post_outer_loop_row_streaming_inner_loop
                end
            end)
        end
    elseif S == Data.Row
        # streamto_inner_loop
        names = Tuple(name(x) for x in selectedcols)
        types = Tuple{(T(x) for x in selectedcols)...}
        inds = Tuple(:($(@val sourceindex(x))) for x in selectedcols)
        vals = @static if isdefined(Core, :NamedTuple)
                :(vals = NamedTuple{$names, $types}(($(inds...),)))
            else
                exprs = [:($nm::$typ) for (nm, typ) in zip(names, types.parameters)]
                nt = NamedTuples.make_tuple(exprs)
                :(vals = $nt($(inds...)))
            end
        push!(streamto_inner_loop.args,
            :(Data.streamto!(sink, Data.Row, $vals, sinkrowoffset + sinkrow, 0, Val{$knownrows})))
    end

    if knownrows && (S == Data.Field || S == Data.Row) && !sorted(code)
        # println("generating loop w/ known rows...")
        return quote
            $pre_outer_loop
            sinkrow = 1
            for sourcerow = $starting_row:$rows
                $streamfrom_inner_loop
                $streamto_inner_loop
                @label end_of_loop
                sinkrow += 1
            end
        end
    else
        return quote
            $pre_outer_loop
            sourcerow = $starting_row
            sinkrow = 1
            cur_row = 1
            while true
                $streamfrom_inner_loop
                $streamto_inner_loop
                @label end_of_loop
                sourcerow += cur_row # will be 1 for Data.Field, length(val) for Data.Column
                sinkrow += cur_row
                Data.isdone(source, sourcerow, cols, $rows, cols) && break
            end
            Data.setrows!(source, sourcerow)
            $post_outer_loop
        end
    end
end

function Data.stream!(q::Query{code, So}, ::Type{Si}, args...; append::Bool=false, kwargs...) where {code, So, Si}
    S = datatype(Si)
    sinkstreamtypes = Data.streamtypes(S)
    for sinkstreamtype in sinkstreamtypes
        if Data.streamtype(datatype(So), sinkstreamtype)
            wk = weakrefstrings(S)
            sourceschema = Data.schema(q.source)
            sinkschema = Data.schema(q, wk)
            if wk
                sink = S(sinkschema, sinkstreamtype, append, args...; reference=Data.reference(q), kwargs...)
            else
                sink = S(sinkschema, sinkstreamtype, append, args...; kwargs...)
            end
            sourcerows = size(sourceschema, 1)
            sinkrows = size(sinkschema, 1)
            sinkrowoffset = ifelse(append, ifelse(ismissing(sourcerows), sinkrows, max(0, sinkrows - sourcerows)), 0)
            return Data.stream!(q, sinkstreamtype, sink, sourceschema, sinkrowoffset)
        end
    end
    throw(ArgumentError("`source` doesn't support the supported streaming types of `sink`: $sinkstreamtypes"))
end

@generated function Data.stream!(q::Query{code, So, columns, extras, limit, offset}, ::Type{S}, sink,
                        source_schema::Data.Schema{R, T1}, sinkrowoffset) where {S <: Data.StreamType, R, T1, code, So, columns, extras, limit, offset}
    types = T1.parameters
    sourcetypes = Tuple(types)
    # runlen = rle(sourcetypes)
    T = isempty(types) ? Any : types[1]
    homogeneous = all(i -> (T === i), types)
    N = length(types)
    knownrows = R && !scalarfiltered(code) && !grouped(code)
    RR = R ? Int : Missing
    r = quote
        rows, cols = size(source_schema)::Tuple{$RR, Int}
        Data.isdone(q.source, 1, 1, rows, cols) && return sink
        source = q.source
        sourcetypes = $sourcetypes
        N = $N
        try
            $(generate_loop(knownrows, S, code, columns, extras, sourcetypes, limit, offset))
        catch e
            Data.cleanup!(sink)
            rethrow(e)
        end
        return sink
    end
    # @show columns
    # println(r)
    return r
end

#TODO: figure out non-unrolled case
  # use Any[ ] to store row vals until stream out or push
#TODO: spread, gather, sample, analytic functions
    # gather: (name=:gathered, gather=true, args=(1,2,3))
    # spread: (spread=1, value=2)

end # module Data

end # module DataStreams
