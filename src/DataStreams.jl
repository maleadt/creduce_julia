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
abstract type StreamType end
struct Field  <: StreamType end
struct Row    <: StreamType end
struct Column <: StreamType end
abstract type Source end
function schema end
function isdone end
isdone(source, row, col, rows, cols) = isdone(source, row, col)
function streamtype end
function reset! end
function streamfrom end
Data.streamfrom(source, ::Type{Data.Column}, T, row, col) = Data.streamfrom(source, Data.Column, T, col)
Data.streamfrom(source, ::Type{Data.Column}, T, r::AbstractRange, col) = Data.streamfrom(source, Data.Column, T, first(r), col)[r]
Data.streamtype(source, ::Type{Data.Row}) = Data.streamtype(source, Data.Field)
Data.streamtype(source, ::Type{<:StreamType}) = false
Data.reset!(source) = nothing
struct RandomAccess end
struct Sequential end
function accesspattern end
accesspattern(x) = Sequential()
const EMPTY_REFERENCE = UInt8[]
function reference end
reference(x) = EMPTY_REFERENCE
abstract type Sink end
function streamtypes end
function streamto! end
Data.streamto!(sink, S, val, row, col, knownrows) = Data.streamto!(sink, S, val, row, col)
Data.streamto!(sink, S, val, row, col) = Data.streamto!(sink, S, val, col)
function cleanup! end
function close! end
cleanup!(sink) = nothing
close!(sink) = sink
function weakrefstrings end
weakrefstrings(x) = false
function stream! end
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
const TRUE = x->true
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
if !isdefined(Core, :NamedTuple)
    using NamedTuples
    function Base.get(f::Function, nt::NamedTuple, k)
        return haskey(nt, k) ? nt[k] : f()
    end
end
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
"A default column-oriented \"table\" that supports both the `Data.Source` and `Data.Sink` interfaces. Can be used like `Data.stream!(source, Data.Table). It is represented as a NamedTuple of AbstractVectors.`"
const Table = NamedTuple{names, T} where {names, T <: NTuple{N, AbstractVector{S} where S}} where {N}
function Data.schema(rt::RowTable{NamedTuple{names, T}}) where {names, T}
    return Data.Schema(Type[A for A in T.parameters],
                        collect(map(string, names)), length(rt))
end
function Data.schema(df::NamedTuple{names, T}) where {names, T}
    return Data.Schema(Type[eltype(A) for A in T.parameters],
                        collect(map(string, names)), length(df) == 0 ? 0 : length(getfield(df, 1)))
end
else # if isdefined(Core, :NamedTuple)
"A default column-oriented \"table\" that supports both the `Data.Source` and `Data.Sink` interfaces. Can be used like `Data.stream!(source, Data.Table). It is represented as a NamedTuple of AbstractVectors.`"
const Table = NamedTuple
function Data.schema(rt::RowTable{T}) where {T}
    return Data.Schema(Type[fieldtype(T, i) for i = 1:nfields(T)],
                        collect(map(string, fieldnames(T))), length(rt))
end
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
Data.streamtype(::Type{<:NamedTuple}, ::Type{Data.Column}) = true
Data.streamtype(::Type{<:NamedTuple}, ::Type{Data.Field}) = true
Data.accesspattern(::Table) = Data.RandomAccess()
@inline Data.streamfrom(source::Table, ::Type{Data.Column}, T, row::Integer, col::Integer) = source[col]
@inline Data.streamfrom(source::Table, ::Type{Data.Field}, T, row::Integer, col::Integer) = source[col][row]
Data.streamtypes(::Type{<:NamedTuple}) = [Data.Column, Data.Field]
Data.weakrefstrings(::Type{<:NamedTuple}) = true
allocate(::Type{T}, rows, ref) where {T} = @uninit Vector{T}(uninitialized, rows)
allocate(::Type{WeakRefString{T}}, rows, ref) where {T} = WeakRefStringArray(ref, WeakRefString{T}, rows)
allocate(::Type{Union{WeakRefString{T}, Missing}}, rows, ref) where {T} = WeakRefStringArray(ref, Union{WeakRefString{T}, Missing}, rows)
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
function NamedTuple(sink::Table, sch::Data.Schema, ::Type{S}, append::Bool; reference::Vector{UInt8}=UInt8[]) where {S}
    return NamedTuple(sch, S, append, sink; reference=reference)
end
@inline Data.streamto!(sink::Table, ::Type{Data.Field}, val, row, col::Int) =
    (A = getfield(sink, col); row > length(A) ? push!(A, val) : setindex!(A, val, row))
@inline Data.streamto!(sink::Table, ::Type{Data.Field}, val, row, col::Int, ::Type{Val{false}}) =
    push!(getfield(sink, col), val)
@inline Data.streamto!(sink::Table, ::Type{Data.Field}, val, row, col::Int, ::Type{Val{true}}) =
    getfield(sink, col)[row] = val
@inline Data.streamto!(sink::Table, ::Type{Data.Column}, column, row, col::Int, knownrows) =
    append!(getfield(sink, col), column)
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
codeblock() = Expr(:block)
macro vals(ex)
    return esc(:(Symbol(string("vals", $ex))))
end
macro val(ex)
    return esc(:(Symbol(string("val", $ex))))
end
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
  # use Any[ ] to store row vals until stream out or push
    # gather: (name=:gathered, gather=true, args=(1,2,3))
    # spread: (spread=1, value=2)
end # module Data
end # module DataStreams
