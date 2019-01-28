__precompile__(true)
module DataStreams
module Data
using Missings, WeakRefStrings
import Core.Compiler: return_type
""" """ mutable struct Schema{R, T}
    header::Vector{String}   # column names
    rows::Union{Int, Missing}# number of rows in the dataset
    cols::Int                # number of columns in a dataset
    metadata::Dict           # for any other metadata we'd like to keep around (not used for '==' operation)
    index::Dict{String, Int} # maps column names as Strings to their index # in `header` and `types`
end
function Schema(types=(), header=["Column$i" for i = 1:length(types)], rows::Union{Integer,Missing}=0, metadata::Dict=Dict())
    !ismissing(rows) && rows < 0 && throw(ArgumentError("Invalid # of rows for Data.Schema; use `missing` to indicate an unknown # of rows"))
    types2 = Tuple(types)
    header2 = String[string(x) for x in header]
    cols = length(header2)
    cols != length(types2) && throw(ArgumentError("length(header): $(length(header2)) must == length(types): $(length(types2))"))
    return Schema{!ismissing(rows), Tuple{types2...}}(header2, rows, cols, metadata, Dict(n=>i for (i, n) in enumerate(header2)))
end
Schema(types, rows::Union{Integer,Missing}, metadata::Dict=Dict()) = Schema(types, ["Column$i" for i = 1:length(types)], rows, metadata)
header(sch::Schema) = sch.header
types(sch::Schema{R, T}) where {R, T} = Tuple(T.parameters)
function anytypes(sch::Schema{R, T}, weakref) where {R, T}
    types = T.parameters
    if !weakref
        types = map(x->x >: Missing ? ifelse(Missings.T(x) <: WeakRefString, Union{String, Missing}, x) : ifelse(x <: WeakRefString, String, x), types)
    end
    return collect(Any, types)
end
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
    newtypes = ((return_type(transforms2[x], (types[x],)) for x = 1:length(types))...,)
    if !weakref
        newtypes = map(x->x >: Missing ? ifelse(Missings.T(x) <: WeakRefString, Union{String, Missing}, x) : ifelse(x <: WeakRefString, String, x), newtypes)
    end
    return Schema(newtypes, Data.header(sch), size(sch, 1), sch.metadata), transforms2
end
transform(sch::Data.Schema, transforms::Dict{String, F}, s) where {F<:Base.Callable} =
    transform(sch, Dict{Int, F}(sch[x]=>f for (x, f) in transforms), s)
abstract type StreamType end
struct Field  <: StreamType end
struct Row    <: StreamType end
struct Column <: StreamType end
abstract type Source end
""" """ function schema end
""" """ function isdone end
isdone(source, row, col, rows, cols) = isdone(source, row, col)
""" """ function streamtype end
""" """ function reset! end
""" """ function streamfrom end
Data.streamfrom(source, ::Type{Data.Column}, T, row, col) = Data.streamfrom(source, Data.Column, T, col)
Data.streamfrom(source, ::Type{Data.Column}, T, r::AbstractRange, col) = Data.streamfrom(source, Data.Column, T, first(r), col)[r]
Data.streamtype(source, ::Type{Data.Row}) = Data.streamtype(source, Data.Field)
Data.streamtype(source, ::Type{<:StreamType}) = false
Data.reset!(source) = nothing
struct RandomAccess end
struct Sequential end
""" """ function accesspattern end
accesspattern(x) = Sequential()
const EMPTY_REFERENCE = UInt8[]
""" """ function reference end
reference(x) = EMPTY_REFERENCE
""" """ abstract type Sink end
""" """ function streamtypes end
""" """ function streamto! end
Data.streamto!(sink, S, val, row, col, knownrows) = Data.streamto!(sink, S, val, row, col)
Data.streamto!(sink, S, val, row, col) = Data.streamto!(sink, S, val, col)
""" """ function cleanup! end
""" """ function close! end
cleanup!(sink) = nothing
close!(sink) = sink
""" """ function weakrefstrings end
weakrefstrings(x) = false
""" """ function stream! end
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
datatype(T) = Core.eval(parentmodule(Base.unwrap_unionall(T)), nameof(T))
include("namedtuples.jl")
include("query.jl")
end # module Data
using .Data
export Data
end # module DataStreams
