const RowTable{T} = Vector{T} where {T <: NamedTuple}
function Data.isdone(source::RowTable, row, col)
end
Data.streamto!(sink::RowTable, ::Type{Data.Row}, val, row, col) =
    row > length(sink) ? push!(sink, val) : setindex!(sink, val, row)
const Table = NamedTuple{names, T} where {names, T <: NTuple{N, AbstractVector{S} where S}} where {N}
function Data.schema(rt::RowTable{NamedTuple{names, T}}) where {names, T}
end
allocate(::Type{Missing}, rows, ref) = fill(missing, rows)
function makeunique(names::Vector{String})
    types = Data.types(sch)
    if !isempty(args) && args[1] isa RowTable && types == Data.types(Data.schema(args[1]))
        if append && !R
        end
    end
end
function Array(sink::RowTable, sch::Data.Schema, ::Type{Data.Row}, append::Bool)
    return Array(sch, Data.Row, append, sink)
end
function NamedTuple(sch::Data.Schema{R}, ::Type{S}=Data.Field,
                    append::Bool=false, args...; reference::Vector{UInt8}=UInt8[]) where {R, S <: StreamType}
    if !isempty(args) && args[1] isa Table && types == Data.types(Data.schema(args[1]))
        if append && (S == Data.Column || !R) # are we appending and either column-streaming or there are an unknown # of rows
        end
        sch.rows = rows
    end
    return sink
end
function NamedTuple(sink::Table, sch::Data.Schema, ::Type{S}, append::Bool; reference::Vector{UInt8}=UInt8[]) where {S}
end
Data.streamto!(sink::Table, ::Type{Data.Field}, val, row, col::Int) =
    getfield(sink, col)[row] = val
Data.streamto!(sink::Table, ::Type{Data.Column}, column, row, col::Int, knownrows) =
    append!(getfield(sink, col), column)
struct Rows{S, NT}
end
function Base.length(rows::Rows)
end
function rows(source::S) where {S}
end
function Base.iterate(rows::Rows{S, NamedTuple{names, types}}, row::Int=1) where {S, names, types}
    if @generated
        return quote
        end
    else
    end
end
