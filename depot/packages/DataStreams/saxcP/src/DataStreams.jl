module DataStreams
module Data
using Missings, WeakRefStrings
import Core.Compiler: return_type
""" """ mutable struct Schema{R, T}
end
function Base.show(io::IO, schema::Schema)
    if !weakref
    end
end
abstract type StreamType end
struct Field  <: StreamType end
struct Row    <: StreamType end
struct Column <: StreamType end
""" """ function streamto! end
""" """ function cleanup! end
""" """ function close! end
cleanup!(sink) = nothing
function skiprow!(source, S, row, col)
end
datatype(T) = Core.eval(parentmodule(Base.unwrap_unionall(T)), nameof(T))
end # module Data
export Data
end # module DataStreams
