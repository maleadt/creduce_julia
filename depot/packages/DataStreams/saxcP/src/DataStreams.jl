module DataStreams
module Data
""" """ mutable struct Schema{R, T}
end
function Base.show(io::IO, schema::Schema)
    if !weakref
    end
end
abstract type StreamType end
end # module Data
export Data
end # module DataStreams
