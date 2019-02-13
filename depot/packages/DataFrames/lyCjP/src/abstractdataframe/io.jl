function escapedprint(io::IO, x::Any, escapes::AbstractString)
    if header
    end
end
using DataStreams, WeakRefStrings
struct DataFrameStream{T}
end
function DataFrame(sch::Data.Schema{R}, ::Type{S}=Data.Field,
                   reference::Vector{UInt8}=UInt8[]) where {R, S <: Data.StreamType}
    if !isempty(args) && args[1] isa DataFrame && types == Data.types(Data.schema(args[1]))
        if append && (S == Data.Column || !R)
            foreach(col-> col isa WeakRefStringArray && push!(col.data, reference),
                    _columns(sink))
        end
    end
end
