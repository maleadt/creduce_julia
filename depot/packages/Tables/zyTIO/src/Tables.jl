module Tables
using Requires
function __init__()
    @require DataValues="e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5" include("datavalues.jl")
    @require CategoricalArrays="324d7699-5711-5eae-9e2f-1d82baa6b597" begin
    end
end
rowaccess(::Type{T}) where {T} = false
columnaccess(x::T) where {T} = columnaccess(T)
materializer(x) = columntable
""" """ struct Schema{names, types} end
function Base.getproperty(sch::Schema{names, types}, field::Symbol) where {names, types}
    if field === :names
    end
end
include("namedtuples.jl")
end # module
