module Tables
using Requires
using TableTraits, IteratorInterfaceExtensions
export rowtable, columntable
function __init__()
    @require DataValues="e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5" include("datavalues.jl")
    @require CategoricalArrays="324d7699-5711-5eae-9e2f-1d82baa6b597" begin
        using .CategoricalArrays
        allocatecolumn(::Type{CategoricalString{R}}, rows) where {R} = CategoricalArray{String, 1, R}(undef, rows)
        allocatecolumn(::Type{Union{CategoricalString{R}, Missing}}, rows) where {R} =
            CategoricalArray{Union{String, Missing}, 1, R}(undef, rows)
        allocatecolumn(::Type{CategoricalValue{T, R}}, rows) where {T, R} =
            CategoricalArray{T, 1, R}(undef, rows)
        allocatecolumn(::Type{Union{Missing, CategoricalValue{T, R}}}, rows) where {T, R} =
            CategoricalArray{Union{Missing, T}, 1, R}(undef, rows)
    end
    @require WeakRefStrings="ea10d353-3f73-51f8-a26c-33c1cb351aa5" begin
        using .WeakRefStrings
        allocatecolumn(::Type{WeakRefString{T}}, rows) where {T} = StringVector(rows)
        allocatecolumn(::Type{Union{Missing, WeakRefString{T}}}, rows) where {T} = StringVector{Union{Missing, String}}(rows)
        unweakref(wk::WeakRefString) = string(wk)
        unweakreftype(::Type{<:WeakRefString}) = String
    end
    @require IteratorInterfaceExtensions="82899510-4779-5014-852e-03e436cf321d" begin
        using .IteratorInterfaceExtensions
        IteratorInterfaceExtensions.getiterator(x::RowTable) = datavaluerows(x)
        IteratorInterfaceExtensions.isiterable(x::RowTable) = true
        IteratorInterfaceExtensions.getiterator(x::ColumnTable) = datavaluerows(x)
        IteratorInterfaceExtensions.isiterable(x::ColumnTable) = true
    end
end
"Abstract row type with a simple required interface: row values are accessible via `getproperty(row, field)`; for example, a NamedTuple like `nt = (a=1, b=2, c=3)` can access its value for `a` like `nt.a` which turns into a call to the function `getproperty(nt, :a)`"
abstract type Row end
""" """ abstract type Table end
istable(x::T) where {T} = istable(T) || TableTraits.isiterabletable(x) === true ||
    TableTraits.isiterabletable(x) === missing
istable(::Type{T}) where {T} = false
rowaccess(x::T) where {T} = rowaccess(T)
rowaccess(::Type{T}) where {T} = false
columnaccess(x::T) where {T} = columnaccess(T)
columnaccess(::Type{T}) where {T} = false
schema(x) = nothing
materializer(x) = columntable
""" """ struct Schema{names, types} end
Schema(names::Tuple{Vararg{Symbol}}, types::Type{T}) where {T <: Tuple} = Schema{names, T}()
Schema(::Type{NamedTuple{names, types}}) where {names, types} = Schema{names, types}()
Schema(names, ::Nothing) = Schema{Tuple(map(Symbol, names)), nothing}()
Schema(names, types) = Schema{Tuple(map(Symbol, names)), Tuple{types...}}()
function Base.show(io::IO, sch::Schema{names, types}) where {names, types}
    println(io, "Tables.Schema:")
    Base.print_matrix(io, hcat(collect(names), collect(fieldtype(types, i) for i = 1:fieldcount(types))))
end
function Base.getproperty(sch::Schema{names, types}, field::Symbol) where {names, types}
    if field === :names
        return names
    elseif field === :types
        return types === nothing ? nothing : Tuple(fieldtype(types, i) for i = 1:fieldcount(types))
    else
        throw(ArgumentError("unsupported property for Tables.Schema"))
    end
end
Base.propertynames(sch::Schema) = (:names, :types)
include("utils.jl")
include("namedtuples.jl")
include("fallbacks.jl")
include("query.jl")
end # module
