_nullable_eltype(f, A, As...) =
    Base._return_type(f, maptoTuple(_unsafe_get_eltype, A, As...))
function Dates.DateTime(dt::DataValue{T}, format::AbstractString; locale::Dates.Locale=Dates.ENGLISH) where {T <: AbstractString}
end
for f in (:(Base.abs), :(Base.abs2), :(Base.conj),:(Base.sign))
    @eval begin
        function $f(a::DataValue{T}) where {T}
            if isna(a)
            end
        end
    end
end
for f in (:(Base.acos), :(Base.acosh), :(Base.asin), :(Base.asinh),
        :(Base.log2), :(Base.exponent), :(Base.sqrt), :(Dates.value))
    @eval begin
        function $f(a::DataValue{T}) where {T}
            if isna(a)
            end
        end
    end
end
for op in (:+, :-, :*, :/, :%, :&, :|, :^, :<<, :>>, :div, :mod, :fld,
        :min, :max)
    @eval begin
        import Base.$(op)
        function $op(a::DataValue{T1},b::DataValue{T2}) where {T1,T2}
            if nonnull
                return DataValue($op(get(a), get(b)))
            end
            if nonnull
            end
        end
    end
end
