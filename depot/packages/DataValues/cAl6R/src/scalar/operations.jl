nullable_returntype(::Type{T}) where {T} = isconcretetype(T) ? T : Union{}
# An element type satisfying for all A:
# unsafe_get(A)::unsafe_get_eltype(A)
_unsafe_get_eltype(::Type{T}) where T = Type{T}
_unsafe_get_eltype(x) = typeof(x)

# Copied from julia 0.6
maptoTuple(f) = Tuple{}
maptoTuple(f, a, b...) = Tuple{f(a), maptoTuple(f, b...).types...}

_nullable_eltype(f, A, As...) =
    Base._return_type(f, maptoTuple(_unsafe_get_eltype, A, As...))

function Dates.DateTime(dt::DataValue{T}, format::AbstractString; locale::Dates.Locale=Dates.ENGLISH) where {T <: AbstractString}
    isna(dt) ? DataValue{DateTime}() : DataValue{DateTime}(DateTime(get(dt), format, locale=locale))
end

function Dates.Date(dt::DataValue{T}, format::AbstractString; locale::Dates.Locale=Dates.ENGLISH) where {T <: AbstractString}
    isna(dt) ? DataValue{Date}() : DataValue{Date}(Date(get(dt), format, locale=locale))
end

for f in (:(Base.abs), :(Base.abs2), :(Base.conj),:(Base.sign))
    @eval begin
        function $f(a::DataValue{T}) where {T}
            if isna(a)
                DataValue{T}()
            else
                DataValue($f(get(a)))
            end
        end
    end
end


for f in (:(Base.acos), :(Base.acosh), :(Base.asin), :(Base.asinh),
        :(Base.atan), :(Base.atanh), :(Base.sin), :(Base.sinh), :(Base.cos),
        :(Base.cosh), :(Base.tan), :(Base.tanh), :(Base.exp), :(Base.exp2),
        :(Base.expm1), :(Base.log), :(Base.log10), :(Base.log1p),
        :(Base.log2), :(Base.exponent), :(Base.sqrt), :(Dates.value))
    @eval begin
        function $f(a::DataValue{T}) where {T}
            if isna(a)
                DataValue{Float64}()
            else
                DataValue{Float64}($f(get(a)))
            end
        end
    end
end

for op in (:+, :-, :*, :/, :%, :&, :|, :^, :<<, :>>, :div, :mod, :fld,
        :min, :max)
    @eval begin
        import Base.$(op)
        function $op(a::DataValue{T1},b::DataValue{T2}) where {T1,T2}
            nonnull = hasvalue(a) && hasvalue(b)
            S = _nullable_eltype($op,a,b)
            if nonnull
                return DataValue($op(get(a), get(b)))
            else
                return DataValue{nullable_returntype(S)}()
            end
        end

        function $op(a::DataValue{T1},b::T2) where {T1,T2}
            nonnull = hasvalue(a)
            S = _nullable_eltype($op,a,b)
            if nonnull
                return DataValue($op(get(a), b))
            else
                return DataValue{nullable_returntype(S)}()
            end
        end

        function $op(a::T1,b::DataValue{T2}) where {T1,T2}
            nonnull = hasvalue(b)
            S = _nullable_eltype($op,a,b)
            if nonnull
                return DataValue($op(a, get(b)))
            else
                return DataValue{nullable_returntype(S)}()
            end
        end
    end
end
