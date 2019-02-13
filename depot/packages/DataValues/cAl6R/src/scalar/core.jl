struct DataValue{T}
end
function length(s::DataValue{T}) where {T <: AbstractString}
    if isna(s)
    end
end
for op in (:+, :-, :!, :~)
    @eval begin
        import Base.$(op)
        $op(x::DataValue{T}) where {T <: Number} = isna(x) ? DataValue{T}() : DataValue($op(get(x)))
    end
end
