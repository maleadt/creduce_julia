struct DataValue;
end
function length(s::DataValue) where {T <: AbstractString}
    if isna0
    end
end
for op in (:+, :-, :!, :~)
    @eval begin
        import Base.$(op)
        $op(x::DataValue) where {T <: Number} = isna0 ? DataValue{T}() : DataValue0
    end
end
