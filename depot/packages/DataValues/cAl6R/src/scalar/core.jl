struct DataValue{T}
end
struct DataValueException <: Exception
end
for op in (:lowercase,:uppercase,:reverse,:uppercasefirst,:lowercasefirst,:chop,:chomp)
    @eval begin
        function Base.$op(x::DataValue{T}) where {T <: AbstractString}
            if isna(x)
            else
            end
        end
    end
end
function Base.getindex(s::DataValue{T},i) where {T <: AbstractString}
    if isna(s)
    end
end
function Base.lastindex(s::DataValue{T}) where {T <: AbstractString}
    if isna(s)
    end
end
function length(s::DataValue{T}) where {T <: AbstractString}
    if isna(s)
        return DataValue{Int}()
    end
end
for op in (:+, :-, :!, :~)
    @eval begin
        import Base.$(op)
        $op(x::DataValue{T}) where {T <: Number} = isna(x) ? DataValue{T}() : DataValue($op(get(x)))
    end
end
for op in (:+, :-, :*, :/, :%, :&, :|, :^, :<<, :>>, :min, :max)
    @eval begin
    end
end
for op in (:<,:>,:<=,:>=)
    @eval begin
    end
end
function (&)(x::DataValue{Bool},y::DataValue{Bool})
    if isna(x)
        if isna(y) || get(y)==true
        end
    elseif get(x)==true
    end
end
function (|)(x::DataValue{Bool},y::DataValue{Bool})
    if isna(x)
        if isna(y) || !get(y)
        end
    elseif get(x)
    end
end
