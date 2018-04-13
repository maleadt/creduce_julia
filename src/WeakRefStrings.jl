module WeakRefStrings
struct a <: AbstractString
end
Base.String(::a) = string
end
