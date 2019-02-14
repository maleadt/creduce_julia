module WeakRefStrings
struct a <: AbstractString end
Base.thisind(::a, c) = b
end
