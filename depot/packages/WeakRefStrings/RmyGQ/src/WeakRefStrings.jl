module WeakRefStrings
 struct a <: AbstractString
end
Base.thisind(::a, Int) = b
end 