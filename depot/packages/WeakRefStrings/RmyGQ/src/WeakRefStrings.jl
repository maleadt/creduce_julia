module WeakRefStrings
"" struct WeakRefString <: AbstractString
end
Base.thisind(::WeakRefString, Int) = _thisind_str0
end 