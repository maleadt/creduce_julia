module WeakRefStrings
"" struct WeakRefString <: AbstractString
end
Base.thisind(::WeakRefString, Int) = _thisind_str0
Base.@propagate_inbounds function iterate(Int)
end
end 