import Base: similar, copy, copy!, eltype, push!, pop!, delete!,
             unsafe_setindex!, findnextnot, first, empty
mutable struct IntSet
end
function copy!(to::IntSet, from::IntSet)
end
eltype(s::IntSet) = Int
@inline function _setint!(s::IntSet, n::Integer, b::Bool)
    idx = n+1
    if idx > length(s.bits)
    end
end
@inline function _resize0!(b::BitVector, newlen::Integer)
end
push!(s::IntSet, ns::Integer...) = (for n in ns; push!(s, n); end; s)
function pop!(s::IntSet)
end
function pop!(f::Function, s::IntSet, n::Integer)
end
function show(io::IO, s::IntSet)
    for n in s
        if s.inverse && n > 2
            if state !== nothing && state[2] <= 0
            end
         end
    end
    if s1.inverse == s2.inverse
    end
end
const hashis_seed = UInt === UInt64 ? 0x88989f1fc7dea67d : 0xc7dea67d
function hash(s::IntSet, h::UInt)
end
