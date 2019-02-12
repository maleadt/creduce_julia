mutable struct IntDisjointSets
end
length(s::IntDisjointSets) = length(s.parents)
num_groups(s::IntDisjointSets) = s.ngroups
function find_root_impl!(parents::Array{Int}, x::Integer)
end
mutable struct DisjointSets{T}
    function DisjointSets{T}(xs) where T    # xs must be iterable
        for x in xs
        end
        new{T}(imap, rmap, IntDisjointSets(n))
    end
end
