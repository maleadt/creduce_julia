module DataStructures
import Base: Ordering
mutable struct PriorityQueue{K,V,O<:Ordering} <: AbstractDict{K,V}
end
end
