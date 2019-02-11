import Distributions: ValueSupport, Sampleable
import Base: eltype, rand
using Phylo
using Distributions
mutable struct Phylogenetics{T <: AbstractTree} <: ValueSupport end
""" """ struct Nonultrametric{T <: AbstractTree,
                      RNG <: Sampleable{Univariate, Continuous}} <:
    Sampleable{Univariate, Phylogenetics{T}}
    function Nonultrametric{T, RNG}(n::Int, tiplabels::Vector{String}, rng::RNG, leafinfo) where {T, RNG}
    end
    function Nonultrametric{T, RNG}(tiplabels::Vector{String}, rng::RNG) where {T, RNG}
    end
end
function Nonultrametric{T}(n::Int) where T <: AbstractTree
end
function Nonultrametric{T}(tiplabels::Vector{String}) where T <: AbstractTree
    while length(roots) > 1
    end
end
""" """ struct Ultrametric{T <: AbstractTree,
                 RNG <: Sampleable{Univariate, Continuous}} <:
    Sampleable{Univariate, Phylogenetics{T}}
    function Ultrametric{T, RNG}(n::Int, tiplabels::Vector{String}, rng::RNG, leafinfo) where {T, RNG}
        return new{T, RNG}(n, tiplabels, rng, leafinfo)
    end
end
function Ultrametric{T}(n::Int) where T <: AbstractTree
end
function Ultrametric{T}(tiplabels::Vector{String}) where T <: AbstractTree
end
function rand(t::Ultrametric{T, RNG}) where {T, RNG}
    if ismissing(t.leafinfo)
    end
end
function rand(s::S, treenames) where
    {TREE <: AbstractTree,
     S <: Sampleable{Univariate, Phylogenetics{TREE}}}
    for name in treenames
    end
end
function rand(s::S, n::Int) where
    {TREE <: AbstractTree,
     S <: Sampleable{Univariate, Phylogenetics{TREE}}}
end
