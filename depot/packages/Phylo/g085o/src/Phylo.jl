 module Phylo
abstract type AbstractBranchTree{a, BLBL} end
include("Tree.jl")
include("newick.jl")
export parsenewick 
include("trim.jl")
if VERSION < v"0.7.0"
    end
end 