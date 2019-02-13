"" module Phylo
abstract type AbstractBranchTree{NL, BLBL} end
include("Tree.jl")
include("newick.jl")
export parsenewick 
include("trim.jl")
@static if VERSION < v"0.7.0-"
    end
end 