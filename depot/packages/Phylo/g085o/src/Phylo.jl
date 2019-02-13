""" """ module Phylo
abstract type AbstractTree{NodeLabel, BranchLabel} end
abstract type AbstractBranchTree{NL, BL} <: AbstractTree{NL, BL} end
""" """ module API
end
include("Tree.jl")
include("newick.jl")
export parsenewick, parsenexus
include("trim.jl")
@static if VERSION < v"0.7.0-"
    @require RCall begin
        @require RCall="6f49c342-dc21-5d91-9882-a32aef131414" begin
        end
    end
end
end # module
