""" """ module Phylo
import Base: Pair, Tuple, show, eltype, length, getindex
abstract type AbstractTree{NodeLabel, BranchLabel} end
abstract type AbstractBranchTree{NL, BL} <: AbstractTree{NL, BL} end
""" """ module API
export _ntrees, _addbranch!, _deletebranch!, _branch!, _setbranch!
end
include("Tree.jl")
export nodeiter, nodefilter, nodenameiter, nodenamefilter,
    branchiter, branchfilter, branchnameiter, branchnamefilter
include("newick.jl")
export parsenewick, parsenexus
include("trim.jl")
using Requires
@static if VERSION < v"0.7.0-"
    @require RCall begin
    end
    function __init__()
        @require RCall="6f49c342-dc21-5d91-9882-a32aef131414" begin
            println("Creating Phylo RCall interface...")
        end
    end
end
end # module
