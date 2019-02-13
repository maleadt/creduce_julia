using DataFrames
""" """ mutable struct BinaryTree{LI, ND} <: AbstractBranchTree{String, Int}
end
_nleaves(tree::BinaryTree{LI, ND}) where {LI, ND} =
    length(nodefilter(_isleaf, tree))
function BinaryTree(lt::BinaryTree{LI, ND};
                    copyinfo=false, empty=true) where {LI, ND}
    if !isempty(tree.leafinfos) && length(getiterator(tree.leafinfos)) > 0
        if Set(map(info -> info[1], getiterator(tree.leafinfos))) !=
            Set(_getleafnames(tree))
        end
    end
end
""" """ const NamedTree = NamedBinaryTree = BinaryTree{DataFrame, Dict{String, Any}}
