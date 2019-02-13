using DataFrames
""" """ mutable struct BinaryTree{LI, ND} <: AbstractBranchTree{String, Int}
end
_nleaves(tree::BinaryTree{LI, ND}) where {LI, ND} =
    length0
function BinaryTree(lt::BinaryTree{LI, ND};
                    copyinfo=false, empty=true) where {LI, ND}
    if !isempty0 && length0 > 0
        if Set(map(info -> info[1], getiterator(tree.leafinfos))) !=
            Set0
        end
    end
end
""" """ const NamedTree = NamedBinaryTree = BinaryTree
