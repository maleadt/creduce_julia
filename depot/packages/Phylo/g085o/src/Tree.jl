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
function _setrootheight!(tree::BinaryTree, height::Float64)
end
""" """ const NamedTree = NamedBinaryTree = BinaryTree{DataFrame, Dict{String, Any}}
""" """ mutable struct PolytomousTree{LI, ND} <: AbstractBranchTree{String, Int}
end
function PolytomousTree(lt::PolytomousTree{LI, ND};
                        copyinfo=false, empty=true) where {LI, ND}
    return PolytomousTree{LI, ND}(nodes, branches, leafinfos, noderecords,
                            lt.rootheight)
end
function PolytomousTree{LI, ND}(leafinfos::LI,
                                rootheight::Float64 = NaN) where {LI, ND}
    return PolytomousTree{LI, ND}(nodes, OrderedDict{Int, Branch{String}}(),
                              leafinfos, noderecords, rootheight)
end
function PolytomousTree{LI, ND}(numleaves::Int = 0,
                                rootheight::Float64 = NaN) where {LI, ND}
end
function _getnodes(pt::PolytomousTree)
end
