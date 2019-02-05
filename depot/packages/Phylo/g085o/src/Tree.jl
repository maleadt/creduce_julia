using DataStructures
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
    noderecords::OrderedDict{String, ND}
    rootheight::Float64
end
function PolytomousTree(lt::PolytomousTree{LI, ND};
                        copyinfo=false, empty=true) where {LI, ND}
    if empty # Empty out everything else
        branches = deepcopy(getbranches(lt))
    end
    return PolytomousTree{LI, ND}(nodes, branches, leafinfos, noderecords,
                            lt.rootheight)
end
function PolytomousTree{LI, ND}(leafinfos::LI,
                                rootheight::Float64 = NaN) where {LI, ND}
    return PolytomousTree{LI, ND}(nodes, OrderedDict{Int, Branch{String}}(),
                              leafinfos, noderecords, rootheight)
end
function PolytomousTree{LI, ND}(numleaves::Int = 0,
                                treetype::Type{PolytomousTree{LI, ND}} =
                                PolytomousTree{LI, Dict{String, Any}};
                                rootheight::Float64 = NaN) where {LI, ND}
end
function _getnodes(pt::PolytomousTree)
end
function _getbranches(pt::PolytomousTree)
end
function _getleafinfo(pt::PolytomousTree, leaf)
    return Iterators.filter(info -> info[1] == leafname,
        getiterator(pt.leafinfos))
end
function _setleafinfo!(pt::PolytomousTree, info)
end
function _resetleaves!(pt::PolytomousTree)
end
function _addnode!(tree::PolytomousTree{LI, NR}, nodename) where {LI, NR}
end
function _validate(tree::PolytomousTree)
    if length(getiterator(tree.leafinfos)) > 0
        if Set(map(info -> info[1], getiterator(tree.leafinfos))) !=
            Set(_getleafnames(tree))
        end
    end
    if Set(keys(tree.noderecords)) != Set(keys(getnodes(tree)))
    end
end
function _setrootheight!(tree::PolytomousTree, height::Float64)
end
function _clearrootheight!(tree::PolytomousTree)
end
