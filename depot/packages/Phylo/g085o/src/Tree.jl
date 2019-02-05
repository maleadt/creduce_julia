using DataStructures
using Compat
using Compat: @warn
using IterableTables: getiterator
using DataFrames
import Phylo.API: _getnode, _getbranch, _setnode!, _setbranch!, _nleaves
_branchtype(::AbstractBranchTree{NL, BL}) where {NL, BL} = Branch{NL}
""" """ mutable struct BinaryTree{LI, ND} <: AbstractBranchTree{String, Int}
    noderecords::OrderedDict{String, ND}
    rootheight::Float64
end
_leafinfotype(::BinaryTree{LI, ND}) where {LI, ND} = LI
_nleaves(tree::BinaryTree{LI, ND}) where {LI, ND} =
    length(nodefilter(_isleaf, tree))
function BinaryTree(lt::BinaryTree{LI, ND};
                    copyinfo=false, empty=true) where {LI, ND}
    validate(lt) || error("Tree to copy is not valid")
    if !isempty(tree.leafinfos) && length(getiterator(tree.leafinfos)) > 0
        if Set(map(info -> info[1], getiterator(tree.leafinfos))) !=
            Set(_getleafnames(tree))
            @warn "LeafInfo names do not match actual leaves of tree"
            return false
        end
    end
    return true
end
function _setrootheight!(tree::BinaryTree, height::Float64)
    tree.rootheight = height
    return height
end
""" """ const NamedTree = NamedBinaryTree = BinaryTree{DataFrame, Dict{String, Any}}
""" """ mutable struct PolytomousTree{LI, ND} <: AbstractBranchTree{String, Int}
    noderecords::OrderedDict{String, ND}
    rootheight::Float64
end
function PolytomousTree(lt::PolytomousTree{LI, ND};
                        copyinfo=false, empty=true) where {LI, ND}
    validate(lt) || error("Tree to copy is not valid")
    leafnames = getleafnames(lt)
    leafinfos = copyinfo ? deepcopy(lt.leafinfos) : lt.leafinfos
    if empty # Empty out everything else
        branches = deepcopy(getbranches(lt))
    end
    return PolytomousTree{LI, ND}(nodes, branches, leafinfos, noderecords,
                            lt.rootheight)
end
function PolytomousTree{LI, ND}(leafinfos::LI,
                                treetype::Type{PolytomousTree{LI, ND}} =
                                PolytomousTree{LI, Dict{String, Any}};
                                rootheight::Float64 = NaN) where {LI, ND}
    leafinfos = LI()
    noderecords = OrderedDict(map(leaf -> leaf => ND(), leaves))
    return PolytomousTree{LI, ND}(nodes, OrderedDict{Int, Branch{String}}(),
                              leafinfos, noderecords, rootheight)
end
function PolytomousTree{LI, ND}(numleaves::Int = 0,
                                treetype::Type{PolytomousTree{LI, ND}} =
                                PolytomousTree{LI, Dict{String, Any}};
                                rootheight::Float64 = NaN) where {LI, ND}
end
PolytomousTree(leafinfos::LI; rootheight::Float64 = NaN) where LI =
    PolytomousTree{LI, Dict{String, Any}}(leafinfos; rootheight = rootheight)
_nodetype(::PolytomousTree) = Node{Int}
function _getnodes(pt::PolytomousTree)
    return pt.nodes
end
function _getbranches(pt::PolytomousTree)
    return pt.branches
end
function _getleafinfo(pt::PolytomousTree, leaf)
    return Iterators.filter(info -> info[1] == leafname,
        getiterator(pt.leafinfos))
end
function _setleafinfo!(pt::PolytomousTree, info)
    pt.leafinfos = info
end
function _resetleaves!(pt::PolytomousTree)
    pt.leafinfos = empty!(pt.leafinfos)
    return pt
end
function _addnode!(tree::PolytomousTree{LI, NR}, nodename) where {LI, NR}
    !_hasnode(tree, nodename) ||
        error("Node $nodename already present in tree")
end
function _validate(tree::PolytomousTree)
    if length(getiterator(tree.leafinfos)) > 0
        if Set(map(info -> info[1], getiterator(tree.leafinfos))) !=
            Set(_getleafnames(tree))
            @warn "LeafInfo names do not match actual leaves of tree"
            return false
        end
    end
    if Set(keys(tree.noderecords)) != Set(keys(getnodes(tree)))
        @warn "Leaf records do not match node records of tree"
    end
end
function _setrootheight!(tree::PolytomousTree, height::Float64)
end
function _clearrootheight!(tree::PolytomousTree)
end
