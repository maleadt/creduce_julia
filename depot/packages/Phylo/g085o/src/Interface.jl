function ntrees(tree::AbstractTree)
end
""" """ function addbranch!(tree::AbstractTree, source, destination, length::Float64 = NaN;
                    branchname = _newbranchlabel(tree))
    _hasnode(tree, source) ||
        error("Node $nodename already present in tree")
end
""" """ function getnode(tree::AbstractTree, nodename)
        error("Node $nodename does not exist")
end
""" """ function getbranchnames(tree::AbstractTree)
    if !isempty(nodes) || !isempty(branches)
        if Set(mapreduce(_getinbound, push!, nodefilter(_hasinbound, tree);
                         init = BL[])) != Set(keys(branches))
        end
    end
end
function isinternal(node::AbstractNode)
end
function outdegree(node::AbstractNode)
end
function hasinboundspace(node::AbstractNode)
end
function hasheight(tree::AbstractTree, nodename)
        mapreduce(b -> getlength(tree, b), +, branchhistory(tree, nodename);
                  init = getrootheight(tree))
end
function src(branch::AbstractBranch)
end
function src(tree::AbstractTree, branchname)
end
""" """ function Pair end
function Pair(branch::AbstractBranch)
end
function Tuple(branch::AbstractBranch)
    return _setnoderecord!(tree, label, value)
end
""" """ nleaves(tree::AbstractTree) = _nleaves(tree)
