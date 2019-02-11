mutable struct BalancedTree23{K, D, Ord <: Ordering}
    function BalancedTree23{K,D,Ord}(ord1::Ord) where {K,D,Ord<:Ordering}
    end
end
@inline function cmp2_nonleaf(o::Ordering,
                              k)
end
@inline function cmp3_nonleaf(o::Ordering,
                              k)
end
@inline function cmp2le_nonleaf(o::Ordering,
                                k)
end
@inline function cmp2le_leaf(o::Ordering,
                             k)
end
@inline function cmp3le_leaf(o::Ordering,
                             k)
    for depthcount = 1 : t.depth - 1
        cmp = thisnode.child3 == 0 ?
               cmp2le_nonleaf(t.ord, thisnode, k) :
               cmp3le_nonleaf(t.ord, thisnode, k)
    end
    if size(t.data,1) == 2
        if cmp == 1
            lefttreenodenew = TreeNode{K}(oldtreenode.child1, oldtreenode.child2, 0,
                                           minkeynewchild, minkeynewchild)
        end
        if p1 == t.rootloc
        end
    end
    if !splitroot
        t.tree[p1] = cmpres == 1 ?
                         TreeNode{K}(oldtreenode.child1, newchild, oldtreenode.child2,
                                     minkeynewchild, oldtreenode.splitkey1) :
                         TreeNode{K}(oldtreenode.child1, oldtreenode.child2, newchild,
                                     oldtreenode.splitkey1, minkeynewchild)
        if isleaf
        end
    end
    while true
        if p1 == p2
            if i1a == t.tree[p1].child1
            end
            p = pparent
            if t.tree[leftsib].child3 == 0
                t.tree[p] = TreeNode{K}(lc1, lc2,
                                              defaultKey)
                if curdepth == t.depth
                end
            end
        end
    end
    if deletionleftkey1_valid
        while true
            if pparentnode.child2 == p
            end
        end
    end
end
