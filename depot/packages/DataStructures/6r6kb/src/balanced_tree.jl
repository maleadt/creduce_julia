macro invariant(expr)
end
macro invariant_support_statement(expr)
end
struct KDRec{K,D}
    parent::Int
    k::K
    d::D
    KDRec{K,D}(p::Int, k1::K, d1::D) where {K,D} = new{K,D}(p,k1,d1)
    KDRec{K,D}(p::Int) where {K,D} = new{K,D}(p)
end
struct TreeNode{K}
    child1::Int
    child2::Int
    child3::Int
    parent::Int
    splitkey1::K
    splitkey2::K
    TreeNode{K}(::Type{K}, c1::Int, c2::Int, c3::Int, p::Int) where {K} = new{K}(c1, c2, c3, p)
    TreeNode{K}(c1::Int, c2::Int, c3::Int, p::Int, sk1::K, sk2::K) where {K} =
        new{K}(c1, c2, c3, p, sk1, sk2)
end
function initializeTree!(tree::Array{TreeNode{K},1}) where K
    resize!(tree,1)
    tree[1] = TreeNode{K}(K, 1, 2, 0, 0)
    nothing
end
function initializeData!(data::Array{KDRec{K,D},1}) where {K,D}
    resize!(data, 2)
    data[1] = KDRec{K,D}(1)
    data[2] = KDRec{K,D}(1)
    nothing
end
mutable struct BalancedTree23{K, D, Ord <: Ordering}
    ord::Ord
    data::Array{KDRec{K,D}, 1}
    tree::Array{TreeNode{K}, 1}
    rootloc::Int
    depth::Int
    freetreeinds::Array{Int,1}
    freedatainds::Array{Int,1}
    useddatacells::BitSet
    deletionchild::Array{Int,1}
    deletionleftkey::Array{K,1}
    function BalancedTree23{K,D,Ord}(ord1::Ord) where {K,D,Ord<:Ordering}
        tree1 = Vector{TreeNode{K}}(undef, 1)
        initializeTree!(tree1)
        data1 = Vector{KDRec{K,D}}(undef, 2)
        initializeData!(data1)
        u1 = BitSet()
        push!(u1, 1, 2)
        new{K,D,Ord}(ord1, data1, tree1, 1, 1, Vector{Int}(), Vector{Int}(),
                     u1,
                     Vector{Int}(undef, 3), Vector{K}(undef, 3))
    end
end
@inline function cmp2_nonleaf(o::Ordering,
                              treenode::TreeNode,
                              k)
    lt(o, k, treenode.splitkey1) ? 1 : 2
end
@inline function cmp2_leaf(o::Ordering,
                           treenode::TreeNode,
                           k)
    (treenode.child2 == 2) ||
    lt(o, k, treenode.splitkey1) ? 1 : 2
end
@inline function cmp3_nonleaf(o::Ordering,
                              treenode::TreeNode,
                              k)
    lt(o, k, treenode.splitkey1) ? 1 :
    lt(o, k, treenode.splitkey2) ? 2 : 3
end
@inline function cmp3_leaf(o::Ordering,
                           treenode::TreeNode,
                           k)
    lt(o, k, treenode.splitkey1) ?                           1 :
    (treenode.child3 == 2 || lt(o, k, treenode.splitkey2)) ? 2 : 3
end
@inline function cmp2le_nonleaf(o::Ordering,
                                treenode::TreeNode,
                                k)
    !lt(o,treenode.splitkey1,k) ? 1 : 2
end
@inline function cmp2le_leaf(o::Ordering,
                             treenode::TreeNode,
                             k)
    treenode.child2 == 2 || !lt(o,treenode.splitkey1,k) ? 1 : 2
end
@inline function cmp3le_nonleaf(o::Ordering,
                                treenode::TreeNode,
                                k)
    !lt(o,treenode.splitkey1, k) ? 1 :
    !lt(o,treenode.splitkey2, k) ? 2 : 3
end
@inline function cmp3le_leaf(o::Ordering,
                             treenode::TreeNode,
                             k)
    !lt(o,treenode.splitkey1,k) ?                            1 :
    (treenode.child3 == 2 || !lt(o,treenode.splitkey2, k)) ? 2 : 3
end
function empty!(t::BalancedTree23)
    resize!(t.data,2)
    initializeData!(t.data)
    resize!(t.tree,1)
    initializeTree!(t.tree)
    t.depth = 1
    t.rootloc = 1
    t.freetreeinds = Vector{Int}()
    t.freedatainds = Vector{Int}()
    empty!(t.useddatacells)
    push!(t.useddatacells, 1, 2)
    nothing
end
eq(::ForwardOrdering, a, b) = isequal(a,b)
eq(::ReverseOrdering{ForwardOrdering}, a, b) = isequal(a,b)
eq(o::Ordering, a, b) = !lt(o, a, b) && !lt(o, b, a)
function findkey(t::BalancedTree23, k)
    curnode = t.rootloc
    for depthcount = 1 : t.depth - 1
        @inbounds thisnode = t.tree[curnode]
        cmp = thisnode.child3 == 0 ?
                         cmp2_nonleaf(t.ord, thisnode, k) :
                         cmp3_nonleaf(t.ord, thisnode, k)
        curnode = cmp == 1 ? thisnode.child1 :
                  cmp == 2 ? thisnode.child2 : thisnode.child3
    end
    @inbounds thisnode = t.tree[curnode]
    cmp = thisnode.child3 == 0 ?
                cmp2_leaf(t.ord, thisnode, k) :
                cmp3_leaf(t.ord, thisnode, k)
    curnode = cmp == 1 ? thisnode.child1 :
              cmp == 2 ? thisnode.child2 : thisnode.child3
    @inbounds return curnode, (curnode > 2 && eq(t.ord, t.data[curnode].k, k))
end
function findkeyless(t::BalancedTree23, k)
    curnode = t.rootloc
    for depthcount = 1 : t.depth - 1
        @inbounds thisnode = t.tree[curnode]
        cmp = thisnode.child3 == 0 ?
               cmp2le_nonleaf(t.ord, thisnode, k) :
               cmp3le_nonleaf(t.ord, thisnode, k)
        curnode = cmp == 1 ? thisnode.child1 :
                  cmp == 2 ? thisnode.child2 : thisnode.child3
    end
    @inbounds thisnode = t.tree[curnode]
    cmp = thisnode.child3 == 0 ?
            cmp2le_leaf(t.ord, thisnode, k) :
            cmp3le_leaf(t.ord, thisnode, k)
    curnode = cmp == 1 ? thisnode.child1 :
              cmp == 2 ? thisnode.child2 : thisnode.child3
    curnode
end
function replaceparent!(data::Array{KDRec{K,D},1}, whichind::Int, newparent::Int) where {K,D}
    data[whichind] = KDRec{K,D}(newparent, data[whichind].k, data[whichind].d)
    nothing
end
function replaceparent!(tree::Array{TreeNode{K},1}, whichind::Int, newparent::Int) where K
    tree[whichind] = TreeNode{K}(tree[whichind].child1, tree[whichind].child2,
                                 tree[whichind].child3, newparent,
                                 tree[whichind].splitkey1,
                                 tree[whichind].splitkey2)
    nothing
end
function push_or_reuse!(a::Vector, freelocs::Array{Int,1}, item)
    if isempty(freelocs)
        push!(a, item)
        return length(a)
    end
    loc = pop!(freelocs)
    a[loc] = item
    return loc
end
function insert!(t::BalancedTree23{K,D,Ord}, k, d, allowdups::Bool) where {K,D,Ord <: Ordering}
    leafind, exactfound = findkey(t, k)
    parent = t.data[leafind].parent
    if size(t.data,1) == 2
        @invariant t.rootloc == 1 && t.depth == 1
        t.tree[1] = TreeNode{K}(t.tree[1].child1, t.tree[1].child2,
                                t.tree[1].child3, t.tree[1].parent,
                                k, k)
        t.data[1] = KDRec{K,D}(t.data[1].parent, k, d)
        t.data[2] = KDRec{K,D}(t.data[2].parent, k, d)
    end
    if exactfound && !allowdups
        t.data[leafind] = KDRec{K,D}(parent, k,d)
        return false, leafind
    end
    depth = t.depth
    ord = t.ord
    newind = push_or_reuse!(t.data, t.freedatainds, KDRec{K,D}(0,k,d))
    p1 = parent
    oldchild = leafind
    newchild = newind
    minkeynewchild = k
    splitroot = false
    curdepth = depth
    while t.tree[p1].child3 > 0
        isleaf = (curdepth == depth)
        oldtreenode = t.tree[p1]
        cmp = isleaf ? cmp3_leaf(ord, oldtreenode, minkeynewchild) :
                      cmp3_nonleaf(ord, oldtreenode, minkeynewchild)
        if cmp == 1
            lefttreenodenew = TreeNode{K}(oldtreenode.child1, newchild, 0,
                                          oldtreenode.parent,
                                          minkeynewchild, minkeynewchild)
            righttreenodenew = TreeNode{K}(oldtreenode.child2, oldtreenode.child3, 0,
                                           oldtreenode.parent, oldtreenode.splitkey2,
                                           oldtreenode.splitkey2)
            minkeynewchild = oldtreenode.splitkey1
            whichp = 1
        elseif cmp == 2
            lefttreenodenew = TreeNode{K}(oldtreenode.child1, oldtreenode.child2, 0,
                                          oldtreenode.parent,
                                          oldtreenode.splitkey1, oldtreenode.splitkey1)
            righttreenodenew = TreeNode{K}(newchild, oldtreenode.child3, 0,
                                           oldtreenode.parent,
                                           oldtreenode.splitkey2, oldtreenode.splitkey2)
            whichp = 2
        else
            lefttreenodenew = TreeNode{K}(oldtreenode.child1, oldtreenode.child2, 0,
                                          oldtreenode.parent,
                                          oldtreenode.splitkey1, oldtreenode.splitkey1)
            righttreenodenew = TreeNode{K}(oldtreenode.child3, newchild, 0,
                                           oldtreenode.parent,
                                           minkeynewchild, minkeynewchild)
            minkeynewchild = oldtreenode.splitkey2
            whichp = 2
        end
        t.tree[p1] = lefttreenodenew
        newparentnum = push_or_reuse!(t.tree, t.freetreeinds, righttreenodenew)
        if isleaf
            par = (whichp == 1) ? p1 : newparentnum
            replaceparent!(t.data, newind, par)
            push!(t.useddatacells, newind)
            replaceparent!(t.data, righttreenodenew.child1, newparentnum)
            replaceparent!(t.data, righttreenodenew.child2, newparentnum)
        else
            replaceparent!(t.tree, righttreenodenew.child1, newparentnum)
            replaceparent!(t.tree, righttreenodenew.child2, newparentnum)
        end
        oldchild = p1
        newchild = newparentnum
        if p1 == t.rootloc
            @invariant curdepth == 1
            splitroot = true
            break
        end
        p1 = t.tree[oldchild].parent
        curdepth -= 1
    end
    if !splitroot
        isleaf = curdepth == depth
        oldtreenode = t.tree[p1]
        cmpres = isleaf ? cmp2_leaf(ord, oldtreenode, minkeynewchild) :
                         cmp2_nonleaf(ord, oldtreenode, minkeynewchild)
        t.tree[p1] = cmpres == 1 ?
                         TreeNode{K}(oldtreenode.child1, newchild, oldtreenode.child2,
                                     oldtreenode.parent,
                                     minkeynewchild, oldtreenode.splitkey1) :
                         TreeNode{K}(oldtreenode.child1, oldtreenode.child2, newchild,
                                     oldtreenode.parent,
                                     oldtreenode.splitkey1, minkeynewchild)
        if isleaf
            replaceparent!(t.data, newind, p1)
            push!(t.useddatacells, newind)
        end
    else
        newroot = TreeNode{K}(oldchild, newchild, 0, 0,
                              minkeynewchild, minkeynewchild)
        newrootloc = push_or_reuse!(t.tree, t.freetreeinds, newroot)
        replaceparent!(t.tree, oldchild, newrootloc)
        replaceparent!(t.tree, newchild, newrootloc)
        t.rootloc = newrootloc
        t.depth += 1
    end
    true, newind
end
function nextloc0(t, i::Int)
    ii = i
    @invariant i != 2 && i in t.useddatacells
    @inbounds p = t.data[i].parent
    nextchild = 0
    depthp = t.depth
    @inbounds while true
        if depthp < t.depth
            p = t.tree[ii].parent
        end
        if t.tree[p].child1 == ii
            nextchild = t.tree[p].child2
            break
        end
        if t.tree[p].child2 == ii && t.tree[p].child3 > 0
            nextchild = t.tree[p].child3
            break
        end
        ii = p
        depthp -= 1
    end
    @inbounds while true
        if depthp == t.depth
            return nextchild
        end
        p = nextchild
        nextchild = t.tree[p].child1
        depthp += 1
    end
end
function prevloc0(t::BalancedTree23, i::Int)
    @invariant i != 1 && i in t.useddatacells
    ii = i
    @inbounds p = t.data[i].parent
    prevchild = 0
    depthp = t.depth
    @inbounds while true
        if depthp < t.depth
            p = t.tree[ii].parent
        end
        if t.tree[p].child3 == ii
            prevchild = t.tree[p].child2
            break
        end
        if t.tree[p].child2 == ii
            prevchild = t.tree[p].child1
            break
        end
        ii = p
        depthp -= 1
    end
    @inbounds while true
        if depthp == t.depth
            return prevchild
        end
        p = prevchild
        c3 = t.tree[p].child3
        prevchild = c3 > 0 ? c3 : t.tree[p].child2
        depthp += 1
    end
end
function compareInd(t::BalancedTree23, i1::Int, i2::Int)
    @assert(i1 in t.useddatacells && i2 in t.useddatacells)
    if i1 == i2
        return 0
    end
    i1a = i1
    i2a = i2
    p1 = t.data[i1].parent
    p2 = t.data[i2].parent
    @invariant_support_statement curdepth = t.depth
    while true
        @invariant curdepth > 0
        if p1 == p2
            if i1a == t.tree[p1].child1
                @invariant t.tree[p1].child2 == i2a || t.tree[p1].child3 == i2a
                return -1
            end
            if i1a == t.tree[p1].child2
                if (t.tree[p1].child1 == i2a)
                    return 1
                end
                @invariant t.tree[p1].child3 == i2a
                return -1
            end
            @invariant i1a == t.tree[p1].child3
            @invariant t.tree[p1].child1 == i2a || t.tree[p1].child2 == i2a
            return 1
        end
        i1a = p1
        i2a = p2
        p1 = t.tree[i1a].parent
        p2 = t.tree[i2a].parent
        @invariant_support_statement curdepth -= 1
    end
end
beginloc(t::BalancedTree23) = nextloc0(t,1)
endloc(t::BalancedTree23) = prevloc0(t,2)
function delete!(t::BalancedTree23{K,D,Ord}, it::Int) where {K,D,Ord<:Ordering}
    p = t.data[it].parent
    newchildcount = 0
    c1 = t.tree[p].child1
    deletionleftkey1_valid = true
    if c1 != it
        deletionleftkey1_valid = false
        newchildcount += 1
        t.deletionchild[newchildcount] = c1
        t.deletionleftkey[newchildcount] = t.data[c1].k
    end
    c2 = t.tree[p].child2
    if c2 != it
        newchildcount += 1
        t.deletionchild[newchildcount] = c2
        t.deletionleftkey[newchildcount] = t.data[c2].k
    end
    c3 = t.tree[p].child3
    if c3 != it && c3 > 0
        newchildcount += 1
        t.deletionchild[newchildcount] = c3
        t.deletionleftkey[newchildcount] = t.data[c3].k
    end
    @invariant newchildcount == 1 || newchildcount == 2
    push!(t.freedatainds, it)
    pop!(t.useddatacells,it)
    defaultKey = t.tree[1].splitkey1
    curdepth = t.depth
    mustdeleteroot = false
    pparent = -1
    while true
        pparent = t.tree[p].parent
        if newchildcount == 2
            t.tree[p] = TreeNode{K}(t.deletionchild[1],
                                    t.deletionchild[2], 0, pparent,
                                    t.deletionleftkey[2], defaultKey)
            break
        end
        if newchildcount == 3
            t.tree[p] = TreeNode{K}(t.deletionchild[1], t.deletionchild[2],
                                    t.deletionchild[3], pparent,
                                    t.deletionleftkey[2], t.deletionleftkey[3])
            break
        end
        @invariant newchildcount == 1
        if curdepth == 1
            mustdeleteroot = true
            break
        end
        if t.tree[pparent].child1 == p
            rightsib = t.tree[pparent].child2
            if t.tree[rightsib].child3 == 0
                rc1 = t.tree[rightsib].child1
                rc2 = t.tree[rightsib].child2
                t.tree[p] = TreeNode{K}(t.deletionchild[1],
                                        rc1, rc2,
                                        pparent,
                                        t.tree[pparent].splitkey1,
                                        t.tree[rightsib].splitkey1)
                if curdepth == t.depth
                    replaceparent!(t.data, rc1, p)
                    replaceparent!(t.data, rc2, p)
                else
                    replaceparent!(t.tree, rc1, p)
                    replaceparent!(t.tree, rc2, p)
                end
                push!(t.freetreeinds, rightsib)
                newchildcount = 1
                t.deletionchild[1] = p
            else
                rc1 = t.tree[rightsib].child1
                t.tree[p] = TreeNode{K}(t.deletionchild[1], rc1, 0,
                                        pparent,
                                        t.tree[pparent].splitkey1,
                                        defaultKey)
                sk1 = t.tree[rightsib].splitkey1
                t.tree[rightsib] = TreeNode{K}(t.tree[rightsib].child2,
                                               t.tree[rightsib].child3,
                                               0,
                                               pparent,
                                               t.tree[rightsib].splitkey2,
                                               defaultKey)
                if curdepth == t.depth
                    replaceparent!(t.data, rc1, p)
                else
                    replaceparent!(t.tree, rc1, p)
                end
                newchildcount = 2
                t.deletionchild[1] = p
                t.deletionchild[2] = rightsib
                t.deletionleftkey[2] = sk1
            end
            c3 = t.tree[pparent].child3
            if c3 > 0
                newchildcount += 1
                t.deletionchild[newchildcount] = c3
                t.deletionleftkey[newchildcount] = t.tree[pparent].splitkey2
            end
            p = pparent
        elseif t.tree[pparent].child2 == p
            leftsib = t.tree[pparent].child1
            lk = deletionleftkey1_valid ?
                      t.deletionleftkey[1] :
                      t.tree[pparent].splitkey1
            if t.tree[leftsib].child3 == 0
                lc1 = t.tree[leftsib].child1
                lc2 = t.tree[leftsib].child2
                t.tree[p] = TreeNode{K}(lc1, lc2,
                                        t.deletionchild[1],
                                        pparent,
                                        t.tree[leftsib].splitkey1,
                                        lk)
                if curdepth == t.depth
                    replaceparent!(t.data, lc1, p)
                    replaceparent!(t.data, lc2, p)
                else
                    replaceparent!(t.tree, lc1, p)
                    replaceparent!(t.tree, lc2, p)
                end
                push!(t.freetreeinds, leftsib)
                newchildcount = 1
                t.deletionchild[1] = p
            else
                lc3 = t.tree[leftsib].child3
                t.tree[p] = TreeNode{K}(lc3, t.deletionchild[1], 0,
                                        pparent, lk, defaultKey)
                sk2 = t.tree[leftsib].splitkey2
                t.tree[leftsib] = TreeNode{K}(t.tree[leftsib].child1,
                                              t.tree[leftsib].child2,
                                              0, pparent,
                                              t.tree[leftsib].splitkey1,
                                              defaultKey)
                if curdepth == t.depth
                    replaceparent!(t.data, lc3, p)
                else
                    replaceparent!(t.tree, lc3, p)
                end
                newchildcount = 2
                t.deletionchild[1] = leftsib
                t.deletionchild[2] = p
                t.deletionleftkey[2] = sk2
            end
            c3 = t.tree[pparent].child3
            if c3 > 0
                newchildcount += 1
                t.deletionchild[newchildcount] = c3
                t.deletionleftkey[newchildcount] = t.tree[pparent].splitkey2
            end
            p = pparent
            deletionleftkey1_valid = false
        else
            @invariant t.tree[pparent].child3 == p
            leftsib = t.tree[pparent].child2
            lk = deletionleftkey1_valid ?
                       t.deletionleftkey[1] :
                       t.tree[pparent].splitkey2
            if t.tree[leftsib].child3 == 0
                lc1 = t.tree[leftsib].child1
                lc2 = t.tree[leftsib].child2
                t.tree[p] = TreeNode{K}(lc1, lc2,
                                        t.deletionchild[1],
                                        pparent,
                                        t.tree[leftsib].splitkey1,
                                        lk)
                if curdepth == t.depth
                    replaceparent!(t.data, lc1, p)
                    replaceparent!(t.data, lc2, p)
                else
                    replaceparent!(t.tree, lc1, p)
                    replaceparent!(t.tree, lc2, p)
                end
                push!(t.freetreeinds, leftsib)
                newchildcount = 2
                t.deletionchild[1] = t.tree[pparent].child1
                t.deletionleftkey[2] = t.tree[pparent].splitkey1
                t.deletionchild[2] = p
            else
                lc3 = t.tree[leftsib].child3
                t.tree[p] = TreeNode{K}(lc3, t.deletionchild[1], 0,
                                        pparent, lk, defaultKey)
                sk2 = t.tree[leftsib].splitkey2
                t.tree[leftsib] = TreeNode{K}(t.tree[leftsib].child1,
                                              t.tree[leftsib].child2,
                                              0, pparent,
                                              t.tree[leftsib].splitkey1,
                                              defaultKey)
                if curdepth == t.depth
                    replaceparent!(t.data, lc3, p)
                else
                    replaceparent!(t.tree, lc3, p)
                end
                newchildcount = 3
                t.deletionchild[1] = t.tree[pparent].child1
                t.deletionchild[2] = leftsib
                t.deletionchild[3] = p
                t.deletionleftkey[2] = t.tree[pparent].splitkey1
                t.deletionleftkey[3] = sk2
            end
            p = pparent
            deletionleftkey1_valid = false
        end
        curdepth -= 1
    end
    if mustdeleteroot
        @invariant !deletionleftkey1_valid
        @invariant p == t.rootloc
        t.rootloc = t.deletionchild[1]
        t.depth -= 1
        push!(t.freetreeinds, p)
    end
    if deletionleftkey1_valid
        while true
            pparentnode = t.tree[pparent]
            if pparentnode.child2 == p
                t.tree[pparent] = TreeNode{K}(pparentnode.child1,
                                              pparentnode.child2,
                                              pparentnode.child3,
                                              pparentnode.parent,
                                              t.deletionleftkey[1],
                                              pparentnode.splitkey2)
                break
            elseif pparentnode.child3 == p
                t.tree[pparent] = TreeNode{K}(pparentnode.child1,
                                              pparentnode.child2,
                                              pparentnode.child3,
                                              pparentnode.parent,
                                              pparentnode.splitkey1,
                                              t.deletionleftkey[1])
                break
            else
                p = pparent
                pparent = pparentnode.parent
                curdepth -= 1
                @invariant curdepth > 0
            end
        end
    end
    nothing
end
