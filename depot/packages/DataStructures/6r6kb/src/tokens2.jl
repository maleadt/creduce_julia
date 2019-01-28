import Base: isless, isequal
const SDMContainer = Union{SortedDict, SortedMultiDict}
const SAContainer = Union{SDMContainer, SortedSet}
const Token = Tuple{SAContainer, IntSemiToken}
const SDMToken = Tuple{SDMContainer, IntSemiToken}
const SetToken = Tuple{SortedSet, IntSemiToken}
@inline startof(m::SAContainer) = IntSemiToken(beginloc(m.bt))
@inline lastindex(m::SAContainer) = IntSemiToken(endloc(m.bt))
@inline pastendsemitoken(::SAContainer) = IntSemiToken(2)
@inline beforestartsemitoken(::SAContainer) = IntSemiToken(1)
@inline function delete!(ii::Token)
    has_data(ii)
    delete!(ii[1].bt, ii[2].address)
end
@inline function advance(ii::Token)
    not_pastend(ii)
    IntSemiToken(nextloc0(ii[1].bt, ii[2].address))
end
@inline function regress(ii::Token)
    not_beforestart(ii)
    IntSemiToken(prevloc0(ii[1].bt, ii[2].address))
end
@inline status(ii::Token) =
       !(ii[2].address in ii[1].bt.useddatacells) ? 0 :
         ii[2].address == 1 ?                       2 :
         ii[2].address == 2 ?                       3 : 1
""" """ @inline compare(m::SAContainer, s::IntSemiToken, t::IntSemiToken) =
      compareInd(m.bt, s.address, t.address)
@inline function deref(ii::SDMToken)
    has_data(ii)
    return Pair(ii[1].bt.data[ii[2].address].k, ii[1].bt.data[ii[2].address].d)
end
@inline function deref(ii::SetToken)
    has_data(ii)
    return ii[1].bt.data[ii[2].address].k
end
@inline function deref_key(ii::SDMToken)
    has_data(ii)
    return ii[1].bt.data[ii[2].address].k
end
@inline function deref_value(ii::SDMToken)
    has_data(ii)
    return ii[1].bt.data[ii[2].address].d
end
@inline function getindex(m::SortedDict,
                          i::IntSemiToken)
    has_data((m,i))
    return m.bt.data[i.address].d
end
@inline function getindex(m::SortedMultiDict,
                          i::IntSemiToken)
    has_data((m,i))
    return m.bt.data[i.address].d
end
@inline function setindex!(m::SortedDict,
                           d_,
                           i::IntSemiToken)
    has_data((m,i))
    m.bt.data[i.address] = KDRec{keytype(m),valtype(m)}(m.bt.data[i.address].parent,
                                                         m.bt.data[i.address].k,
                                                         convert(valtype(m),d_))
    m
end
@inline function setindex!(m::SortedMultiDict,
                           d_,
                           i::IntSemiToken)
    has_data((m,i))
    m.bt.data[i.address] = KDRec{keytype(m),valtype(m)}(m.bt.data[i.address].parent,
                                                         m.bt.data[i.address].k,
                                                         convert(valtype(m),d_))
    m
end
@inline function searchsortedfirst(m::SAContainer, k_)
    i = findkeyless(m.bt, convert(keytype(m), k_))
    IntSemiToken(nextloc0(m.bt, i))
end
@inline function searchsortedafter(m::SAContainer, k_)
    i, exactfound = findkey(m.bt, convert(keytype(m), k_))
    IntSemiToken(nextloc0(m.bt, i))
end
@inline function searchsortedlast(m::SAContainer, k_)
    i, exactfound = findkey(m.bt, convert(keytype(m),k_))
    IntSemiToken(i)
end
@inline not_beforestart(i::Token) =
    (!(i[2].address in i[1].bt.useddatacells) ||
     i[2].address == 1) && throw(BoundsError())
@inline not_pastend(i::Token) =
    (!(i[2].address in i[1].bt.useddatacells) ||
     i[2].address == 2) && throw(BoundsError())
@inline has_data(i::Token) =
    (!(i[2].address in i[1].bt.useddatacells) ||
     i[2].address < 3) && throw(BoundsError())
