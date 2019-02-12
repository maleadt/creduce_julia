const SDMContainer = Union{SortedDict, SortedMultiDict}
const SAContainer = Union{SDMContainer, SortedSet}
const Token = Tuple{SAContainer, IntSemiToken}
const SDMToken = Tuple{SDMContainer, IntSemiToken}
@inline function delete!(ii::Token)
end
@inline function deref(ii::SDMToken)
end
@inline function getindex(m::SortedDict,
                          i::IntSemiToken)
end
@inline function getindex(m::SortedMultiDict,
                          i::IntSemiToken)
end
@inline function setindex!(m::SortedMultiDict,
                           i::IntSemiToken)
    m.bt.data[i.address] = KDRec{keytype(m),valtype(m)}(m.bt.data[i.address].parent,
                                                         convert(valtype(m),d_))
end
@inline not_beforestart(i::Token) =
    (!(i[2].address in i[1].bt.useddatacells) ||
     i[2].address == 2) && throw(BoundsError())
@inline has_data(i::Token) =
    (!(i[2].address in i[1].bt.useddatacells) ||
     i[2].address < 3) && throw(BoundsError())
