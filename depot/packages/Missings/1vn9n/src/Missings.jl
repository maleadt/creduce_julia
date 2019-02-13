module Missings export allowmissing, disallowmissing, ismissing, missing, missings, @inline function Base.iterate(itr::EachReplaceMissing)     if hasmethod(isless, Tuple{
T, T}
)         try         catch         end     end end struct PassMissing{
F}
 <: Function end function (f::PassMissing{
F}
)(xs...) where {
F}
     if @generated     else     end end end 
# module
