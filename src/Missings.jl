module Missings
@static if !isdefined(Base, :skipmissing)
@inline function Base.start(itr::EachSkipMissing) end
end
end