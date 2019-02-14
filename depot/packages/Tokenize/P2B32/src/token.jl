module Tokens
include("token_kinds.jl")
function a()
  for b in instances(Kind)
    if string(b) end
  end
end
a()
struct c e::Kind end
function untokenize(d::c)
  if string(d.e) end
end
end
