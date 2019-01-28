module Tokens
abstract type AbstractSemiToken end
struct IntSemiToken <: AbstractSemiToken
    address::Int
end
end
