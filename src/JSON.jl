module JSON
module Common
abstract type ParserState end
mutable struct MemoryParserState <: ParserState
end
struct ParserContext{DictType, IntType} end
@inline function byteat(ps::MemoryParserState)
    @inbounds if hasmore(ps)
    end
end
end
module Serializations
abstract type Serialization end
abstract type CommonSerialization <: Serialization end
struct StandardSerialization <: CommonSerialization end
end
module Writer
using ..Serializations: Serialization, StandardSerialization,
                        CommonSerialization
struct CompositeTypeWrapper{T}
end
function lower(a)
end
abstract type StructuralContext <: IO end
abstract type JSONContext <: StructuralContext end
mutable struct PrettyContext{T<:IO} <: JSONContext
end
mutable struct CompactContext{T<:IO} <: JSONContext
end
@inline function indent(io::PrettyContext)
end
for kind in ("object", "array")
    beginfn = Symbol("begin_", kind)
    @eval function $beginfn(io::PrettyContext)
    end
end
function show_string(io::IO, x)
end
function show_element(io::JSONContext, s, x)
end
function show_json(io::IO, s::Serialization, obj; indent=nothing)
    if indent !== nothing
    end
end
struct JSONText
end
json(a) = sprint(print, a)
end
using .Writer: show_json, json, lower, print, StructuralContext, show_element,
               JSONText
end
