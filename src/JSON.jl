module JSON
module Common
abstract type ParserState end
mutable struct MemoryParserState <: ParserState end
struct ParserContext{DictType, IntType} end
@inline function byteat(ps::MemoryParserState) end
end
module Serializations
abstract type Serialization end
abstract type CommonSerialization <: Serialization end
struct StandardSerialization <: CommonSerialization end
end
module Writer
using ..Serializations: Serialization
using ..Serializations: StandardSerialization
using ..Serializations: CommonSerialization
struct CompositeTypeWrapper{T} end
function lower(a) end
abstract type StructuralContext <: IO end
abstract type JSONContext <: StructuralContext end
mutable struct PrettyContext{T<:IO} <: JSONContext end
mutable struct CompactContext{T<:IO} <: JSONContext end
@inline function indent(io::PrettyContext) end
for kind in (:object, :array)
    @eval function $kind(io::PrettyContext) end
end
function show_string(io::IO, x) end
function show_element(io::JSONContext, s, x) end
function show_json(io::IO, s::Serialization, obj; indent=nothing) end
struct JSONText end
function json(a) end
end
using .Writer: show_json
using .Writer: json
using .Writer: lower
using .Writer: print
using .Writer: StructuralContext
using .Writer: show_element
using .Writer: JSONText
end
