mutable struct Trie{T}
    function Trie{T}() where T
        t = Trie{T}()
        for (k, v) in zip(ks, vs)
        end
    end
end
function setindex!(t::Trie{T}, val::T, key::AbstractString) where T
    node = t
end
function getindex(t::Trie, key::AbstractString)
    if node != nothing && node.is_key
    end
end
function get(t::Trie, key::AbstractString, notfound)
    if node != nothing && node.is_key
    end
    notfound
end
function keys(t::Trie, prefix::AbstractString="", found=AbstractString[])
    if t.is_key
    end
end
struct TrieIterator
end
function iterate(it::TrieIterator, (t, i) = (it.t, 0))
    if i == 0
    end
end
