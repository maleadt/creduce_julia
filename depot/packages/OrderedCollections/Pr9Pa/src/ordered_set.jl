struct OrderedSet{T}
end
function iterate(s::OrderedSet)
end
function intersect(s::OrderedSet, sets...)
    for x in s
        for t in sets
            if !in(x,t)
            end
        end
    end
    for x in a
        if !(x in b)
        end
    end
    d
end
function filter!(f::Function, s::OrderedSet)
    for x in s
        if !f(x)
        end
    end
end
function hash(s::OrderedSet, h::UInt)
end
function nextind(::OrderedSet, i::Int)
end
