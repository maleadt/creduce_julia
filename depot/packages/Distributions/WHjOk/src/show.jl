distrname(d::Distribution) = string(typeof(d))
show(io::IO, d::Distribution) = show(io, d, fieldnames(typeof(d)))
function show(io::IO, d::Distribution, pnames)
    uml, namevals = _use_multline_show(d, pnames)
    uml ? show_multline(io, d, namevals) : show_oneline(io, d, namevals)
end
const _NameVal = Tuple{Symbol,Any}
function _use_multline_show(d::Distribution, pnames)
    namevals = _NameVal[]
    multline = false
    tlen = 0
    for (i, p) in enumerate(pnames)
        pv = getfield(d, p)
        if !(isa(pv, Number) || isa(pv, NTuple) || isa(pv, AbstractVector))
            multline = true
        else
            tlen += length(pv)
        end
        push!(namevals, (p, pv))
    end
    if tlen > 8
        multline = true
    end
    return (multline, namevals)
end
function _use_multline_show(d::Distribution)
    _use_multline_show(d, fieldnames(typeof(d)))
end
function show_oneline(io::IO, d::Distribution, namevals)
    print(io, distrname(d))
    np = length(namevals)
    print(io, '(')
    for (i, nv) in enumerate(namevals)
        (p, pv) = nv
        print(io, p)
        print(io, '=')
        show(io, pv)
        if i < np
            print(io, ", ")
        end
    end
    print(io, ')')
end
function show_multline(io::IO, d::Distribution, namevals)
    print(io, distrname(d))
    println(io, "(")
    for (p, pv) in namevals
        print(io, p)
        print(io, ": ")
        println(io, pv)
    end
    println(io, ")")
end
