function _check_randparams(rks, x, p)
    n = length(rks)
    length(x) == length(p) == n || raise_dimerror()
    return n
end
function ordinalrank!(rks::AbstractArray, x::AbstractArray, p::IntegerArray)
    n = _check_randparams(rks, x, p)
    if n > 0
        i = 1
        while i <= n
            rks[p[i]] = i
            i += 1
        end
    end
    return rks
end
""" """ ordinalrank(x::AbstractArray; lt = isless, rev::Bool = false) =
    ordinalrank!(Array{Int}(undef, size(x)), x, sortperm(x; lt = lt, rev = rev))
function competerank!(rks::AbstractArray, x::AbstractArray, p::IntegerArray)
    n = _check_randparams(rks, x, p)
    if n > 0
        p1 = p[1]
        v = x[p1]
        rks[p1] = k = 1
        i = 2
        while i <= n
            pi = p[i]
            xi = x[pi]
            if xi == v
                rks[pi] = k
            else
                rks[pi] = k = i
                v = xi
            end
            i += 1
        end
    end
    return rks
end
""" """ competerank(x::AbstractArray; lt = isless, rev::Bool = false) =
    competerank!(Array{Int}(undef, size(x)), x, sortperm(x; lt = lt, rev = rev))
function denserank!(rks::AbstractArray, x::AbstractArray, p::IntegerArray)
    n = _check_randparams(rks, x, p)
    if n > 0
        p1 = p[1]
        v = x[p1]
        rks[p1] = k = 1
        i = 2
        while i <= n
            pi = p[i]
            xi = x[pi]
            if xi == v
                rks[pi] = k
            else
                rks[pi] = (k += 1)
                v = xi
            end
            i += 1
        end
    end
    return rks
end
""" """ denserank(x::AbstractArray; lt = isless, rev::Bool = false) =
    denserank!(Array{Int}(undef, size(x)), x, sortperm(x; lt = lt, rev = rev))
function tiedrank!(rks::AbstractArray, x::AbstractArray, p::IntegerArray)
    n = _check_randparams(rks, x, p)
    if n > 0
        v = x[p[1]]
        s = 1  # starting index of current range
        e = 2  # pass-by-end index of current range
        while e <= n
            cx = x[p[e]]
            if cx != v
                ar = (s + e - 1) / 2
                for i = s : e-1
                    rks[p[i]] = ar
                end
                s = e
                v = cx
            end
            e += 1
        end
        ar = (s + n) / 2
        for i = s : n
            rks[p[i]] = ar
        end
    end
    return rks
end
""" """ tiedrank(x::AbstractArray; lt = isless, rev::Bool = false) =
    tiedrank!(Array{Float64}(undef, size(x)), x, sortperm(x; lt = lt, rev = rev))
for (f, f!, S) in zip([:ordinalrank, :competerank, :denserank, :tiedrank],
                      [:ordinalrank!, :competerank!, :denserank!, :tiedrank!],
                      [Int, Int, Int, Float64])
    @eval begin
        function $f(x::AbstractArray{>: Missing}; lt = isless, rev::Bool = false)
            inds = findall(!ismissing, x)
            isempty(inds) && return missings($S, size(x))
            xv = disallowmissing(view(x, inds))
            sp = sortperm(xv; lt = lt, rev = rev)
            rks = missings($S, length(x))
            $(f!)(view(rks, inds), xv, sp)
            rks
        end
    end
end