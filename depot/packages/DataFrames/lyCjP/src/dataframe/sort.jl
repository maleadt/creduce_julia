""" """ function Base.sort!(df::DataFrame, cols_new=[]; cols=[], alg=nothing,
                    lt=isless, by=identity, rev=false, order=Forward)
    if !(isa(by, Function) || eltype(by) <: Function)
        msg = "'by' must be a Function or a vector of Functions. Perhaps you wanted 'cols'."
        throw(ArgumentError(msg))
    end
    if cols != []
        Base.depwarn("sort!(df, cols=cols) is deprecated, use sort!(df, cols) instead",
                     :sort!)
        cols_new = cols
    end
    ord = ordering(df, cols_new, lt, by, rev, order)
    _alg = Sort.defalg(df, ord; alg=alg, cols=cols_new)
    sort!(df, _alg, ord)
end
function Base.sort!(df::DataFrame, a::Base.Sort.Algorithm, o::Base.Sort.Ordering)
    p = sortperm(df, a, o)
    pp = similar(p)
    c = _columns(df)
    for (i,col) in enumerate(c)
        if any(j -> c[j]===col, 1:i-1)
            continue
        end
        copyto!(pp,p)
        Base.permute!!(col, pp)
    end
    df
end
