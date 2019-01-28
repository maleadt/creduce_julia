using Random: RangeGenerator, Random.GLOBAL_RNG
function direct_sample!(rng::AbstractRNG, a::UnitRange, x::AbstractArray)
    s = RangeGenerator(1:length(a))
    b = a[1] - 1
    if b == 0
        for i = 1:length(x)
            @inbounds x[i] = rand(rng, s)
        end
    else
        for i = 1:length(x)
            @inbounds x[i] = b + rand(rng, s)
        end
    end
    return x
end
direct_sample!(a::UnitRange, x::AbstractArray) = direct_sample!(Random.GLOBAL_RNG, a, x)
""" """ function direct_sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    s = RangeGenerator(1:length(a))
    for i = 1:length(x)
        @inbounds x[i] = a[rand(rng, s)]
    end
    return x
end
direct_sample!(a::AbstractArray, x::AbstractArray) = direct_sample!(Random.GLOBAL_RNG, a, x)
""" """ function samplepair(rng::AbstractRNG, n::Int)
    i1 = rand(rng, 1:n)
    i2 = rand(rng, 1:n-1)
    return (i1, ifelse(i2 == i1, n, i2))
end
samplepair(n::Int) = samplepair(Random.GLOBAL_RNG, n)
""" """ function samplepair(rng::AbstractRNG, a::AbstractArray)
    i1, i2 = samplepair(rng, length(a))
    return a[i1], a[i2]
end
samplepair(a::AbstractArray) = samplepair(Random.GLOBAL_RNG, a)
""" """ function knuths_sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray;
                        initshuffle::Bool=true)
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")
    for i = 1:k
        @inbounds x[i] = a[i]
    end
    if initshuffle
        @inbounds for j = 1:k
            l = rand(rng, j:k)
            if l != j
                t = x[j]
                x[j] = x[l]
                x[l] = t
            end
        end
    end
    s = RangeGenerator(1:k)
    for i = k+1:n
        if rand(rng) * i < k  # keep it with probability k / i
            @inbounds x[rand(rng, s)] = a[i]
        end
    end
    return x
end
knuths_sample!(a::AbstractArray, x::AbstractArray; initshuffle::Bool=true) =
    knuths_sample!(Random.GLOBAL_RNG, a, x; initshuffle=initshuffle)
""" """ function fisher_yates_sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")
    inds = Vector{Int}(undef, n)
    for i = 1:n
        @inbounds inds[i] = i
    end
    @inbounds for i = 1:k
        j = rand(rng, i:n)
        t = inds[j]
        inds[j] = inds[i]
        inds[i] = t
        x[i] = a[t]
    end
    return x
end
fisher_yates_sample!(a::AbstractArray, x::AbstractArray) =
    fisher_yates_sample!(Random.GLOBAL_RNG, a, x)
""" """ function self_avoid_sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")
    s = Set{Int}()
    sizehint!(s, k)
    rgen = RangeGenerator(1:n)
    idx = rand(rng, rgen)
    x[1] = a[idx]
    push!(s, idx)
    for i = 2:k
        idx = rand(rng, rgen)
        while idx in s
            idx = rand(rng, rgen)
        end
        x[i] = a[idx]
        push!(s, idx)
    end
    return x
end
self_avoid_sample!(a::AbstractArray, x::AbstractArray) =
    self_avoid_sample!(Random.GLOBAL_RNG, a, x)
""" """ function seqsample_a!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")
    i = 0
    j = 0
    while k > 1
        u = rand(rng)
        q = (n - k) / n
        while q > u  # skip
            i += 1
            n -= 1
            q *= (n - k) / n
        end
        @inbounds x[j+=1] = a[i+=1]
        n -= 1
        k -= 1
    end
    if k > 0  # checking k > 0 is necessary: x can be empty
        s = trunc(Int, n * rand(rng))
        x[j+1] = a[i+(s+1)]
    end
    return x
end
seqsample_a!(a::AbstractArray, x::AbstractArray) = seqsample_a!(Random.GLOBAL_RNG, a, x)
""" """ function seqsample_c!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray)
    n = length(a)
    k = length(x)
    k <= n || error("length(x) should not exceed length(a)")
    i = 0
    j = 0
    while k > 1
        l = n - k + 1
        minv = l
        u = n
        while u >= l
            v = u * rand(rng)
            if v < minv
                minv = v
            end
            u -= 1
        end
        s = trunc(Int, minv) + 1
        x[j+=1] = a[i+=s]
        n -= s
        k -= 1
    end
    if k > 0
        s = trunc(Int, n * rand(rng))
        x[j+1] = a[i+(s+1)]
    end
    return x
end
seqsample_c!(a::AbstractArray, x::AbstractArray) = seqsample_c!(Random.GLOBAL_RNG, a, x)
""" """ sample(rng::AbstractRNG, a::AbstractArray) = a[rand(rng, 1:length(a))]
sample(a::AbstractArray) = sample(Random.GLOBAL_RNG, a)
""" """ function sample!(rng::AbstractRNG, a::AbstractArray, x::AbstractArray;
                 replace::Bool=true, ordered::Bool=false)
    n = length(a)
    k = length(x)
    k == 0 && return x
    if replace  # with replacement
        if ordered
            sort!(direct_sample!(rng, a, x))
        else
            direct_sample!(rng, a, x)
        end
    else  # without replacement
        k <= n || error("Cannot draw more samples without replacement.")
        if ordered
            if n > 10 * k * k
                seqsample_c!(rng, a, x)
            else
                seqsample_a!(rng, a, x)
            end
        else
            if k == 1
                @inbounds x[1] = sample(rng, a)
            elseif k == 2
                @inbounds (x[1], x[2]) = samplepair(rng, a)
            elseif n < k * 24
                fisher_yates_sample!(rng, a, x)
            else
                self_avoid_sample!(rng, a, x)
            end
        end
    end
    return x
end
sample!(a::AbstractArray, x::AbstractArray; replace::Bool=true, ordered::Bool=false) =
    sample!(Random.GLOBAL_RNG, a, x; replace=replace, ordered=ordered)
""" """ function sample(rng::AbstractRNG, a::AbstractArray{T}, n::Integer;
                replace::Bool=true, ordered::Bool=false) where T
    sample!(rng, a, Vector{T}(undef, n); replace=replace, ordered=ordered)
end
sample(a::AbstractArray, n::Integer; replace::Bool=true, ordered::Bool=false) =
    sample(Random.GLOBAL_RNG, a, n; replace=replace, ordered=ordered)
""" """ function sample(rng::AbstractRNG, a::AbstractArray{T}, dims::Dims;
                replace::Bool=true, ordered::Bool=false) where T
    sample!(rng, a, Array{T}(undef, dims), rng; replace=replace, ordered=ordered)
end
sample(a::AbstractArray, dims::Dims; replace::Bool=true, ordered::Bool=false) =
    sample(Random.GLOBAL_RNG, a, dims; replace=replace, ordered=ordered)
""" """ function sample(rng::AbstractRNG, wv::AbstractWeights)
    t = rand(rng) * sum(wv)
    w = values(wv)
    n = length(w)
    i = 1
    cw = w[1]
    while cw < t && i < n
        i += 1
        @inbounds cw += w[i]
    end
    return i
end
sample(wv::AbstractWeights) = sample(Random.GLOBAL_RNG, wv)
sample(rng::AbstractRNG, a::AbstractArray, wv::AbstractWeights) = a[sample(rng, wv)]
sample(a::AbstractArray, wv::AbstractWeights) = sample(Random.GLOBAL_RNG, a, wv)
""" """ function direct_sample!(rng::AbstractRNG, a::AbstractArray,
                        wv::AbstractWeights, x::AbstractArray)
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("Inconsistent lengths."))
    for i = 1:length(x)
        x[i] = a[sample(rng, wv)]
    end
    return x
end
direct_sample!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    direct_sample!(Random.GLOBAL_RNG, a, wv, x)
function make_alias_table!(w::AbstractVector{Float64}, wsum::Float64,
                           a::AbstractVector{Float64},
                           alias::AbstractVector{Int})
    n = length(w)
    length(a) == length(alias) == n ||
        throw(DimensionMismatch("Inconsistent array lengths."))
    ac = n / wsum
    for i = 1:n
        @inbounds a[i] = w[i] * ac
    end
    larges = Vector{Int}(undef, n)
    smalls = Vector{Int}(undef, n)
    kl = 0  # actual number of larges
    ks = 0  # actual number of smalls
    for i = 1:n
        @inbounds ai = a[i]
        if ai > 1.0
            larges[kl+=1] = i  # push to larges
        elseif ai < 1.0
            smalls[ks+=1] = i  # push to smalls
        end
    end
    while kl > 0 && ks > 0
        s = smalls[ks]; ks -= 1  # pop from smalls
        l = larges[kl]; kl -= 1  # pop from larges
        @inbounds alias[s] = l
        @inbounds al = a[l] = (a[l] - 1.0) + a[s]
        if al > 1.0
            larges[kl+=1] = l  # push to larges
        else
            smalls[ks+=1] = l  # push to smalls
        end
    end
    for i = 1:ks
        @inbounds a[smalls[i]] = 1.0
    end
    nothing
end
""" """ function alias_sample!(rng::AbstractRNG, a::AbstractArray, wv::AbstractWeights, x::AbstractArray)
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("Inconsistent lengths."))
    ap = Vector{Float64}(undef, n)
    alias = Vector{Int}(undef, n)
    make_alias_table!(values(wv), sum(wv), ap, alias)
    s = RangeGenerator(1:n)
    for i = 1:length(x)
        j = rand(rng, s)
        x[i] = rand(rng) < ap[j] ? a[j] : a[alias[j]]
    end
    return x
end
alias_sample!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    alias_sample!(Random.GLOBAL_RNG, a, wv, x)
""" """ function naive_wsample_norep!(rng::AbstractRNG, a::AbstractArray,
                              wv::AbstractWeights, x::AbstractArray)
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("Inconsistent lengths."))
    k = length(x)
    w = Vector{Float64}(undef, n)
    copyto!(w, values(wv))
    wsum = sum(wv)
    for i = 1:k
        u = rand(rng) * wsum
        j = 1
        c = w[1]
        while c < u && j < n
            @inbounds c += w[j+=1]
        end
        @inbounds x[i] = a[j]
        @inbounds wsum -= w[j]
        @inbounds w[j] = 0.0
    end
    return x
end
naive_wsample_norep!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    naive_wsample_norep!(Random.GLOBAL_RNG, a, wv, x)
""" """ function efraimidis_a_wsample_norep!(rng::AbstractRNG, a::AbstractArray,
                                     wv::AbstractWeights, x::AbstractArray)
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("a and wv must be of same length (got $n and $(length(wv)))."))
    k = length(x)
    keys = randexp(rng, n)
    for i in 1:n
        @inbounds keys[i] = wv.values[i]/keys[i]
    end
    index = sortperm(keys; alg = PartialQuickSort(k), rev = true)
    for i in 1:k
        @inbounds x[i] = a[index[i]]
    end
    return x
end
efraimidis_a_wsample_norep!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    efraimidis_a_wsample_norep!(Random.GLOBAL_RNG, a, wv, x)
""" """ function efraimidis_ares_wsample_norep!(rng::AbstractRNG, a::AbstractArray,
                                        wv::AbstractWeights, x::AbstractArray)
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("a and wv must be of same length (got $n and $(length(wv)))."))
    k = length(x)
    k > 0 || return x
    pq = Vector{Pair{Float64,Int}}(undef, k)
    i = 0
    s = 0
    @inbounds for _s in 1:n
        s = _s
        w = wv.values[s]
        w < 0 && error("Negative weight found in weight vector at index $s")
        if w > 0
            i += 1
            pq[i] = (w/randexp(rng) => s)
        end
        i >= k && break
    end
    i < k && throw(DimensionMismatch("wv must have at least $k strictly positive entries (got $i)"))
    heapify!(pq)
    @inbounds threshold = pq[1].first
    @inbounds for i in s+1:n
        w = wv.values[i]
        w < 0 && error("Negative weight found in weight vector at index $i")
        w > 0 || continue
        key = w/randexp(rng)
        if key > threshold
            pq[1] = (key => i)
            percolate_down!(pq, 1)
            threshold = pq[1].first
        end
    end
    @inbounds for i in k:-1:1
        x[i] = a[heappop!(pq).second]
    end
    return x
end
efraimidis_ares_wsample_norep!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    efraimidis_ares_wsample_norep!(Random.GLOBAL_RNG, a, wv, x)
""" """ function efraimidis_aexpj_wsample_norep!(rng::AbstractRNG, a::AbstractArray,
                                         wv::AbstractWeights, x::AbstractArray)
    n = length(a)
    length(wv) == n || throw(DimensionMismatch("a and wv must be of same length (got $n and $(length(wv)))."))
    k = length(x)
    k > 0 || return x
    pq = Vector{Pair{Float64,Int}}(undef, k)
    i = 0
    s = 0
    @inbounds for _s in 1:n
        s = _s
        w = wv.values[s]
        w < 0 && error("Negative weight found in weight vector at index $s")
        if w > 0
            i += 1
            pq[i] = (w/randexp(rng) => s)
        end
        i >= k && break
    end
    i < k && throw(DimensionMismatch("wv must have at least $k strictly positive entries (got $i)"))
    heapify!(pq)
    @inbounds threshold = pq[1].first
    X = threshold*randexp(rng)
    @inbounds for i in s+1:n
        w = wv.values[i]
        w < 0 && error("Negative weight found in weight vector at index $i")
        w > 0 || continue
        X -= w
        X <= 0 || continue
        t = exp(-w/threshold)
        pq[1] = (-w/log(t+rand(rng)*(1-t)) => i)
        percolate_down!(pq, 1)
        threshold = pq[1].first
        X = threshold * randexp(rng)
    end
    @inbounds for i in k:-1:1
        x[i] = a[heappop!(pq).second]
    end
    return x
end
efraimidis_aexpj_wsample_norep!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    efraimidis_aexpj_wsample_norep!(Random.GLOBAL_RNG, a, wv, x)
function sample!(rng::AbstractRNG, a::AbstractArray, wv::AbstractWeights, x::AbstractArray;
                 replace::Bool=true, ordered::Bool=false)
    n = length(a)
    k = length(x)
    if replace
        if ordered
            sort!(direct_sample!(rng, a, wv, x))
        else
            if n < 40
                direct_sample!(rng, a, wv, x)
            else
                t = ifelse(n < 500, 64, 32)
                if k < t
                    direct_sample!(rng, a, wv, x)
                else
                    alias_sample!(rng, a, wv, x)
                end
            end
        end
    else
        k <= n || error("Cannot draw $n samples from $k samples without replacement.")
        efraimidis_aexpj_wsample_norep!(rng, a, wv, x)
        if ordered
            sort!(x)
        end
    end
    return x
end
sample!(a::AbstractArray, wv::AbstractWeights, x::AbstractArray) =
    sample!(Random.GLOBAL_RNG, a, wv, x)
sample(rng::AbstractRNG, a::AbstractArray{T}, wv::AbstractWeights, n::Integer;
       replace::Bool=true, ordered::Bool=false) where {T} =
    sample!(rng, a, wv, Vector{T}(undef, n); replace=replace, ordered=ordered)
sample(a::AbstractArray, wv::AbstractWeights, n::Integer;
       replace::Bool=true, ordered::Bool=false) =
    sample(Random.GLOBAL_RNG, a, wv, n; replace=replace, ordered=ordered)
sample(rng::AbstractRNG, a::AbstractArray{T}, wv::AbstractWeights, dims::Dims;
       replace::Bool=true, ordered::Bool=false) where {T} =
    sample!(rng, a, wv, Array{T}(undef, dims); replace=replace, ordered=ordered)
sample(a::AbstractArray, wv::AbstractWeights, dims::Dims;
       replace::Bool=true, ordered::Bool=false) =
    sample(Random.GLOBAL_RNG, a, wv, dims; replace=replace, ordered=ordered)
""" """ wsample!(rng::AbstractRNG, a::AbstractArray, w::RealVector, x::AbstractArray;
         replace::Bool=true, ordered::Bool=false) =
    sample!(rng, a, weights(w), x; replace=replace, ordered=ordered)
wsample!(a::AbstractArray, w::RealVector, x::AbstractArray;
         replace::Bool=true, ordered::Bool=false) =
    sample!(Random.GLOBAL_RNG, a, weights(w), x; replace=replace, ordered=ordered)
""" """ wsample(rng::AbstractRNG, w::RealVector) = sample(rng, weights(w))
wsample(w::RealVector) = wsample(Random.GLOBAL_RNG, w)
wsample(rng::AbstractRNG, a::AbstractArray, w::RealVector) = sample(rng, a, weights(w))
wsample(a::AbstractArray, w::RealVector) = wsample(Random.GLOBAL_RNG, a, w)
""" """ wsample(rng::AbstractRNG, a::AbstractArray{T}, w::RealVector, n::Integer;
        replace::Bool=true, ordered::Bool=false) where {T} =
    wsample!(rng, a, w, Vector{T}(undef, n); replace=replace, ordered=ordered)
wsample(a::AbstractArray, w::RealVector, n::Integer;
        replace::Bool=true, ordered::Bool=false) =
    wsample(Random.GLOBAL_RNG, a, w, n; replace=replace, ordered=ordered)
""" """ wsample(rng::AbstractRNG, a::AbstractArray{T}, w::RealVector, dims::Dims;
        replace::Bool=true, ordered::Bool=false) where {T} =
    wsample!(rng, a, w, Array{T}(undef, dims); replace=replace, ordered=ordered)
wsample(a::AbstractArray, w::RealVector, dims::Dims;
        replace::Bool=true, ordered::Bool=false) =
    wsample(Random.GLOBAL_RNG, a, w, dims; replace=replace, ordered=ordered)
