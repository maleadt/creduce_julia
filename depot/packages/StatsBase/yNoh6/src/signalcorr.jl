default_laglen(lx::Int) = min(lx-1, round(Int,10*log10(lx)))
check_lags(lx::Int, lags::AbstractVector) = (maximum(lags) < lx || error("lags must be less than the sample length."))
function demean_col!(z::AbstractVector{T}, x::AbstractMatrix{T}, j::Int, demean::Bool) where T<:RealFP
    m = size(x, 1)
    @assert m == length(z)
    b = m * (j-1)
    if demean
        s = zero(T)
        for i = 1 : m
            s += x[b + i]
        end
        mv = s / m
        for i = 1 : m
            z[i] = x[b + i] - mv
        end
    else
        copyto!(z, 1, x, b+1, m)
    end
    z
end
default_autolags(lx::Int) = 0 : default_laglen(lx)
_autodot(x::AbstractVector{<:RealFP}, lx::Int, l::Int) = dot(x, 1:lx-l, x, 1+l:lx)
""" """ function autocov!(r::RealVector, x::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = length(x)
    m = length(lags)
    length(r) == m || throw(DimensionMismatch())
    check_lags(lx, lags)
    z::Vector{T} = demean ? x .- mean(x) : x
    for k = 1 : m  # foreach lag value
        r[k] = _autodot(z, lx, lags[k]) / lx
    end
    return r
end
function autocov!(r::RealMatrix, x::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = size(x, 1)
    ns = size(x, 2)
    m = length(lags)
    size(r) == (m, ns) || throw(DimensionMismatch())
    check_lags(lx, lags)
    z = Vector{T}(undef, lx)
    for j = 1 : ns
        demean_col!(z, x, j, demean)
        for k = 1 : m
            r[k,j] = _autodot(z, lx, lags[k]) / lx
        end
    end
    return r
end
""" """ function autocov(x::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    autocov!(Vector{fptype(T)}(undef, length(lags)), float(x), lags; demean=demean)
end
function autocov(x::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    autocov!(Matrix{fptype(T)}(undef, length(lags), size(x,2)), float(x), lags; demean=demean)
end
autocov(x::AbstractVecOrMat{<:Real}; demean::Bool=true) = autocov(x, default_autolags(size(x,1)); demean=demean)
""" """ function autocor!(r::RealVector, x::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = length(x)
    m = length(lags)
    length(r) == m || throw(DimensionMismatch())
    check_lags(lx, lags)
    z::Vector{T} = demean ? x .- mean(x) : x
    zz = dot(z, z)
    for k = 1 : m  # foreach lag value
        r[k] = _autodot(z, lx, lags[k]) / zz
    end
    return r
end
function autocor!(r::RealMatrix, x::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = size(x, 1)
    ns = size(x, 2)
    m = length(lags)
    size(r) == (m, ns) || throw(DimensionMismatch())
    check_lags(lx, lags)
    z = Vector{T}(undef, lx)
    for j = 1 : ns
        demean_col!(z, x, j, demean)
        zz = dot(z, z)
        for k = 1 : m
            r[k,j] = _autodot(z, lx, lags[k]) / zz
        end
    end
    return r
end
""" """ function autocor(x::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    autocor!(Vector{fptype(T)}(undef, length(lags)), float(x), lags; demean=demean)
end
function autocor(x::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    autocor!(Matrix{fptype(T)}(undef, length(lags), size(x,2)), float(x), lags; demean=demean)
end
autocor(x::AbstractVecOrMat{<:Real}; demean::Bool=true) = autocor(x, default_autolags(size(x,1)); demean=demean)
default_crosslags(lx::Int) = (l=default_laglen(lx); -l:l)
_crossdot(x::AbstractVector{T}, y::AbstractVector{T}, lx::Int, l::Int) where {T<:RealFP} =
    (l >= 0 ? dot(x, 1:lx-l, y, 1+l:lx) : dot(x, 1-l:lx, y, 1:lx+l))
""" """ function crosscov!(r::RealVector, x::AbstractVector{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = length(x)
    m = length(lags)
    (length(y) == lx && length(r) == m) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zx::Vector{T} = demean ? x .- mean(x) : x
    zy::Vector{T} = demean ? y .- mean(y) : y
    for k = 1 : m  # foreach lag value
        r[k] = _crossdot(zx, zy, lx, lags[k]) / lx
    end
    return r
end
function crosscov!(r::RealMatrix, x::AbstractMatrix{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = size(x, 1)
    ns = size(x, 2)
    m = length(lags)
    (length(y) == lx && size(r) == (m, ns)) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zx = Vector{T}(undef, lx)
    zy::Vector{T} = demean ? y .- mean(y) : y
    for j = 1 : ns
        demean_col!(zx, x, j, demean)
        for k = 1 : m
            r[k,j] = _crossdot(zx, zy, lx, lags[k]) / lx
        end
    end
    return r
end
function crosscov!(r::RealMatrix, x::AbstractVector{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = length(x)
    ns = size(y, 2)
    m = length(lags)
    (size(y, 1) == lx && size(r) == (m, ns)) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zx::Vector{T} = demean ? x .- mean(x) : x
    zy = Vector{T}(undef, lx)
    for j = 1 : ns
        demean_col!(zy, y, j, demean)
        for k = 1 : m
            r[k,j] = _crossdot(zx, zy, lx, lags[k]) / lx
        end
    end
    return r
end
function crosscov!(r::AbstractArray{T,3}, x::AbstractMatrix{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = size(x, 1)
    nx = size(x, 2)
    ny = size(y, 2)
    m = length(lags)
    (size(y, 1) == lx && size(r) == (m, nx, ny)) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zxs = Vector{T}[]
    sizehint!(zxs, nx)
    for j = 1 : nx
        xj = x[:,j]
        if demean
            mv = mean(xj)
            for i = 1 : lx
                xj[i] -= mv
            end
        end
        push!(zxs, xj)
    end
    zx = Vector{T}(undef, lx)
    zy = Vector{T}(undef, lx)
    for j = 1 : ny
        demean_col!(zy, y, j, demean)
        for i = 1 : nx
            zx = zxs[i]
            for k = 1 : m
                r[k,i,j] = _crossdot(zx, zy, lx, lags[k]) / lx
            end
        end
    end
    return r
end
""" """ function crosscov(x::AbstractVector{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscov!(Vector{fptype(T)}(undef, length(lags)), float(x), float(y), lags; demean=demean)
end
function crosscov(x::AbstractMatrix{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscov!(Matrix{fptype(T)}(undef, length(lags), size(x,2)), float(x), float(y), lags; demean=demean)
end
function crosscov(x::AbstractVector{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscov!(Matrix{fptype(T)}(undef, length(lags), size(y,2)), float(x), float(y), lags; demean=demean)
end
function crosscov(x::AbstractMatrix{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscov!(Array{fptype(T),3}(undef, length(lags), size(x,2), size(y,2)), float(x), float(y), lags; demean=demean)
end
crosscov(x::AbstractVecOrMat{T}, y::AbstractVecOrMat{T}; demean::Bool=true) where {T<:Real} = crosscov(x, y, default_crosslags(size(x,1)); demean=demean)
""" """ function crosscor!(r::RealVector, x::AbstractVector{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = length(x)
    m = length(lags)
    (length(y) == lx && length(r) == m) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zx::Vector{T} = demean ? x .- mean(x) : x
    zy::Vector{T} = demean ? y .- mean(y) : y
    sc = sqrt(dot(zx, zx) * dot(zy, zy))
    for k = 1 : m  # foreach lag value
        r[k] = _crossdot(zx, zy, lx, lags[k]) / sc
    end
    return r
end
function crosscor!(r::RealMatrix, x::AbstractMatrix{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = size(x, 1)
    ns = size(x, 2)
    m = length(lags)
    (length(y) == lx && size(r) == (m, ns)) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zx = Vector{T}(undef, lx)
    zy::Vector{T} = demean ? y .- mean(y) : y
    yy = dot(zy, zy)
    for j = 1 : ns
        demean_col!(zx, x, j, demean)
        sc = sqrt(dot(zx, zx) * yy)
        for k = 1 : m
            r[k,j] = _crossdot(zx, zy, lx, lags[k]) / sc
        end
    end
    return r
end
function crosscor!(r::RealMatrix, x::AbstractVector{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = length(x)
    ns = size(y, 2)
    m = length(lags)
    (size(y, 1) == lx && size(r) == (m, ns)) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zx::Vector{T} = demean ? x .- mean(x) : x
    zy = Vector{T}(undef, lx)
    xx = dot(zx, zx)
    for j = 1 : ns
        demean_col!(zy, y, j, demean)
        sc = sqrt(xx * dot(zy, zy))
        for k = 1 : m
            r[k,j] = _crossdot(zx, zy, lx, lags[k]) / sc
        end
    end
    return r
end
function crosscor!(r::AbstractArray{T,3}, x::AbstractMatrix{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:RealFP
    lx = size(x, 1)
    nx = size(x, 2)
    ny = size(y, 2)
    m = length(lags)
    (size(y, 1) == lx && size(r) == (m, nx, ny)) || throw(DimensionMismatch())
    check_lags(lx, lags)
    zxs = Vector{T}[]
    sizehint!(zxs, nx)
    xxs = Vector{T}(undef, nx)
    for j = 1 : nx
        xj = x[:,j]
        if demean
            mv = mean(xj)
            for i = 1 : lx
                xj[i] -= mv
            end
        end
        push!(zxs, xj)
        xxs[j] = dot(xj, xj)
    end
    zx = Vector{T}(undef, lx)
    zy = Vector{T}(undef, lx)
    for j = 1 : ny
        demean_col!(zy, y, j, demean)
        yy = dot(zy, zy)
        for i = 1 : nx
            zx = zxs[i]
            sc = sqrt(xxs[i] * yy)
            for k = 1 : m
                r[k,i,j] = _crossdot(zx, zy, lx, lags[k]) / sc
            end
        end
    end
    return r
end
""" """ function crosscor(x::AbstractVector{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscor!(Vector{fptype(T)}(undef, length(lags)), float(x), float(y), lags; demean=demean)
end
function crosscor(x::AbstractMatrix{T}, y::AbstractVector{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscor!(Matrix{fptype(T)}(undef, length(lags), size(x,2)), float(x), float(y), lags; demean=demean)
end
function crosscor(x::AbstractVector{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscor!(Matrix{fptype(T)}(undef, length(lags), size(y,2)), float(x), float(y), lags; demean=demean)
end
function crosscor(x::AbstractMatrix{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    crosscor!(Array{fptype(T),3}(undef, length(lags), size(x,2), size(y,2)), float(x), float(y), lags; demean=demean)
end
crosscor(x::AbstractVecOrMat{T}, y::AbstractVecOrMat{T}; demean::Bool=true) where {T<:Real} = crosscor(x, y, default_crosslags(size(x,1)); demean=demean)
function pacf_regress!(r::RealMatrix, X::AbstractMatrix{T}, lags::IntegerVector, mk::Integer) where T<:RealFP
    lx = size(X, 1)
    tmpX = ones(T, lx, mk + 1)
    for j = 1 : size(X,2)
        for l = 1 : mk
            for i = 1+l:lx
                tmpX[i,l+1] = X[i-l,j]
            end
        end
        for i = 1 : length(lags)
            l = lags[i]
            sX = view(tmpX, 1+l:lx, 1:l+1)
            r[i,j] = l == 0 ? 1 : (cholesky!(sX'sX, Val(false)) \ (sX'view(X, 1+l:lx, j)))[end]
        end
    end
    r
end
function pacf_yulewalker!(r::RealMatrix, X::AbstractMatrix{T}, lags::IntegerVector, mk::Integer) where T<:RealFP
    tmp = Vector{T}(undef, mk)
    for j = 1 : size(X,2)
        acfs = autocor(X[:,j], 1:mk)
        for i = 1 : length(lags)
            l = lags[i]
            r[i,j] = l == 0 ? 1 : l == 1 ? acfs[i] : -durbin!(view(acfs, 1:l), tmp)[l]
        end
    end
end
""" """ function pacf!(r::RealMatrix, X::AbstractMatrix{T}, lags::IntegerVector; method::Symbol=:regression) where T<:RealFP
    lx = size(X, 1)
    m = length(lags)
    minlag, maxlag = extrema(lags)
    (0 <= minlag && 2maxlag < lx) || error("Invalid lag value.")
    size(r) == (m, size(X,2)) || throw(DimensionMismatch())
    if method == :regression
        pacf_regress!(r, X, lags, maxlag)
    elseif method == :yulewalker
        pacf_yulewalker!(r, X, lags, maxlag)
    else
        error("Invalid method: $method")
    end
    return r
end
""" """ function pacf(X::AbstractMatrix{T}, lags::IntegerVector; method::Symbol=:regression) where T<:Real
    pacf!(Matrix{fptype(T)}(undef, length(lags), size(X,2)), float(X), lags; method=method)
end
function pacf(x::AbstractVector{T}, lags::IntegerVector; method::Symbol=:regression) where T<:Real
    vec(pacf(reshape(x, length(x), 1), lags, method=method))
end
