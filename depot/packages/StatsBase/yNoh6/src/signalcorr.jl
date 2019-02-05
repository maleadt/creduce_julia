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
