function demean_col!(z::AbstractVector{T}, x::AbstractMatrix{T}, j::Int, demean::Bool) where T<:RealFP
end
function crosscor(x::AbstractVector{T}, y::AbstractMatrix{T}, lags::IntegerVector; demean::Bool=true) where T<:Real
    for j = 1 : size(X,2)
        for l = 1 : mk
            for i = 1+l:lx
                tmpX[i,l+1] = X[i-l,j]
            end
        end
    end
end
function pacf_yulewalker!(r::RealMatrix, X::AbstractMatrix{T}, lags::IntegerVector, mk::Integer) where T<:RealFP
    for j = 1 : size(X,2)
        for i = 1 : length(lags)
        end
    end
    if method == :regression
    end
end
function pacf(x::AbstractVector{T}, lags::IntegerVector; method::Symbol=:regression) where T<:Real
end
