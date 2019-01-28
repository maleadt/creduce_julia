import .RFunctions:
    betapdf,
    betalogpdf,
    betacdf,
    betaccdf,
    betalogcdf,
    betalogccdf,
    betainvcdf,
    betainvccdf,
    betainvlogcdf,
    betainvlogccdf
betapdf(α::Real, β::Real, x::Number) = x^(α - 1) * (1 - x)^(β - 1) / beta(α, β)
betalogpdf(α::Real, β::Real, x::Number) = (α - 1) * log(x) + (β - 1) * log1p(-x) - lbeta(α, β)
