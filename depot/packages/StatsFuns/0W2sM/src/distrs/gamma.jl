import .RFunctions:
    gammapdf,
    gammalogpdf,
    gammacdf,
    gammaccdf,
    gammalogcdf,
    gammalogccdf,
    gammainvcdf,
    gammainvccdf,
    gammainvlogcdf,
    gammainvlogccdf
gammapdf(k::Real, θ::Real, x::Number) = 1 / (gamma(k) * θ^k) * x^(k - 1) * exp(-x / θ)
gammalogpdf(k::Real, θ::Real, x::Number) = -lgamma(k) - k * log(θ) + (k - 1) * log(x) - x / θ
