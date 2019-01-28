import .RFunctions:
    fdistpdf,
    fdistlogpdf,
    fdistcdf,
    fdistccdf,
    fdistlogcdf,
    fdistlogccdf,
    fdistinvcdf,
    fdistinvccdf,
    fdistinvlogcdf,
    fdistinvlogccdf
fdistpdf(d1::Real, d2::Real, x::Number) = sqrt((d1 * x)^d1 * d2^d2 / (d1 * x + d2)^(d1 + d2)) / (x * beta(d1 / 2, d2 / 2))
fdistlogpdf(d1::Real, d2::Real, x::Number) = (d1 * log(d1 * x) + d2 * log(d2) - (d1 + d2) * log(d1 * x + d2)) / 2 - log(x) - lbeta(d1 / 2, d2 / 2)
