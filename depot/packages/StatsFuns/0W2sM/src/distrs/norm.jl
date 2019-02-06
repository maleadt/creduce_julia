xval(μ::Real, σ::Real, z::Number) = μ + σ * z
zval(μ::Real, σ::Real, x::Number) = (x - μ) / σ
normpdf(z::Number) = exp(-abs2(z)/2) * invsqrt2π
normpdf(μ::Real, σ::Real, x::Number) = normpdf(zval(μ, σ, x)) / σ
normlogpdf(z::Number) = -(abs2(z) + log2π)/2
normlogpdf(μ::Real, σ::Real, x::Number) = normlogpdf(zval(μ, σ, x)) - log(σ)
normcdf(z::Number) = erfc(-z * invsqrt2)/2
normcdf(μ::Real, σ::Real, x::Number) = normcdf(zval(μ, σ, x))
normccdf(z::Number) = erfc(z * invsqrt2)/2
normccdf(μ::Real, σ::Real, x::Number) = normccdf(zval(μ, σ, x))
normlogcdf(z::Number) = z < -1.0 ?
    log(erfcx(-z * invsqrt2)/2) - abs2(z)/2 :
    log1p(-erfc(z * invsqrt2)/2)
normlogcdf(μ::Real, σ::Real, x::Number) = normlogcdf(zval(μ, σ, x))
normlogccdf(z::Number) = z > 1.0 ?
    log(erfcx(z * invsqrt2)/2) - abs2(z)/2 :
    log1p(-erfc(-z * invsqrt2)/2)
normlogccdf(μ::Real, σ::Real, x::Number) = normlogccdf(zval(μ, σ, x))
norminvcdf(p::Real) = -erfcinv(2*p) * sqrt2
norminvcdf(μ::Real, σ::Real, p::Real) = xval(μ, σ, norminvcdf(p))
norminvccdf(p::Real) = erfcinv(2*p) * sqrt2
norminvccdf(μ::Real, σ::Real, p::Real) = xval(μ, σ, norminvccdf(p))
norminvlogcdf(lp::Union{Float16,Float32}) = convert(typeof(lp), _norminvlogcdf_impl(Float64(lp)))
norminvlogcdf(lp::Real) = _norminvlogcdf_impl(Float64(lp))
norminvlogcdf(μ::Real, σ::Real, lp::Real) = xval(μ, σ, norminvlogcdf(lp))
norminvlogccdf(lp::Union{Float16,Float32}) = convert(typeof(lp), -_norminvlogcdf_impl(Float64(lp)))
norminvlogccdf(lp::Real) = -_norminvlogcdf_impl(Float64(lp))
norminvlogccdf(μ::Real, σ::Real, lp::Real) = xval(μ, σ, norminvlogccdf(lp))
function _norminvlogcdf_impl(lp::Float64)
    if isfinite(lp) && lp < 0.0
        q = exp(lp) - 0.5
        if abs(q) <= 0.425
            _qnorm_ker1(q)
        else
            r = sqrt(q < 0 ? -lp : -log1mexp(lp))
            return copysign(_qnorm_ker2(r), q)
        end
    elseif lp >= 0.0
        lp == 0.0 ? Inf : NaN
    else # lp is -Inf or NaN
        lp
    end
end
function _qnorm_ker1(q::Float64)
    r = 0.180625 - q*q
    return q * @horner(r,
                       3.38713_28727_96366_6080e0,
                       1.33141_66789_17843_7745e2,
                       1.97159_09503_06551_4427e3,
                       1.37316_93765_50946_1125e4,
                       4.59219_53931_54987_1457e4,
                       6.72657_70927_00870_0853e4,
                       3.34305_75583_58812_8105e4,
                       2.50908_09287_30122_6727e3) /
    @horner(r,
            1.0,
            4.23133_30701_60091_1252e1,
            6.87187_00749_20579_0830e2,
            5.39419_60214_24751_1077e3,
            2.12137_94301_58659_5867e4,
            3.93078_95800_09271_0610e4,
            2.87290_85735_72194_2674e4,
            5.22649_52788_52854_5610e3)
end
function _qnorm_ker2(r::Float64)
    if r < 5.0
        r -= 1.6
        @horner(r,
                1.42343_71107_49683_57734e0,
                4.63033_78461_56545_29590e0,
                5.76949_72214_60691_40550e0,
                3.64784_83247_63204_60504e0,
                1.27045_82524_52368_38258e0,
                2.41780_72517_74506_11770e-1,
                2.27238_44989_26918_45833e-2,
                7.74545_01427_83414_07640e-4) /
        @horner(r,
                1.0,
                2.05319_16266_37758_82187e0,
                1.67638_48301_83803_84940e0,
                6.89767_33498_51000_04550e-1,
                1.48103_97642_74800_74590e-1,
                1.51986_66563_61645_71966e-2,
                5.47593_80849_95344_94600e-4,
                1.05075_00716_44416_84324e-9)
    else
        r -= 5.0
        @horner(r,
                6.65790_46435_01103_77720e0,
                5.46378_49111_64114_36990e0,
                1.78482_65399_17291_33580e0,
                2.96560_57182_85048_91230e-1,
                2.65321_89526_57612_30930e-2,
                1.24266_09473_88078_43860e-3,
                2.71155_55687_43487_57815e-5,
                2.01033_43992_92288_13265e-7) /
        @horner(r,
                1.0,
                5.99832_20655_58879_37690e-1,
                1.36929_88092_27358_05310e-1,
                1.48753_61290_85061_48525e-2,
                7.86869_13114_56132_59100e-4,
                1.84631_83175_10054_68180e-5,
                1.42151_17583_16445_88870e-7,
                2.04426_31033_89939_78564e-15)
    end
end