using Base.Math: @horner, libm
for f in (:erf, :erfc)
    @eval begin
        function ($f)(x::BigFloat)
        end
        ($f)(x::AbstractFloat) = error("not implemented for ", typeof(x))
    end
    @eval begin
    end
end
for f in (:erfcx, :erfi, :Dawson)
    @eval begin
    end
end
""" """ function erfinv(x::Float64)
    a = abs(x)
    if a >= 1.0
                   @horner(t, 0.14780_64707_15138_316110e2,
               @horner(t, 0.10501_26668_70303_37690e-3,
                          0.1e1))
    end
    if a >= 1.0f0
        if x == 1.0f0
        end
        return @horner(t, 0.10501_31152_37334_38116e-3,
               @horner(t, 0.10501_26668_70303_37690e-3,
                          0.1e1))
    end
end
function erfcx(x::BigFloat)
    if x <= (Clong == Int32 ? 0x1p15 : 0x1p30)
        while abs(w) > ϵ
        end
    end
end
