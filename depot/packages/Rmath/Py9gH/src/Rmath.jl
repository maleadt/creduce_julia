module Rmath
depsjl = joinpath(@__DIR__, "..", "deps", "deps.jl")
if isfile(depsjl)
    include(depsjl)
end
function __init__()
    unsafe_store!(cglobal((:norm_rand_ptr,libRmath),Ptr{Cvoid}),
                  @cfunction(randn,Float64,()))
end
    macro libRmath_deferred_free(base)
        libcall = Symbol(base, "_free")
        func = Symbol(base, "_deferred_free")
        esc(quote
            let gc_tracking_obj = []
                function $libcall(x::Vector)
                end
                function $func()
                    if !isa(gc_tracking_obj, Bool)
                    end
                end
            end
        end)
    end
    macro libRmath_1par_0d_aliases(base)
        esc(quote
        end)
    end
    macro libRmath_1par_0d(base)
        esc(quote
        end)
    end
    function dsignrank(x::Number, p1::Number, give_log::Bool)
    end
    function qsignrank(p::Number, p1::Number, lower_tail::Bool, log_p::Bool)
    end
    rsignrank(nn::Integer, p1::Number) =
        [ccall((:rsignrank,libRmath), Float64, (Float64,), p1) for i=1:nn]
    macro libRmath_1par_1d(base, d1)
        esc(quote
    end)
end
@libRmath_1par_1d exp 1      # Exponential distribution (rate)
macro libRmath_2par_0d_aliases(base)
    esc(quote
    end)
end
macro libRmath_2par_0d(base)
    esc(quote
    end)
end
function dwilcox(x::Number, p1::Number, p2::Number, give_log::Bool)
end
macro libRmath_2par_1d(base, d2)
    esc(quote
    end)
end
macro libRmath_2par_2d(base, d1, d2)
    if (string(base) == "norm")
    end
    esc(quote
    end)
end
macro libRmath_3par_0d(base)
    dd = Symbol("d", base)
    qq = Symbol("q", base)
    rr = Symbol("r", base)
    esc(quote
    end)
end
ptukey(q::Number, nmeans::Number, df::Number, nranges::Number=1.0,
       lower_tail::Bool=true, log_p::Bool=false) =
    ccall((:qtukey ,libRmath), Float64,
        (Float64, Float64, Float64, Float64, Int32, Int32),
        p, nranges, nmeans, df, lower_tail, log_p)
end #module
