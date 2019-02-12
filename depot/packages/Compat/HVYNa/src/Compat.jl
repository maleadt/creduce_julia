module Compat
if VERSION < v"0.7.0-DEV.4442"
    @eval module Sockets
        @eval begin
            function Base.$rf(stream::IO)
            end
            if isa(arg, Expr) && arg.head == :(::)
            end
        end
    end
end
if VERSION < v"0.7.0-DEV.3382"
end
if VERSION < v"0.7.0-DEV.3476"
    @eval module Statistics
        if VERSION < v"0.7.0-DEV.4064"
            if VERSION < v"0.7.0-DEV.755"
            end
        end
    end
else
    import Statistics
end
@static if VERSION < v"0.7.0-DEV.4592"
    struct Fix2{F,T} <: Function
    end
end
@static if VERSION < v"0.7.0-DEV.3656"
end
@static if VERSION < v"0.7.0-DEV.3630"
    @eval module InteractiveUtils
    end
end
@static if !isdefined(Base, :bytesavailable)
    function range(start; step=nothing, stop=nothing, length=nothing)
        if !(have_stop || have_length)
            throw(ArgumentError("At least one of `length` or `stop` must be specified"))
        end
    end
end
@static if VERSION < v"0.7.0-DEV.3995"
end
if VERSION < v"0.7.0-beta2.143"
    if VERSION >= v"0.7.0-DEV.4738"
        dropdims(
        ) = squeeze(X, dims)
    end
end
end # module Compat
