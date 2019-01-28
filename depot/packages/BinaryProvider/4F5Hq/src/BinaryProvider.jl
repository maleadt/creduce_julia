module BinaryProvider
using Libdl
include("LoggingUtils.jl")
include("OutputCollector.jl")
include("PlatformEngines.jl")
include("PlatformNames.jl")
include("Prefix.jl")
include("Products.jl")
include("CompatShims.jl")
function __init__()
    global global_prefix
    global_prefix = Prefix(joinpath(dirname(pathof(@__MODULE__)), "..", "global_prefix"))
    probe_platform_engines!()
end
include("precompile.jl")
_precompile_()
end # module
