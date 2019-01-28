export Product, LibraryProduct, FileProduct, ExecutableProduct, satisfied,
       locate, write_deps_file, variable_name
import Base: repr
""" """ abstract type Product end
""" """ function satisfied(p::Product; platform::Platform = platform_key_abi(),
                               verbose::Bool = false, isolate::Bool = false)
    return locate(p; platform=platform, verbose=verbose, isolate=isolate) != nothing
end
""" """ function variable_name(p::Product)
    return string(p.variable_name)
end
""" """ struct LibraryProduct <: Product
    dir_path::Union{String, Nothing}
    libnames::Vector{String}
    variable_name::Symbol
    prefix::Union{Prefix, Nothing}
    """
        LibraryProduct(prefix::Prefix, libname::AbstractString,
                       varname::Symbol)
    Declares a `LibraryProduct` that points to a library located within the
    `libdir` of the given `Prefix`, with a name containing `libname`.  As an
    example, given that `libdir(prefix)` is equal to `usr/lib`, and `libname`
    is equal to `libnettle`, this would be satisfied by the following paths:
        usr/lib/libnettle.so
        usr/lib/libnettle.so.6
        usr/lib/libnettle.6.dylib
        usr/lib/libnettle-6.dll
    Libraries matching the search pattern are rejected if they are not
    `dlopen()`'able.
    """
    function LibraryProduct(prefix::Prefix, libname::AbstractString,
                            varname::Symbol)
        return LibraryProduct(prefix, [libname], varname)
    end
    function LibraryProduct(prefix::Prefix, libnames::Vector{S},
                            varname::Symbol) where {S <: AbstractString}
        return new(nothing, libnames, varname, prefix)
    end
    """
        LibraryProduct(dir_path::AbstractString, libname::AbstractString,
                       varname::Symbol)
    For finer-grained control over `LibraryProduct` locations, you may directly
    pass in the `dir_path` instead of auto-inferring it from `libdir(prefix)`.
    """
    function LibraryProduct(dir_path::AbstractString, libname::AbstractString,
                            varname::Symbol)
        return LibraryProduct(dir_path, [libname], varname)
    end
    function LibraryProduct(dir_path::AbstractString, libnames::Vector{S},
                            varname::Symbol) where {S <: AbstractString}
       return new(dir_path, libnames, varname, nothing)
    end
end
function repr(p::LibraryProduct)
    libnames = repr(p.libnames)
    varname = repr(p.variable_name)
    if p.prefix === nothing
        return "LibraryProduct($(repr(p.dir_path)), $(libnames), $(varname))"
    else
        return "LibraryProduct(prefix, $(libnames), $(varname))"
    end
end
""" """ function locate(lp::LibraryProduct; verbose::Bool = false,
                platform::Platform = platform_key_abi(), isolate::Bool = false)
    dir_path = lp.dir_path
    if dir_path === nothing
        dir_path = libdir(lp.prefix, platform)
    end
    if !isdir(dir_path)
        if verbose
            @info("Directory $(dir_path) does not exist!")
        end
        return nothing
    end
    for f in readdir(dir_path)
        if !valid_dl_path(f, platform)
            continue
        end
        if verbose
            @info("Found a valid dl path $(f) while looking for $(join(lp.libnames, ", "))")
        end
        for libname in lp.libnames
            if startswith(basename(f), libname)
                dl_path = abspath(joinpath(dir_path), f)
                if verbose
                    @info("$(dl_path) matches our search criteria of $(libname)")
                end
                if platforms_match(platform, platform_key_abi())
                    if isolate
                        if success(`$(Base.julia_cmd()) -e "import Libdl; Libdl.dlopen(\"$dl_path\")"`)
                            return dl_path
                        end
                    else
                        hdl = Libdl.dlopen_e(dl_path)
                        if !(hdl in (C_NULL, nothing))
                            Libdl.dlclose(hdl)
                            return dl_path
                        end
                    end
                    if verbose
                        @info("$(dl_path) cannot be dlopen'ed")
                    end
                else
                    return dl_path
                end
            end
        end
    end
    if verbose
        @info("Could not locate $(join(lp.libnames, ", ")) inside $(dir_path)")
    end
    return nothing
end
""" """ struct ExecutableProduct <: Product
    path::AbstractString
    variable_name::Symbol
    prefix::Union{Prefix, Nothing}
    """
    `ExecutableProduct(prefix::Prefix, binname::AbstractString,
                       varname::Symbol)`
    Declares an `ExecutableProduct` that points to an executable located within
    the `bindir` of the given `Prefix`, named `binname`.
    """
    function ExecutableProduct(prefix::Prefix, binname::AbstractString,
                               varname::Symbol)
        return new(joinpath(bindir(prefix), binname), varname, prefix)
    end
    """
    `ExecutableProduct(binpath::AbstractString, varname::Symbol)`
    For finer-grained control over `ExecutableProduct` locations, you may directly
    pass in the full `binpath` instead of auto-inferring it from `bindir(prefix)`.
    """
    function ExecutableProduct(binpath::AbstractString, varname::Symbol)
        return new(binpath, varname, nothing)
    end
end
function repr(p::ExecutableProduct)
    varname = repr(p.variable_name)
    if p.prefix === nothing
        return "ExecutableProduct($(repr(p.path)), $(varname))"
    else
        rp = relpath(p.path, bindir(p.prefix))
        return "ExecutableProduct(prefix, $(repr(rp)), $(varname))"
    end
end
""" """ function locate(ep::ExecutableProduct; platform::Platform = platform_key_abi(),
                verbose::Bool = false, isolate::Bool = false)
    path = if platform isa Windows && !endswith(ep.path, ".exe")
        "$(ep.path).exe"
    else
        ep.path
    end
    if !isfile(path)
        if verbose
            @info("$(ep.path) does not exist, reporting unsatisfied")
        end
        return nothing
    end
    @static if !Sys.iswindows()
        if uperm(path) & 0x1 == 0
            if verbose
                @info("$(path) is not executable, reporting unsatisfied")
            end
            return nothing
        end
    end
    return path
end
""" """ struct FileProduct <: Product
    path::AbstractString
    variable_name::Symbol
    prefix::Union{Prefix, Nothing}
    """
        FileProduct(prefix::Prefix, relative_path::AbstractString,
                                    varname::Symbol)`
    Declares a `FileProduct` that points to a file located relative to a the
    root of a `Prefix`.
    """
    function FileProduct(prefix::Prefix, relative_path::AbstractString,
                                         varname::Symbol)
        file_path = joinpath(prefix.path, relative_path)
        return new(file_path, varname, prefix)
    end
    """
        FileProduct(file_path::AbstractString, varname::Symbol)
    For finer-grained control over `FileProduct` locations, you may directly
    pass in the full `file_pathpath` instead of defining it in reference to
    a root `Prefix`.
    """
    function FileProduct(file_path::AbstractString, varname::Symbol)
        return new(file_path, varname, nothing)
    end
end
function repr(p::FileProduct)
    varname = repr(p.variable_name)
    if p.prefix === nothing
        return "FileProduct($(repr(p.path)), $(varname))"
    else
        rp = relpath(p.path, p.prefix.path)
        return "FileProduct(prefix, $(repr(rp)), $(varname))"
    end
end
""" """ function locate(fp::FileProduct; platform::Platform = platform_key_abi(),
                                 verbose::Bool = false, isolate::Bool = false)
    mappings = Dict()
    for (var, val) in [("target", triplet(platform)), ("nbits", wordsize(platform))]
        mappings["\$$(var)"] = string(val)
        mappings["\${$(var)}"] = string(val)
    end
    expanded = fp.path
    for (old, new) in mappings
        expanded = replace(expanded, old => new)
    end
    if isfile(expanded)
        if verbose
            @info("FileProduct $(fp.path) found at $(realpath(expanded))")
        end
        return expanded
    end
    if verbose
        @info("FileProduct $(fp.path) not found")
    end
    return nothing
end
""" """ function write_deps_file(depsjl_path::AbstractString, products::Vector{P};
                         verbose::Bool=false) where {P <: Product}
    escape_path = path -> replace(path, "\\" => "\\\\")
    if basename(dirname(dirname(dirname(dirname(depsjl_path))))) == "packages"
        package_name = basename(dirname(dirname(dirname(depsjl_path))))
    else
        package_name = basename(dirname(dirname(depsjl_path)))
    end
    rebuild = strip("""
    Please re-run Pkg.build(\\\"$(package_name)\\\"), and restart Julia.
    """)
    for p in products
        if !satisfied(p; verbose=verbose)
            error("$p is not satisfied, cannot generate deps.jl!")
        end
    end
    open(depsjl_path, "w") do depsjl_file
        println(depsjl_file, strip("""
        if isdefined((@static VERSION < v"0.7.0-DEV.484" ? current_module() : @__MODULE__), :Compat)
            import Compat.Libdl
        elseif VERSION >= v"0.7.0-DEV.3382"
            import Libdl
        end
        """))
        for product in products
            product_path = locate(product, platform=platform_key_abi(),
                                           verbose=verbose)
            product_path = relpath(product_path, dirname(depsjl_path))
            product_path = escape_path(product_path)
            vp = variable_name(product)
            println(depsjl_file, strip("""
            const $(vp) = joinpath(dirname(@__FILE__), \"$(product_path)\")
            """))
        end
        println(depsjl_file, "function check_deps()")
        for product in products
            varname = variable_name(product)
            println(depsjl_file, "    global $(varname)");
            println(depsjl_file, """
                if !isfile($(varname))
                    error("\$($(varname)) does not exist, $(rebuild)")
                end
            """)
            if typeof(product) <: LibraryProduct
                println(depsjl_file, """
                    if Libdl.dlopen_e($(varname)) in (C_NULL, nothing)
                        error("\$($(varname)) cannot be opened, $(rebuild)")
                    end
                """)
            end
        end
        @static if !Sys.iswindows()
            if any(p isa ExecutableProduct for p in products)
                dllist = Libdl.dllist()
                libjulia = filter(x -> occursin("libjulia", x), dllist)[1]
                julia_libdir = repr(joinpath(dirname(libjulia), "julia"))
                envvar_name = @static if Sys.isapple()
                    "DYLD_LIBRARY_PATH"
                else Sys.islinux()
                    "LD_LIBRARY_PATH"
                end
                envvar_name = repr(envvar_name)
                println(depsjl_file, """
                    libpaths = split(get(ENV, $(envvar_name), ""), ":")
                    if !($(julia_libdir) in libpaths)
                        push!(libpaths, $(julia_libdir))
                    end
                    ENV[$(envvar_name)] = join(filter(!isempty, libpaths), ":")
                """)
            end
        end
        println(depsjl_file, "end")
    end
end
