import Base: convert, joinpath, show, withenv
using SHA
export Prefix, bindir, libdir, includedir, logdir, activate, deactivate,
       extract_name_version_platform_key, extract_platform_key, isinstalled,
       install, uninstall, manifest_from_url, manifest_for_file,
       list_tarball_files, verify, temp_prefix, package
function safe_isfile(path)
    try
        return isfile(path)
    catch e
        if typeof(e) <: Base.IOError && e.code == Base.UV_EINVAL
            return false
        end
        rethrow(e)
    end
end
""" """ function temp_prefix(func::Function)
    function _tempdir()
        @static if Sys.isapple()
            return "/tmp"
        else
            return tempdir()
        end
    end
    mktempdir(_tempdir()) do path
        prefix = Prefix(path)
        func(prefix)
    end
end
global_prefix = nothing
struct Prefix
    path::String
    """
        Prefix(path::AbstractString)
    A `Prefix` represents a binary installation location.  There is a default
    global `Prefix` (available at `BinaryProvider.global_prefix`) that packages
    are installed into by default, however custom prefixes can be created
    trivially by simply constructing a `Prefix` with a given `path` to install
    binaries into, likely including folders such as `bin`, `lib`, etc...
    """
    function Prefix(path::AbstractString)
        path = abspath(path)
        mkpath(path)
        return new(path)
    end
end
joinpath(prefix::Prefix, args...) = joinpath(prefix.path, args...)
joinpath(s::AbstractString, prefix::Prefix, args...) = joinpath(s, prefix.path, args...)
convert(::Type{AbstractString}, prefix::Prefix) = prefix.path
show(io::IO, prefix::Prefix) = show(io, "Prefix($(prefix.path))")
""" """ function withenv(f::Function, prefixes::Vector{Prefix};
                 julia_libdir::Bool = true)
    function joinenv(key, dirs, sep, tail_dirs = [])
        value = [dirs..., split(get(ENV, key, ""), sep)..., tail_dirs...]
        return join(unique([abspath(d) for d in value if isdir(d)]), sep)
    end
    sep = Sys.iswindows() ? ";" : ":"
    mapping = ["PATH" => joinenv("PATH", bindir.(prefixes), sep)]
    if !Sys.iswindows()
        libdirs = libdir.(prefixes)
        tail_dirs = []
        if julia_libdir
            tail_dirs = [joinpath(Sys.BINDIR, Base.PRIVATE_LIBDIR)]
        end
        envname = Sys.isapple() ? "DYLD_FALLBACK_LIBRARY_PATH" : "LD_LIBRARY_PATH"
        push!(mapping, envname => joinenv(envname, libdirs, ":", tail_dirs))
    end
    return withenv(f, mapping...)
end
withenv(f::Function, prefix::Prefix) = withenv(f, [prefix])
""" """ function bindir(prefix::Prefix)
    return joinpath(prefix, "bin")
end
""" """ function libdir(prefix::Prefix, platform = platform_key_abi())
    if Sys.iswindows(platform)
        return joinpath(prefix, "bin")
    else
        return joinpath(prefix, "lib")
    end
end
""" """ function includedir(prefix::Prefix)
    return joinpath(prefix, "include")
end
""" """ function logdir(prefix::Prefix)
    return joinpath(prefix, "logs")
end
""" """ function extract_platform_key(path::AbstractString)
    try
        return extract_name_version_platform_key(path)[3]
    catch
        @warn("Could not extract the platform key of $(path); continuing...")
        return platform_key_abi()
    end
end
""" """ function extract_name_version_platform_key(path::AbstractString)
    m = match(r"^(.*?)\.v(.*?)\.([^\.\-]+-[^\.\-]+-([^\-]+-){0,2}[^\-]+).tar.gz$", basename(path))
    if m === nothing
        error("Could not parse name, platform key and version from $(path)")
    end
    name = m.captures[1]
    version = VersionNumber(m.captures[2])
    platkey = platform_key_abi(m.captures[3])
    return name, version, platkey
end
""" """ function isinstalled(tarball_url::AbstractString, hash::AbstractString;
                     prefix::Prefix = global_prefix)
    tarball_path = joinpath(prefix, "downloads", basename(tarball_url))
    hash_path = "$(tarball_path).sha256"
    if safe_isfile(tarball_url)
        tarball_path = tarball_url
    end
    try
        verify(tarball_path, hash; verbose=false, hash_path=hash_path)
    catch
        return false
    end
    tarball_time = stat(tarball_path).mtime
    manifest_path = manifest_from_url(tarball_url, prefix=prefix)
    isfile(manifest_path) || return false
    stat(manifest_path).mtime >= tarball_time || return false
    for installed_file in (joinpath(prefix, f) for f in chomp.(readlines(manifest_path)))
        ((isfile(installed_file) || islink(installed_file)) &&
         stat(installed_file).ctime >= tarball_time) || return false
    end
    return true
end
""" """ function install(tarball_url::AbstractString,
                 hash::AbstractString;
                 prefix::Prefix = global_prefix,
                 tarball_path::AbstractString =
                     joinpath(prefix, "downloads", basename(tarball_url)),
                 force::Bool = false,
                 ignore_platform::Bool = false,
                 verbose::Bool = false)
    if !ignore_platform
        try
            platform = extract_platform_key(tarball_url)
            if !platforms_match(platform, platform_key_abi())
                msg = replace(strip("""
                Will not install a tarball of platform $(triplet(platform)) on
                a system of platform $(triplet(platform_key_abi())) unless
                `ignore_platform` is explicitly set to `true`.
                """), "\n" => " ")
                throw(ArgumentError(msg))
            end
        catch e
            if isa(e, ArgumentError)
                msg = "$(e.msg), override this by setting `ignore_platform`"
                throw(ArgumentError(msg))
            else
                rethrow(e)
            end
        end
    end
    try mkpath(dirname(tarball_path)) catch; end
    if safe_isfile(tarball_url)
        hash_path = "$(tarball_path).sha256"
        tarball_path = tarball_url
        verify(tarball_path, hash; verbose=verbose, hash_path=hash_path)
    else
        download_verify(tarball_url, hash, tarball_path;
                        force=force, verbose=verbose)
    end
    if verbose
        @info("Installing $(tarball_path) into $(prefix.path)")
    end
    manifest_path = manifest_from_url(tarball_url, prefix=prefix)
    force && isfile(manifest_path) && uninstall(manifest_path, verbose=verbose)
    file_list = list_tarball_files(tarball_path)
    for file in file_list
        if isfile(joinpath(prefix, file))
            if !force
                msg  = "$(file) already exists and would be overwritten while "
                msg *= "installing $(basename(tarball_path))\n"
                msg *= "Will not overwrite unless `force = true` is set."
                error(msg)
            else
                if verbose
                    @info("$(file) already exists, force-removing")
                end
                rm(file; force=true)
            end
        end
    end
    unpack(tarball_path, prefix.path; verbose=verbose)
    mkpath(dirname(manifest_path))
    open(manifest_path, "w") do f
        write(f, join(file_list, "\n"))
    end
    return true
end
""" """ function uninstall(manifest::AbstractString;
                   verbose::Bool = false)
    if !isfile(manifest)
        error("Manifest path $(manifest) does not exist")
    end
    prefix_path = dirname(dirname(manifest))
    if verbose
        relmanipath = relpath(manifest, prefix_path)
        @info("Removing files installed by $(relmanipath)")
    end
    for path in [chomp(l) for l in readlines(manifest)]
        delpath = joinpath(prefix_path, path)
        if !isfile(delpath) && !islink(delpath)
            if verbose
                @info("  $delpath does not exist, but ignoring")
            end
        else
            if verbose
                delrelpath = relpath(delpath, prefix_path)
                @info("  $delrelpath removed")
            end
            rm(delpath; force=true)
            deldir = abspath(dirname(delpath))
            if isempty(readdir(deldir)) && deldir != abspath(prefix_path)
                if verbose
                    delrelpath = relpath(deldir, prefix_path)
                    @info("  Culling empty directory $delrelpath")
                end
                rm(deldir; force=true, recursive=true)
            end
        end
    end
    if verbose
        @info("  $(relmanipath) removed")
    end
    rm(manifest; force=true)
    return true
end
""" """ function manifest_from_url(url::AbstractString;
                           prefix::Prefix = global_prefix())
    return joinpath(prefix, "manifests", basename(url)[1:end-7] * ".list")
end
""" """ function manifest_for_file(path::AbstractString;
                           prefix::Prefix = global_prefix)
    if !isfile(path)
        error("File $(path) does not exist")
    end
    search_path = relpath(path, prefix.path)
    if startswith(search_path, "..")
        error("Cannot search for paths outside of the given Prefix!")
    end
    manidir = joinpath(prefix, "manifests")
    for fname in [f for f in readdir(manidir) if endswith(f, ".list")]
        manifest_path = joinpath(manidir, fname)
        if search_path in [chomp(l) for l in readlines(manifest_path)]
            return manifest_path
        end
    end
    error("Could not find $(search_path) in any manifest files")
end
""" """ function list_tarball_files(path::AbstractString; verbose::Bool = false)
    if !isfile(path)
        error("Tarball path $(path) does not exist")
    end
    oc = OutputCollector(gen_list_tarball_cmd(path); verbose=verbose)
    try
        if !wait(oc)
            error()
        end
    catch
        error("Could not list contents of tarball $(path)")
    end
    return parse_tarball_listing(collect_stdout(oc))
end
""" """ function verify(path::AbstractString, hash::AbstractString; verbose::Bool = false,
                report_cache_status::Bool = false, hash_path::AbstractString="$(path).sha256")
    if length(hash) != 64
        msg  = "Hash must be 256 bits (64 characters) long, "
        msg *= "given hash is $(length(hash)) characters long"
        error(msg)
    end
    status = :hash_consistent
    if isfile(hash_path)
        if read(hash_path, String) == hash
            if stat(hash_path).mtime >= stat(path).mtime
                if verbose
                    info_onchange(
                        "Hash cache is consistent, returning true",
                        "verify_$(hash_path)",
                        @__LINE__,
                    )
                end
                status = :hash_cache_consistent
                if report_cache_status
                    return true, status
                else
                    return true
                end
            else
                if verbose
                    info_onchange(
                        "File has been modified, hash cache invalidated",
                        "verify_$(hash_path)",
                        @__LINE__,
                    )
                end
                status = :file_modified
            end
        else
            if verbose
                info_onchange(
                    "Verification hash mismatch, hash cache invalidated",
                    "verify_$(hash_path)",
                    @__LINE__,
                )
            end
            status = :hash_cache_mismatch
        end
    else
        if verbose
            info_onchange(
                "No hash cache found",
                "verify_$(hash_path)",
                @__LINE__,
            )
        end
        status = :hash_cache_missing
    end
    open(path) do file
        calc_hash = bytes2hex(sha256(file))
        if verbose
            info_onchange(
                "Calculated hash $calc_hash for file $path",
                "hash_$(hash_path)",
                @__LINE__,
            )
        end
        if calc_hash != hash
            msg  = "Hash Mismatch!\n"
            msg *= "  Expected sha256:   $hash\n"
            msg *= "  Calculated sha256: $calc_hash"
            error(msg)
        end
    end
    open(hash_path, "w") do file
        write(file, hash)
    end
    if report_cache_status
        return true, status
    else
        return true
    end
end
""" """ function package(prefix::Prefix,
                 output_base::AbstractString,
                 version::VersionNumber;
                 platform::Platform = platform_key_abi(),
                 verbose::Bool = false,
                 force::Bool = false)
    out_path = "$(output_base).v$(version).$(triplet(platform)).tar.gz"
    if isfile(out_path)
        if force
            if verbose
                @info("$(out_path) already exists, force-overwriting...")
            end
            rm(out_path; force=true)
        else
            msg = replace(strip("""
            $(out_path) already exists, refusing to package into it without
            `force` being set to `true`.
            """), "\n" => " ")
            error(msg)
        end
    end
    package(prefix.path, out_path; verbose=verbose)
    hash = open(out_path, "r") do f
        return bytes2hex(sha256(f))
    end
    if verbose
        @info("SHA256 of $(basename(out_path)): $(hash)")
    end
    return out_path, hash
end
