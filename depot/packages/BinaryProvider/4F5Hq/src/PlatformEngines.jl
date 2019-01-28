export gen_download_cmd, gen_unpack_cmd, gen_package_cmd, gen_list_tarball_cmd,
       parse_tarball_listing, gen_sh_cmd, parse_7z_list, parse_tar_list,
       download_verify_unpack, download_verify, unpack
"""
    gen_download_cmd(url::AbstractString, out_path::AbstractString)
Return a `Cmd` that will download resource located at `url` and store it at
the location given by `out_path`.
This method is initialized by `probe_platform_engines()`, which should be
automatically called upon first import of `BinaryProvider`.
"""
gen_download_cmd = (url::AbstractString, out_path::AbstractString) ->
    error("Call `probe_platform_engines()` before `gen_download_cmd()`")
"""
    gen_unpack_cmd(tarball_path::AbstractString, out_path::AbstractString)
Return a `Cmd` that will unpack the given `tarball_path` into the given
`out_path`.  If `out_path` is not already a directory, it will be created.
This method is initialized by `probe_platform_engines()`, which should be
automatically called upon first import of `BinaryProvider`.
"""
gen_unpack_cmd = (tarball_path::AbstractString, out_path::AbstractString) ->
    error("Call `probe_platform_engines()` before `gen_unpack_cmd()`")
"""
    gen_package_cmd(in_path::AbstractString, tarball_path::AbstractString)
Return a `Cmd` that will package up the given `in_path` directory into a
tarball located at `tarball_path`.
This method is initialized by `probe_platform_engines()`, which should be
automatically called upon first import of `BinaryProvider`.
"""
gen_package_cmd = (in_path::AbstractString, tarball_path::AbstractString) ->
    error("Call `probe_platform_engines()` before `gen_package_cmd()`")
"""
    gen_list_tarball_cmd(tarball_path::AbstractString)
Return a `Cmd` that will list the files contained within the tarball located at
`tarball_path`.  The list will not include directories contained within the
tarball.
This method is initialized by `probe_platform_engines()`, which should be
automatically called upon first import of `BinaryProvider`.
"""
gen_list_tarball_cmd = (tarball_path::AbstractString) ->
    error("Call `probe_platform_engines()` before `gen_list_tarball_cmd()`")
"""
    parse_tarball_listing(output::AbstractString)
Parses the result of `gen_list_tarball_cmd()` into something useful.
This method is initialized by `probe_platform_engines()`, which should be
automatically called upon first import of `BinaryProvider`.
"""
parse_tarball_listing = (output::AbstractString) ->
    error("Call `probe_platform_engines()` before `parse_tarball_listing()`")
"""
    gen_sh_cmd(cmd::Cmd)
Runs a command using `sh`.  On Unices, this will default to the first `sh`
found on the `PATH`, however on Windows if that is not found it will fall back
to the `sh` provided by the `busybox.exe` shipped with Julia.
This method is initialized by `probe_platform_engines()`, which should be
automatically called upon first import of `BinaryProvider`.
"""
gen_sh_cmd = (cmd::Cmd) ->
    error("Call `probe_platform_engines()` before `gen_sh_cmd()`")
"""
    probe_cmd(cmd::Cmd; verbose::Bool = false)
Returns `true` if the given command executes successfully, `false` otherwise.
"""
function probe_cmd(cmd::Cmd; verbose::Bool = false)
    if verbose
        @info("Probing $(cmd.exec[1]) as a possibility...")
    end
    try
        success(cmd)
        if verbose
            @info("  Probe successful for $(cmd.exec[1])")
        end
        return true
    catch
        return false
    end
end
"""
    probe_symlink_creation(dest::AbstractString)
Probes whether we can create a symlink within the given destination directory,
to determine whether a particular filesystem is "symlink-unfriendly".
"""
function probe_symlink_creation(dest::AbstractString)
    while !isdir(dest)
        dest = dirname(dest)
    end
    link_path = joinpath(dest, "binaryprovider_symlink_test")
    while ispath(link_path)
        link_path *= "1"
    end
    try
        symlink("foo", link_path)
        return true
    catch e
        if isa(e, Base.IOError)
            return false
        end
        rethrow(e)
    finally
        rm(link_path; force=true)
    end
end
tempdir_symlink_creation = false
"""
    probe_platform_engines!(;verbose::Bool = false)
Searches the environment for various tools needed to download, unpack, and
package up binaries.  Searches for a download engine to be used by
`gen_download_cmd()` and a compression engine to be used by `gen_unpack_cmd()`,
`gen_package_cmd()`, `gen_list_tarball_cmd()` and `parse_tarball_listing()`, as
well as a `sh` execution engine for `gen_sh_cmd()`.  Running this function
will set the global functions to their appropriate implementations given the
environment this package is running on.
This probing function will automatically search for download engines using a
particular ordering; if you wish to override this ordering and use one over all
others, set the `BINARYPROVIDER_DOWNLOAD_ENGINE` environment variable to its
name, and it will be the only engine searched for. For example, put:
    ENV["BINARYPROVIDER_DOWNLOAD_ENGINE"] = "fetch"
within your `~/.juliarc.jl` file to force `fetch` to be used over `curl`.  If
the given override does not match any of the download engines known to this
function, a warning will be printed and the typical ordering will be performed.
Similarly, if you wish to override the compression engine used, set the
`BINARYPROVIDER_COMPRESSION_ENGINE` environment variable to its name (e.g. `7z`
or `tar`) and it will be the only engine searched for.  If the given override
does not match any of the compression engines known to this function, a warning
will be printed and the typical searching will be performed.
If `verbose` is `true`, print out the various engines as they are searched.
"""
function probe_platform_engines!(;verbose::Bool = false)
    global gen_download_cmd, gen_list_tarball_cmd, gen_package_cmd
    global gen_unpack_cmd, parse_tarball_listing, gen_sh_cmd
    global tempdir_symlink_creation
    tempdir_symlink_creation = probe_symlink_creation(tempdir())
    if verbose
        @info("Symlinks allowed in $(tempdir()): $(tempdir_symlink_creation)")
    end
    agent = "BinaryProvider.jl (https://github.com/JuliaPackaging/BinaryProvider.jl)"
    download_engines = [
        (`curl --help`, (url, path) -> `curl -H "User-Agent: $agent" -C - -\# -f -o $path -L $url`),
        (`wget --help`, (url, path) -> `wget --tries=5 -U $agent -c -O $path $url`),
        (`fetch --help`, (url, path) -> `fetch --user-agent=$agent -f $path $url`),
        (`busybox wget --help`, (url, path) -> `busybox wget -U $agent -c -O $path $url`),
    ]
    unpack_7z = (exe7z) -> begin
        return (tarball_path, out_path) ->
            pipeline(`$exe7z x $(tarball_path) -y -so`,
                     `$exe7z x -si -y -ttar -o$(out_path)`)
    end
    package_7z = (exe7z) -> begin
        return (in_path, tarball_path) ->
            pipeline(`$exe7z a -ttar -so a.tar "$(joinpath(".",in_path,"*"))"`,
                     `$exe7z a -si $(tarball_path)`)
    end
    list_7z = (exe7z) -> begin
        return (path) ->
            pipeline(`$exe7z x $path -so`, `$exe7z l -ttar -y -si`)
    end
    gen_7z = (p) -> (unpack_7z(p), package_7z(p), list_7z(p), parse_7z_list)
    compression_engines = Tuple[]
    for tar_cmd in [`tar`, `busybox tar`]
        unpack_tar = (tarball_path, out_path) -> begin
            Jjz = "z"
            if endswith(tarball_path, ".xz")
                Jjz = "J"
            elseif endswith(tarball_path, ".bz2")
                Jjz = "j"
            end
            return `$tar_cmd -x$(Jjz)f $(tarball_path) --directory=$(out_path)`
        end
        package_tar = (in_path, tarball_path) ->
            `$tar_cmd -czvf $tarball_path -C $(in_path) .`
        list_tar = (in_path) -> `$tar_cmd -tzf $in_path`
        push!(compression_engines, (
            `$tar_cmd --help`,
            unpack_tar,
            package_tar,
            list_tar,
            parse_tar_list,
        ))
    end
    sh_engines = [
        `sh`,
    ]
    @static if Sys.iswindows()
        psh_download = (psh_path) -> begin
            return (url, path) -> begin
                webclient_code = """
                [System.Net.ServicePointManager]::SecurityProtocol =
                    [System.Net.SecurityProtocolType]::Tls12;
                \$webclient = (New-Object System.Net.Webclient);
                \$webclient.Headers.Add("user-agent", "$agent");
                \$webclient.DownloadFile("$url", "$path")
                """
                replace(webclient_code, "\n" => " ")
                return `$psh_path -NoProfile -Command "$webclient_code"`
            end
        end
        psh_path = joinpath(get(ENV, "SYSTEMROOT", "C:\\Windows"), "System32\\WindowsPowerShell\\v1.0\\powershell.exe")
        prepend!(download_engines, [
            (`$psh_path -Command ""`, psh_download(psh_path))
        ])
        prepend!(download_engines, [
            (`powershell -Command ""`, psh_download(`powershell`))
        ])
        prepend!(compression_engines, [(`7z --help`, gen_7z("7z")...)])
        exe7z = joinpath(Sys.BINDIR, "7z.exe")
        prepend!(compression_engines, [(`$exe7z --help`, gen_7z(exe7z)...)])
        busybox = joinpath(Sys.BINDIR, "busybox.exe")
        prepend!(sh_engines, [(`$busybox sh`)])
    end
    if haskey(ENV, "BINARYPROVIDER_DOWNLOAD_ENGINE")
        engine = ENV["BINARYPROVIDER_DOWNLOAD_ENGINE"]
        es = split(engine)
        dl_ngs = filter(e -> e[1].exec[1:length(es)] == es, download_engines)
        if isempty(dl_ngs)
            all_ngs = join([d[1].exec[1] for d in download_engines], ", ")
            warn_msg  = "Ignoring BINARYPROVIDER_DOWNLOAD_ENGINE as its value "
            warn_msg *= "of `$(engine)` doesn't match any known valid engines."
            warn_msg *= " Try one of `$(all_ngs)`."
            @warn(warn_msg)
        else
            download_engines = dl_ngs
        end
    end
    if haskey(ENV, "BINARYPROVIDER_COMPRESSION_ENGINE")
        engine = ENV["BINARYPROVIDER_COMPRESSION_ENGINE"]
        es = split(engine)
        comp_ngs = filter(e -> e[1].exec[1:length(es)] == es, compression_engines)
        if isempty(comp_ngs)
            all_ngs = join([c[1].exec[1] for c in compression_engines], ", ")
            warn_msg  = "Ignoring BINARYPROVIDER_COMPRESSION_ENGINE as its "
            warn_msg *= "value of `$(engine)` doesn't match any known valid "
            warn_msg *= "engines. Try one of `$(all_ngs)`."
            @warn(warn_msg)
        else
            compression_engines = comp_ngs
        end
    end
    download_found = false
    compression_found = false
    sh_found = false
    if verbose
        @info("Probing for download engine...")
    end
    for (test, dl_func) in download_engines
        if probe_cmd(`$test`; verbose=verbose)
            gen_download_cmd = dl_func
            download_found = true
            if verbose
                @info("Found download engine $(test.exec[1])")
            end
            break
        end
    end
    if verbose
        @info("Probing for compression engine...")
    end
    for (test, unpack, package, list, parse) in compression_engines
        if probe_cmd(`$test`; verbose=verbose)
            gen_unpack_cmd = unpack
            gen_package_cmd = package
            gen_list_tarball_cmd = list
            parse_tarball_listing = parse
            if verbose
                @info("Found compression engine $(test.exec[1])")
            end
            compression_found = true
            break
        end
    end
    if verbose
        @info("Probing for sh engine...")
    end
    for path in sh_engines
        if probe_cmd(`$path --help`; verbose=verbose)
            gen_sh_cmd = (cmd) -> `$path -c $cmd`
            if verbose
                @info("Found sh engine $(path.exec[1])")
            end
            sh_found = true
            break
        end
    end
    errmsg = ""
    if !download_found
        errmsg *= "No download engines found. We looked for: "
        errmsg *= join([d[1].exec[1] for d in download_engines], ", ")
        errmsg *= ". Install one and ensure it  is available on the path.\n"
    end
    if !compression_found
        errmsg *= "No compression engines found. We looked for: "
        errmsg *= join([c[1].exec[1] for c in compression_engines], ", ")
        errmsg *= ". Install one and ensure it is available on the path.\n"
    end
    if !sh_found && verbose
        @warn("No sh engines found.  Test suite will fail.")
    end
    if !download_found || !compression_found
        error(errmsg)
    end
end
"""
    parse_7z_list(output::AbstractString)
Given the output of `7z l`, parse out the listed filenames.  This funciton used
by  `list_tarball_files`.
"""
function parse_7z_list(output::AbstractString)
    lines = [chomp(l) for l in split(output, "\n")]
    if isempty(lines)
        return []
    end
    for idx in 1:length(lines)
        if endswith(lines[idx], '\r')
            lines[idx] = lines[idx][1:end-1]
        end
    end
    header_row = findfirst(collect(occursin(" Name", l) && occursin(" Attr", l) for l in lines))
    name_idx = findfirst("Name", lines[header_row])[1]
    attr_idx = findfirst("Attr", lines[header_row])[1] - 1
    lines = [l[name_idx:end] for l in lines if length(l) > name_idx && l[attr_idx] != 'D']
    if isempty(lines)
        return []
    end
    bounds = [i for i in 1:length(lines) if all([c for c in lines[i]] .== Ref('-'))]
    lines = lines[bounds[1]+1:bounds[2]-1]
    for idx in 1:length(lines)
        if startswith(lines[idx], "./") || startswith(lines[idx], ".\\")
            lines[idx] = lines[idx][3:end]
        end
    end
    return lines
end
"""
    parse_tar_list(output::AbstractString)
Given the output of `tar -t`, parse out the listed filenames.  This funciton
used by `list_tarball_files`.
"""
function parse_tar_list(output::AbstractString)
    lines = [chomp(l) for l in split(output, "\n")]
    lines = [l for l in lines if !isempty(l) && !endswith(l, '/')]
    for idx in 1:length(lines)
        if startswith(lines[idx], "./") || startswith(lines[idx], ".\\")
            lines[idx] = lines[idx][3:end]
        end
    end
    return lines
end
"""
    download(url::AbstractString, dest::AbstractString;
             verbose::Bool = false)
Download file located at `url`, store it at `dest`, continuing if `dest`
already exists and the server and download engine support it.
"""
function download(url::AbstractString, dest::AbstractString;
                  verbose::Bool = false)
    download_cmd = gen_download_cmd(url, dest)
    if verbose
        @info("Downloading $(url) to $(dest)...")
    end
    oc = OutputCollector(download_cmd; verbose=verbose)
    try
        if !wait(oc)
            error()
        end
    catch e
        if isa(e, InterruptException)
            rethrow()
        end
        error("Could not download $(url) to $(dest):\n$(e)")
    end
end
"""
    download_verify(url::AbstractString, hash::AbstractString,
                    dest::AbstractString; verbose::Bool = false,
                    force::Bool = false, quiet_download::Bool = false)
Download file located at `url`, verify it matches the given `hash`, and throw
an error if anything goes wrong.  If `dest` already exists, just verify it. If
`force` is set to `true`, overwrite the given file if it exists but does not
match the given `hash`.
This method returns `true` if the file was downloaded successfully, `false`
if an existing file was removed due to the use of `force`, and throws an error
if `force` is not set and the already-existent file fails verification, or if
`force` is set, verification fails, and then verification fails again after
redownloading the file.
If `quiet_download` is set to `false` (the default), this method will print to
stdout when downloading a new file.  If it is set to `true` (and `verbose` is
set to `false`) the downloading process will be completely silent.  If
`verbose` is set to `true`, messages about integrity verification will be
printed in addition to messages regarding downloading.
"""
function download_verify(url::AbstractString, hash::AbstractString,
                         dest::AbstractString; verbose::Bool = false,
                         force::Bool = false, quiet_download::Bool = false)
    file_existed = false
    if isfile(dest)
        file_existed = true
        if verbose
            info_onchange(
                "Destination file $(dest) already exists, verifying...",
                "download_verify_$(dest)",
                @__LINE__,
            )
        end
        try
            verify(dest, hash; verbose=verbose)
            return true
        catch e
            if isa(e, InterruptException)
                rethrow()
            end
            if !force
                rethrow()
            end
            if verbose
                info_onchange(
                    "Verification failed, re-downloading...",
                    "download_verify_$(dest)",
                    @__LINE__,
                )
            end
        end
    end
    mkpath(dirname(dest))
    try
        download(url, dest; verbose=verbose || !quiet_download)
        verify(dest, hash; verbose=verbose)
    catch e
        if isa(e, InterruptException)
            rethrow()
        end
        if file_existed
            if verbose
                @info("Continued download didn't work, restarting from scratch")
            end
            rm(dest; force=true)
            download(url, dest; verbose=verbose || !quiet_download)
            verify(dest, hash; verbose=verbose)
        else
            rethrow()
        end
    end
    return !file_existed
end
"""
    package(src_dir::AbstractString, tarball_path::AbstractString;
            verbose::Bool = false)
Compress `src_dir` into a tarball located at `tarball_path`.
"""
function package(src_dir::AbstractString, tarball_path::AbstractString;
                  verbose::Bool = false)
    withenv("GZIP" => "-9") do
        oc = OutputCollector(gen_package_cmd(src_dir, tarball_path); verbose=verbose)
        try
            if !wait(oc)
                error()
            end
        catch e
            if isa(e, InterruptException)
                rethrow()
            end
            error("Could not package $(src_dir) into $(tarball_path)")
        end
    end
end
"""
    unpack(tarball_path::AbstractString, dest::AbstractString;
           verbose::Bool = false)
Unpack tarball located at file `tarball_path` into directory `dest`.
"""
function unpack(tarball_path::AbstractString, dest::AbstractString;
                verbose::Bool = false)
    copyderef = get(ENV, "BINARYPROVIDER_COPYDEREF", "") == "true" ||
                (tempdir_symlink_creation && !probe_symlink_creation(dest))
    true_dest = dest
    if copyderef
        dest = mktempdir()
    end
    mkpath(dest)
    oc = OutputCollector(gen_unpack_cmd(tarball_path, dest); verbose=verbose)
    try
        if !wait(oc)
            error()
        end
    catch e
        if isa(e, InterruptException)
            rethrow()
        end
        error("Could not unpack $(tarball_path) into $(dest)")
    end
    if copyderef
        function cptry_harder(src, dst)
            mkpath(dst)
            for name in readdir(src)
                srcname = joinpath(src, name)
                dstname = joinpath(dst, name)
                if isdir(srcname)
                    cptry_harder(srcname, dstname)
                else
                    try
                        Base.Filesystem.sendfile(srcname, dstname)
                    catch e
                        if isa(e, Base.IOError)
                            if verbose
                                @warn("Could not copy $(srcname) to $(dstname)")
                            end
                        else
                            rethrow(e)
                        end
                    end
                end
            end
        end
        cptry_harder(dest, true_dest)
        rm(dest; recursive=true, force=true)
    end
end
"""
    download_verify_unpack(url::AbstractString, hash::AbstractString,
                           dest::AbstractString; tarball_path = nothing,
                           verbose::Bool = false, ignore_existence::Bool = false,
                           force::Bool = false)
Helper method to download tarball located at `url`, verify it matches the
given `hash`, then unpack it into folder `dest`.  In general, the method
`install()` should be used to download and install tarballs into a `Prefix`;
this method should only be used if the extra functionality of `install()` is
undesired.
If `tarball_path` is specified, the given `url` will be downloaded to
`tarball_path`, and it will not be removed after downloading and verification
is complete.  If it is not specified, the tarball will be downloaded to a
temporary location, and removed after verification is complete.
If `force` is specified, a verification failure will cause `tarball_path` to be
deleted (if it exists), the `dest` folder to be removed (if it exists) and the
tarball to be redownloaded and reverified.  If the verification check is failed
a second time, an exception is raised.  If `force` is not specified, a
verification failure will result in an immediate raised exception.
If `ignore_existence` is set, the tarball is unpacked even if the destination
directory already exists.
Returns `true` if a tarball was actually unpacked, `false` if nothing was
changed in the destination prefix.
"""
function download_verify_unpack(url::AbstractString,
                                hash::AbstractString,
                                dest::AbstractString;
                                tarball_path = nothing,
                                ignore_existence::Bool = false,
                                force::Bool = false,
                                verbose::Bool = false)
    remove_tarball = false
    if tarball_path === nothing
        remove_tarball = true
        function url_ext(url)
            url = basename(url)
            qidx = findfirst(isequal('?'), url)
            if qidx !== nothing
                url = url[1:qidx]
            end
            dot_idx = findlast(isequal('.'), url)
            if dot_idx === nothing
                return nothing
            end
            return url[dot_idx+1:end]
        end
        ext = url_ext(url)
        if !(ext in ["tar", "gz", "tgz", "bz2", "xz"])
            ext = "gz"
        end
        tarball_path = "$(tempname())-download.$(ext)"
    end
    should_delete = !download_verify(url, hash, tarball_path;
                                     force=force, verbose=verbose)
    if should_delete
        if verbose
            @info("Removing dest directory $(dest) as source tarball changed")
        end
        rm(dest; recursive=true, force=true)
    end
    if !ignore_existence && isdir(dest)
        if verbose
            @info("Destination directory $(dest) already exists, returning")
        end
        return false
    end
    try
        if verbose
            @info("Unpacking $(tarball_path) into $(dest)...")
        end
        unpack(tarball_path, dest; verbose=verbose)
    finally
        if remove_tarball
            rm(tarball_path)
        end
    end
    return true
end
