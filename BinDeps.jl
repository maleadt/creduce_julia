__precompile__()
module BinDeps
using Compat
if VERSION >= v"0.7.0-DEV.3073"
    const _HOME = Sys.BINDIR
else
    const _HOME = JULIA_HOME
end
if VERSION >= v"0.7.0-DEV.3382"
    using Libdl
end
export @build_steps, find_library, download_cmd, unpack_cmd,
    Choice, Choices, CCompile, FileDownloader, FileRule,
    ChangeDirectory, FileUnpacker, prepare_src,
    autotools_install, CreateDirectory, MakeTargets,
    MAKE_CMD, glibc_version
function find_library(pkg,libname,files)
    Base.warn_once("BinDeps.find_library is deprecated; use Base.find_library instead.")
    dl = C_NULL
    for filename in files
        dl = Libdl.dlopen_e(joinpath(Pkg.dir(),pkg,"deps","usr","lib",filename))
        if dl != C_NULL
            ccall(:add_library_mapping,Cint,(Ptr{Cchar},Ptr{Cvoid}),libname,dl)
            return true
        end
        dl = Libdl.dlopen_e(filename)
        if dl != C_NULL
            ccall(:add_library_mapping,Cint,(Ptr{Cchar},Ptr{Cvoid}),libname,dl)
            return true
        end
    end
    dl = Libdl.dlopen_e(libname)
    dl != C_NULL ? true : false
end
macro make_rule(condition,command)
    quote
        if(!$(esc(condition)))
            $(esc(command))
            @assert $(esc(condition))
        end
    end
end
abstract type BuildStep end
downloadcmd = nothing
function download_cmd(url::AbstractString, filename::AbstractString)
    global downloadcmd
    if downloadcmd === nothing
        for download_engine in (Compat.Sys.iswindows() ? ("C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell",
                :powershell, :curl, :wget, :fetch) : (:curl, :wget, :fetch))
            if endswith(string(download_engine), "powershell")
                checkcmd = `$download_engine -NoProfile -Command ""`
            else
                checkcmd = `$download_engine --help`
            end
            try
                if success(checkcmd)
                    downloadcmd = download_engine
                    break
                end
            catch
                continue # don't bail if one of these fails
            end
        end
    end
    if downloadcmd == :wget
        return `$downloadcmd -O $filename $url`
    elseif downloadcmd == :curl
        return `$downloadcmd -f -o $filename -L $url`
    elseif downloadcmd == :fetch
        return `$downloadcmd -f $filename $url`
    elseif endswith(string(downloadcmd), "powershell")
        tls_cmd = "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12"
        download_cmd = "(new-object net.webclient).DownloadFile(\"$(url)\", \"$(filename)\")"
        return `$downloadcmd -NoProfile -Command "$(tls_cmd); $(download_cmd)"`
    else
        extraerr = Compat.Sys.iswindows() ? "check if powershell is on your path or " : ""
        error("No download agent available; $(extraerr)install curl, wget, or fetch.")
    end
end
if Compat.Sys.isunix() && Sys.KERNEL != :FreeBSD
    function unpack_cmd(file,directory,extension,secondary_extension)
        if ((extension == ".gz" || extension == ".Z") && secondary_extension == ".tar") || extension == ".tgz"
            return (`tar xzf $file --directory=$directory`)
        elseif (extension == ".bz2" && secondary_extension == ".tar") || extension == ".tbz"
            return (`tar xjf $file --directory=$directory`)
        elseif extension == ".xz" && secondary_extension == ".tar"
            return pipeline(`unxz -c $file `, `tar xv --directory=$directory`)
        elseif extension == ".tar"
            return (`tar xf $file --directory=$directory`)
        elseif extension == ".zip"
            return (`unzip -x $file -d $directory`)
        elseif extension == ".gz"
            return pipeline(`mkdir $directory`, `cp $file $directory`, `gzip -d $directory/$file`)
        end
        error("I don't know how to unpack $file")
    end
end
if Sys.KERNEL == :FreeBSD
    # The `tar` on FreeBSD can auto-detect the archive format via libarchive.
    # The supported formats can be found in libarchive-formats(5).
    # For NetBSD and OpenBSD, libarchive is not available.
    # For macOS, it is. But the previous unpack function works fine already.
    function unpack_cmd(file, dir, ext, secondary_ext)
        tar_args = ["--no-same-owner", "--no-same-permissions"]
        return pipeline(
            `/bin/mkdir -p $dir`,
            `/usr/bin/tar -xf $file -C $dir $tar_args`)
    end
end
if Compat.Sys.iswindows()
    const exe7z = joinpath(_HOME, "7z.exe")
    function unpack_cmd(file,directory,extension,secondary_extension)
        if ((extension == ".Z" || extension == ".gz" || extension == ".xz" || extension == ".bz2") &&
                secondary_extension == ".tar") || extension == ".tgz" || extension == ".tbz"
            return pipeline(`$exe7z x $file -y -so`, `$exe7z x -si -y -ttar -o$directory`)
        elseif (extension == ".zip" || extension == ".7z" || extension == ".tar" ||
                (extension == ".exe" && secondary_extension == ".7z"))
            return (`$exe7z x $file -y -o$directory`)
        end
        error("I don't know how to unpack $file")
    end
end
mutable struct SynchronousStepCollection
    steps::Vector{Any}
    cwd::AbstractString
    oldcwd::AbstractString
    SynchronousStepCollection(cwd) = new(Any[],cwd,cwd)
    SynchronousStepCollection() = new(Any[],"","")
end
import Base: push!, run, |
push!(a::SynchronousStepCollection,args...) = push!(a.steps,args...)
mutable struct ChangeDirectory <: BuildStep
    dir::AbstractString
end
mutable struct CreateDirectory <: BuildStep
    dest::AbstractString
    mayexist::Bool
    CreateDirectory(dest, me) = new(dest,me)
    CreateDirectory(dest) = new(dest,true)
end
struct RemoveDirectory <: BuildStep
    dest::AbstractString
end
mutable struct FileDownloader <: BuildStep
    src::AbstractString     # url
    dest::AbstractString    # local_file
end
mutable struct ChecksumValidator <: BuildStep
    sha::AbstractString
    path::AbstractString
end
mutable struct FileUnpacker <: BuildStep
    src::AbstractString     # archive file
    dest::AbstractString    # directory to unpack into
    target::AbstractString  # file or directory inside the archive to test
                            # for existence (or blank to check for a.tgz => a/)
end
mutable struct MakeTargets <: BuildStep
    dir::AbstractString
    targets::Vector{String}
    env::Dict
    MakeTargets(dir,target;env = Dict{AbstractString,AbstractString}()) = new(dir,target,env)
    MakeTargets(target::Vector{<:AbstractString};env = Dict{AbstractString,AbstractString}()) = new("",target,env)
    MakeTargets(target::String;env = Dict{AbstractString,AbstractString}()) = new("",[target],env)
    MakeTargets(;env = Dict{AbstractString,AbstractString}()) = new("",String[],env)
end
mutable struct AutotoolsDependency <: BuildStep
    src::AbstractString     #src direcory
    prefix::AbstractString
    builddir::AbstractString
    configure_options::Vector{AbstractString}
    libtarget::Vector{AbstractString}
    include_dirs::Vector{AbstractString}
    lib_dirs::Vector{AbstractString}
    rpath_dirs::Vector{AbstractString}
    installed_libpath::Vector{String} # The library is considered installed if any of these paths exist
    config_status_dir::AbstractString
    force_rebuild::Bool
    env
    AutotoolsDependency(;srcdir::AbstractString = "", prefix = "", builddir = "", configure_options=AbstractString[], libtarget = AbstractString[], include_dirs=AbstractString[], lib_dirs=AbstractString[], rpath_dirs=AbstractString[], installed_libpath = String[], force_rebuild=false, config_status_dir = "", env = Dict{String,String}()) =
        new(srcdir,prefix,builddir,configure_options,isa(libtarget,Vector) ? libtarget : AbstractString[libtarget],include_dirs,lib_dirs,rpath_dirs,installed_libpath,config_status_dir,force_rebuild,env)
end
mutable struct Choice
    name::Symbol
    description::AbstractString
    step::SynchronousStepCollection
    Choice(name,description,step) = (s=SynchronousStepCollection();lower(step,s);new(name,description,s))
end
mutable struct Choices <: BuildStep
    choices::Vector{Choice}
    Choices() = new(Choice[])
    Choices(choices::Vector{Choice}) = new(choices)
end
push!(c::Choices, args...) = push!(c.choices, args...)
function run(c::Choices)
    println()
    info("There are multiple options available for installing this dependency:")
    while true
        for x in c.choices
            println("- "*string(x.name)*": "*x.description)
        end
        while true
            print("Plese select the desired method: ")
            method = Symbol(chomp(readline(STDIN)))
            for x in c.choices
                if(method == x.name)
                    return run(x.step)
                end
            end
            warn("Invalid method")
        end
    end
end
mutable struct CCompile <: BuildStep
    srcFile::AbstractString
    destFile::AbstractString
    options::Vector{String}
    libs::Vector{String}
end
lower(cc::CCompile,c) = lower(FileRule(cc.destFile,`gcc $(cc.options) $(cc.srcFile) $(cc.libs) -o $(cc.destFile)`),c)
mutable struct DirectoryRule <: BuildStep
    dir::AbstractString
    step
end
mutable struct PathRule <: BuildStep
    path::AbstractString
    step
end
function meta_lower(a::Expr,blk::Expr,collection)
    if(a.head == :block || a.head == :tuple)
        for x in a.args
            if(isa(x,Expr))
                if(x.head == :block)
                    new_collection = gensym()
                    push!(blk.args,quote
                        $(esc(new_collection)) = SynchronousStepCollection($(esc(collection)).cwd)
                        push!($(esc(collection)),$(esc(new_collection)))
                    end)
                    meta_lower(x,blk,new_collection)
                 elseif(x.head != :line)
                     push!(blk.args,quote
                         lower($(esc(x)), $(esc(collection)))
                     end)
                 end
            elseif(!isa(x,LineNumberNode))
                meta_lower(x,blk,collection)
            end
        end
    else
        push!(blk.args,quote
            $(esc(collection)),lower($(esc(a)), $(esc(collection)))
        end)
    end
end
function meta_lower(a::Tuple,blk::Expr,collection)
    for x in a
        meta_lower(a,blk,collection)
    end
end
function meta_lower(a,blk::Expr,collection)
    push!(blk.args,quote
        $(esc(collection)), lower($(esc(a)), $(esc(collection)))
    end)
end
macro dependent_steps(steps)
    blk = Expr(:block)
    meta_lower(steps,blk,:collection)
    blk
end
macro build_steps(steps)
    collection = gensym()
    blk = Expr(:block)
    push!(blk.args,quote
        $(esc(collection)) = SynchronousStepCollection()
    end)
    meta_lower(steps,blk,collection)
    push!(blk.args, quote; $(esc(collection)); end)
    blk
end
src(b::BuildStep) = b.src
dest(b::BuildStep) = b.dest
(|)(a::BuildStep,b::BuildStep) = SynchronousStepCollection()
function (|)(a::SynchronousStepCollection,b::SynchronousStepCollection)
    if a.cwd == b.cwd
        append!(a.steps,b.steps)
    else
        push!(a.steps,b)
    end
    a
end
(|)(a::SynchronousStepCollection,b::Function) = (lower(b,a);a)
(|)(a::SynchronousStepCollection,b) = (lower(b,a);a)
(|)(b::Function,a::SynchronousStepCollection) = (c=SynchronousStepCollection(); ((c|b)|a))
(|)(b,a::SynchronousStepCollection) = (c=SynchronousStepCollection(); ((c|b)|a))
mutable struct FileRule <: BuildStep
    file::Array{AbstractString}
    step
    FileRule(file::AbstractString,step) = FileRule(AbstractString[file],step)
    function FileRule(files::Vector{AbstractString},step)
        new(files,@build_steps (step,) )
    end
end
FileRule(files::Vector{T},step) where {T <: AbstractString} = FileRule(AbstractString[f for f in files],step)
function lower(s::ChangeDirectory,collection)
    if !isempty(collection.steps)
        error("Change of directory must be the first instruction")
    end
    collection.cwd = s.dir
end
lower(s::Nothing,collection) = nothing
lower(s::Function,collection) = push!(collection,s)
lower(s::CreateDirectory,collection) = @dependent_steps ( DirectoryRule(s.dest,()->(mkpath(s.dest))), )
lower(s::RemoveDirectory,collection) = @dependent_steps ( `rm -rf $(s.dest)` )
lower(s::BuildStep,collection) = push!(collection,s)
lower(s::Base.AbstractCmd,collection) = push!(collection,s)
lower(s::FileDownloader,collection) = @dependent_steps ( CreateDirectory(dirname(s.dest),true), ()->info("Downloading file $(s.src)"), FileRule(s.dest,download_cmd(s.src,s.dest)), ()->info("Done downloading file $(s.src)") )
lower(s::ChecksumValidator,collection) = isempty(s.sha) || @dependent_steps ()->sha_check(s.path, s.sha)
function splittarpath(path)
    path,extension = splitext(path)
    base_filename,secondary_extension = splitext(path)
    if extension == ".tgz" || extension == ".tbz" || extension == ".zip" && !isempty(secondary_extension)
        base_filename *= secondary_extension
        secondary_extension = ""
    end
    (base_filename,extension,secondary_extension)
end
function lower(s::FileUnpacker,collection)
    base_filename,extension,secondary_extension = splittarpath(s.src)
    target = if isempty(s.target)
        basename(base_filename)
    elseif s.target == "."
        ""
    else
        s.target
    end
    @dependent_steps begin
        CreateDirectory(dirname(s.dest),true)
        PathRule(joinpath(s.dest,target),unpack_cmd(s.src,s.dest,extension,secondary_extension))
    end
end
function adjust_env(env)
    ret = similar(env)
    merge!(ret,ENV)
    merge!(ret,env) # s.env overrides ENV
    ret
end
if Compat.Sys.isunix()
    function lower(a::MakeTargets,collection)
        cmd = `make -j8`
        if Sys.KERNEL == :FreeBSD
            jobs = readchomp(`make -V MAKE_JOBS_NUMBER`)
            if isempty(jobs)
                jobs = readchomp(`sysctl -n hw.ncpu`)
            end
            # Tons of project have written their Makefile in GNU Make only syntax,
            # but the implementation of `make` on FreeBSD system base is `bmake`
            cmd = `gmake -j$jobs`
        end
        if !isempty(a.dir)
            cmd = `$cmd -C $(a.dir)`
        end
        if !isempty(a.targets)
            cmd = `$cmd $(a.targets)`
        end
        @dependent_steps ( setenv(cmd, adjust_env(a.env)), )
    end
end
Compat.Sys.iswindows() && (lower(a::MakeTargets,collection) = @dependent_steps ( setenv(`make $(!isempty(a.dir) ? "-C "*a.dir : "") $(a.targets)`, adjust_env(a.env)), ))
lower(s::SynchronousStepCollection,collection) = (collection|=s)
lower(s) = (c=SynchronousStepCollection();lower(s,c);c)
function lower(s::AutotoolsDependency,collection)
    prefix = s.prefix
    if Compat.Sys.iswindows()
        prefix = replace(replace(s.prefix, "\\" => "/"), "C:/" => "/c/")
    end
    cmdstring = "pwd && ./configure --prefix=$(prefix) "*join(s.configure_options," ")
    env = adjust_env(s.env)
    for path in s.include_dirs
        if !haskey(env,"CPPFLAGS")
            env["CPPFLAGS"] = ""
        end
        env["CPPFLAGS"]*=" -I$path"
    end
    for path in s.lib_dirs
        if !haskey(env,"LDFLAGS")
            env["LDFLAGS"] = ""
        end
        env["LDFLAGS"]*=" -L$path"
    end
    for path in s.rpath_dirs
        if !haskey(env,"LDFLAGS")
            env["LDFLAGS"] = ""
        end
        env["LDFLAGS"]*=" -Wl,-rpath -Wl,$path"
    end
    if s.force_rebuild
        @dependent_steps begin
            RemoveDirectory(s.builddir)
        end
    end
    @static if Compat.Sys.isunix()
        @dependent_steps begin
            CreateDirectory(s.builddir)
            begin
                ChangeDirectory(s.builddir)
                FileRule(isempty(s.config_status_dir) ? "config.status" : joinpath(s.config_status_dir,"config.status"), setenv(`$(s.src)/configure $(s.configure_options) --prefix=$(prefix)`,env))
                FileRule(s.libtarget,MakeTargets(;env=s.env))
                MakeTargets("install";env=env)
            end
        end
    end
    @static if Compat.Sys.iswindows()
        @dependent_steps begin
            ChangeDirectory(s.src)
            FileRule(isempty(s.config_status_dir) ? "config.status" : joinpath(s.config_status_dir,"config.status"),setenv(`sh -c $cmdstring`,env))
            FileRule(s.libtarget,MakeTargets())
            MakeTargets("install")
        end
    end
end
function run(f::Function)
    f()
end
function run(s::FileRule)
    if !any(map(isfile,s.file))
        run(s.step)
        if !any(map(isfile,s.file))
            error("File $(s.file) was not created successfully (Tried to run $(s.step) )")
        end
    end
end
function run(s::DirectoryRule)
    info("Attempting to create directory $(s.dir)")
    if !isdir(s.dir)
        run(s.step)
        if !isdir(s.dir)
            error("Directory $(s.dir) was not created successfully (Tried to run $(s.step) )")
        end
    else
        info("Directory $(s.dir) already exists")
    end
end
function run(s::PathRule)
    if !ispath(s.path)
        run(s.step)
        if !ispath(s.path)
            error("Path $(s.path) was not created successfully (Tried to run $(s.step) )")
        end
    else
        info("Path $(s.path) already exists")
    end
end
function run(s::BuildStep)
    error("Unimplemented BuildStep: $(typeof(s))")
end
function run(s::SynchronousStepCollection)
    for x in s.steps
        if !isempty(s.cwd)
            info("Changing directory to $(s.cwd)")
            cd(s.cwd)
        end
        run(x)
        if !isempty(s.oldcwd)
            info("Changing directory to $(s.oldcwd)")
            cd(s.oldcwd)
        end
    end
end
const MAKE_CMD = Compat.Sys.isbsd() && !Compat.Sys.isapple() ? `gmake` : `make`
function prepare_src(depsdir,url, downloaded_file, directory_name)
    local_file = joinpath(joinpath(depsdir,"downloads"),downloaded_file)
    @build_steps begin
        FileDownloader(url,local_file)
        FileUnpacker(local_file,joinpath(depsdir,"src"),directory_name)
    end
end
function autotools_install(depsdir,url, downloaded_file, configure_opts, directory_name, directory, libname, installed_libname, confstatusdir)
    prefix = joinpath(depsdir,"usr")
    libdir = joinpath(prefix,"lib")
    srcdir = joinpath(depsdir,"src",directory)
    dir = joinpath(joinpath(depsdir,"builds"),directory)
    prepare_src(depsdir,url, downloaded_file,directory_name) |
    @build_steps begin
        AutotoolsDependency(srcdir=srcdir,prefix=prefix,builddir=dir,configure_options=configure_opts,libtarget=libname,installed_libpath=[joinpath(libdir,installed_libname)],config_status_dir=confstatusdir)
    end
end
autotools_install(depsdir,url, downloaded_file, configure_opts, directory_name, directory, libname, installed_libname) = autotools_install(depsdir,url, downloaded_file, configure_opts, directory_name, directory, libname, installed_libname, "")
autotools_install(depsdir,url, downloaded_file, configure_opts, directory, libname)=autotools_install(depsdir,url,downloaded_file,configure_opts,directory,directory,libname,libname)
autotools_install(args...) = error("autotools_install has been removed")
function eval_anon_module(context, file)
    m = Module(:__anon__)
    if isdefined(Base, Symbol("@__MODULE__"))
        eval(m, :(ARGS=[$context]))
        Base.include(m, file)
    else
        body = Expr(:toplevel, :(ARGS=[$context]), :(include($file)))
        eval(m, body)
    end
    return
end
"""
    glibc_version()
For Linux-based systems, return the version of glibc in use. For non-glibc Linux and
other platforms, returns `nothing`.
"""
function glibc_version()
    Compat.Sys.islinux() || return
    libc = ccall(:jl_dlopen, Ptr{Cvoid}, (Ptr{Cvoid}, UInt32), C_NULL, 0)
    ptr = Libdl.dlsym_e(libc, :gnu_get_libc_version)
    ptr == C_NULL && return # non-glibc
    v = unsafe_string(ccall(ptr, Ptr{UInt8}, ()))
    contains(v, Base.VERSION_REGEX) ? VersionNumber(v) : nothing
end
import Base: show
const OSNAME = Compat.Sys.iswindows() ? :Windows : Sys.KERNEL
if !isdefined(Base, :pairs)
    pairs(x) = (a => b for (a, b) in x)
end
abstract type DependencyProvider end
abstract type DependencyHelper end
mutable struct PackageContext
    do_install::Bool
    dir::AbstractString
    package::AbstractString
    deps::Vector{Any}
end
mutable struct LibraryDependency
    name::AbstractString
    context::PackageContext
    providers::Vector{Tuple{DependencyProvider,Dict{Symbol,Any}}}
    helpers::Vector{Tuple{DependencyHelper,Dict{Symbol,Any}}}
    properties::Dict{Symbol,Any}
    libvalidate::Function
end
mutable struct LibraryGroup
    name::AbstractString
    deps::Vector{LibraryDependency}
end
pkgdir(dep) = dep.context.dir
depsdir(dep) = joinpath(pkgdir(dep),"deps")
usrdir(dep) = joinpath(depsdir(dep),"usr")
libdir(dep) = joinpath(usrdir(dep),"lib")
bindir(dep) = joinpath(usrdir(dep),"bin")
includedir(dep) = joinpath(usrdir(dep),"include")
builddir(dep) = joinpath(depsdir(dep),"builds")
downloadsdir(dep) = joinpath(depsdir(dep),"downloads")
srcdir(dep) = joinpath(depsdir(dep),"src")
libdir(provider, dep) = [libdir(dep), libdir(dep)*"32", libdir(dep)*"64"]
bindir(provider, dep) = bindir(dep)
successful_validate(l,p) = true
function _library_dependency(context::PackageContext, name; props...)
    validate = successful_validate
    group = nothing
    properties = collect(pairs(props))
    for i in 1:length(properties)
        k,v = properties[i]
        if k == :validate
            validate = v
            splice!(properties,i)
        end
        if k == :group
            group = v
        end
    end
    r = LibraryDependency(name, context, Tuple{DependencyProvider,Dict{Symbol,Any}}[], Tuple{DependencyHelper,Dict{Symbol,Any}}[], Dict{Symbol,Any}(properties), validate)
    if group !== nothing
        push!(group.deps,r)
    else
        push!(context.deps,r)
    end
    r
end
function _library_group(context,name)
    r = LibraryGroup(name,LibraryDependency[])
    push!(context.deps,r)
    r
end
macro setup()
    dir = normpath(joinpath(pwd(),".."))
    package = basename(dir)
    esc(quote
        if length(ARGS) > 0 && isa(ARGS[1],BinDeps.PackageContext)
            bindeps_context = ARGS[1]
        else
            bindeps_context = BinDeps.PackageContext(true,$dir,$package,Any[])
        end
        library_group(args...) = BinDeps._library_group(bindeps_context,args...)
        library_dependency(args...; properties...) = BinDeps._library_dependency(bindeps_context,args...;properties...)
    end)
end
export library_dependency, bindir, srcdir, usrdir, libdir
library_dependency(args...; properties...) = error("No context provided. Did you forget `@BinDeps.setup`?")
abstract type PackageManager <: DependencyProvider end
const DEBIAN_VERSION_REGEX = r"^
    ([0-9]+\:)?                                           # epoch
    (?:(?:([0-9][a-z0-9.\-+:~]*)-([0-9][a-z0-9.+~]*)) |   # upstream version + debian revision
          ([0-9][a-z0-9.+:~]*))                           # upstream version
"ix
const has_apt = try success(`apt-get -v`) && success(`apt-cache -v`) catch e false end
mutable struct AptGet <: PackageManager
    package::AbstractString
end
can_use(::Type{AptGet}) = has_apt && Compat.Sys.islinux()
package_available(p::AptGet) = can_use(AptGet) && !isempty(available_versions(p))
function available_versions(p::AptGet)
    vers = String[]
    lookfor_version = false
    for l in eachline(`apt-cache showpkg $(p.package)`)
        if startswith(l,"Version:")
            try
                vs = l[(1+length("Version: ")):end]
                push!(vers, vs)
            end
        elseif lookfor_version && (m = match(DEBIAN_VERSION_REGEX, l)) !== nothing
            m.captures[2] !== nothing ? push!(vers, m.captures[2]) :
                                       push!(vers, m.captures[4])
        elseif startswith(l, "Versions:")
            lookfor_version = true
        elseif startswith(l, "Reverse Depends:")
            lookfor_version = false
        end
    end
    return vers
end
function available_version(p::AptGet)
    vers = available_versions(p)
    isempty(vers) && error("apt-cache did not return version information. This shouldn't happen. Please file a bug!")
    length(vers) > 1 && warn("Multiple versions of $(p.package) are available.  Use BinDeps.available_versions to get all versions.")
    return vers[end]
end
pkg_name(a::AptGet) = a.package
libdir(p::AptGet,dep) = ["/usr/lib", "/usr/lib64", "/usr/lib32", "/usr/lib/x86_64-linux-gnu", "/usr/lib/i386-linux-gnu"]
const has_yum = try success(`yum --version`) catch e false end
mutable struct Yum <: PackageManager
    package::AbstractString
end
can_use(::Type{Yum}) = has_yum && Compat.Sys.islinux()
package_available(y::Yum) = can_use(Yum) && success(`yum list $(y.package)`)
function available_version(y::Yum)
    uname = readchomp(`uname -m`)
    found_uname = false
    found_version = false
    for l in eachline(`yum info $(y.package)`)
        VERSION < v"0.6" && (l = chomp(l))
        if !found_uname
            # On 64-bit systems, we may have multiple arches installed
            # this makes sure we get the right one
            found_uname = endswith(l, uname)
            continue
        end
        if startswith(l, "Version")
            return convert(VersionNumber, split(l)[end])
        end
    end
    error("yum did not return version information.  This shouldn't happen. Please file a bug!")
end
pkg_name(y::Yum) = y.package
const has_pacman = try success(`pacman -Qq`) catch e false end
mutable struct Pacman <: PackageManager
    package::AbstractString
end
can_use(::Type{Pacman}) = has_pacman && Compat.Sys.islinux()
package_available(p::Pacman) = can_use(Pacman) && success(`pacman -Si $(p.package)`)
function available_version(p::Pacman)
    for l in eachline(`/usr/bin/pacman -Si $(p.package)`) # To circumvent alias problems
        if startswith(l, "Version")
            # The following isn't perfect, but it's hopefully less brittle than
            # writing a regex for pacman's nonexistent version-string standard.
            # This also strips away the sometimes leading epoch as in ffmpeg's
            # Version        : 1:2.3.3-1
            versionstr = strip(split(l, ":")[end])
            try
                return convert(VersionNumber, versionstr)
            catch e
                # For too long versions like imagemagick's 6.8.9.6-1, give it
                # a second try just discarding superfluous stuff.
                return convert(VersionNumber, join(split(versionstr, '.')[1:3], '.'))
            end
        end
    end
    error("pacman did not return version information. This shouldn't happen. Please file a bug!")
end
pkg_name(p::Pacman) = p.package
libdir(p::Pacman,dep) = ["/usr/lib", "/usr/lib32"]
const has_zypper = try success(`zypper --version`) catch e false end
mutable struct Zypper <: PackageManager
    package::AbstractString
end
can_use(::Type{Zypper}) = has_zypper && Compat.Sys.islinux()
package_available(z::Zypper) = can_use(Zypper) && success(`zypper se $(z.package)`)
function available_version(z::Zypper)
    uname = readchomp(`uname -m`)
    found_uname = false
    ENV2 = copy(ENV)
    ENV2["LC_ALL"] = "C"
    for l in eachline(setenv(`zypper info $(z.package)`, ENV2))
        VERSION < v"0.6" && (l = chomp(l))
        if !found_uname
            found_uname = endswith(l, uname)
            continue
        end
        if startswith(l, "Version:")
            versionstr = strip(split(l, ":")[end])
            return convert(VersionNumber, versionstr)
        end
    end
    error("zypper did not return version information.  This shouldn't happen. Please file a bug!")
end
pkg_name(z::Zypper) = z.package
libdir(z::Zypper,dep) = ["/usr/lib", "/usr/lib32", "/usr/lib64"]
const has_bsdpkg = try success(`pkg -v`) catch e false end
mutable struct BSDPkg <: PackageManager
    package::AbstractString
end
can_use(::Type{BSDPkg}) = has_bsdpkg && Sys.KERNEL === :FreeBSD
function package_available(p::BSDPkg)
    can_use(BSDPkg) || return false
    rgx = Regex(string("^(", p.package, ")(\\s+.+)?\$"))
    for line in eachline(`pkg search -L name $(p.package)`)
        contains(line, rgx) && return true
    end
    return false
end
function available_version(p::BSDPkg)
    looknext = false
    for line in eachline(`pkg search -L name -Q version $(p.package)`)
        if rstrip(line) == p.package
            looknext = true
            continue
        end
        if looknext && startswith(line, "Version")
            # Package versioning is [SOFTWARE VERSION]_[PORT REVISION],[PORT EPOCH]
            # In our case we only care about the software version, not the port revision
            # or epoch. The software version should be recognizable as semver-ish.
            rawversion = chomp(line[findfirst(c->c==':', line)+2:end])
            # Chop off the port revision and epoch by removing everything after and
            # including the first underscore
            libversion = replace(rawversion, r"_.+$" => "")
            # This should be a valid version, but it's still possible that it isn't
            if contains(libversion, Base.VERSION_REGEX)
                return VersionNumber(libversion)
            else
                error("\"$rawversion\" is not recognized as a version. Please report this to BinDeps.jl.")
            end
        end
    end
    error("pkg did not return version information. This should not happen. Please file a bug!")
end
pkg_name(p::BSDPkg) = p.package
libdir(p::BSDPkg, dep) = ["/usr/local/lib"]
can_use(::Type) = true
abstract type Sources <: DependencyHelper end
abstract type Binaries <: DependencyProvider end
struct SystemPaths <: DependencyProvider; end
show(io::IO, ::SystemPaths) = print(io,"System Paths")
using URIParser
export URI
mutable struct NetworkSource <: Sources
    uri::URI
end
srcdir(s::Sources, dep::LibraryDependency) = srcdir(dep,s,Dict{Symbol,Any}())
function srcdir( dep::LibraryDependency, s::NetworkSource,opts)
    joinpath(srcdir(dep),get(opts,:unpacked_dir,splittarpath(basename(s.uri.path))[1]))
end
mutable struct RemoteBinaries <: Binaries
    uri::URI
end
mutable struct CustomPathBinaries <: Binaries
    path::AbstractString
end
libdir(p::CustomPathBinaries,dep) = p.path
abstract type BuildProcess <: DependencyProvider end
mutable struct SimpleBuild <: BuildProcess
    steps
end
mutable struct Autotools <: BuildProcess
    source
    opts
end
mutable struct GetSources <: BuildStep
    dep::LibraryDependency
end
lower(x::GetSources,collection) = push!(collection,generate_steps(x.dep,gethelper(x.dep,Sources)...))
Autotools(;opts...) = Autotools(nothing, Dict{Any,Any}(pairs(opts)))
export AptGet, Yum, Pacman, Zypper, BSDPkg, Sources, Binaries, provides, BuildProcess, Autotools,
       GetSources, SimpleBuild, available_version
provider(::Type{T},package::AbstractString; opts...) where {T <: PackageManager} = T(package)
provider(::Type{Sources},uri::URI; opts...) = NetworkSource(uri)
provider(::Type{Binaries},uri::URI; opts...) = RemoteBinaries(uri)
provider(::Type{Binaries},path::AbstractString; opts...) = CustomPathBinaries(path)
provider(::Type{SimpleBuild},steps; opts...) = SimpleBuild(steps)
provider(::Type{BuildProcess},p::T; opts...) where {T <: BuildProcess} = provider(T,p; opts...)
provider(::Type{BuildProcess},steps::Union{BuildStep,SynchronousStepCollection}; opts...) = provider(SimpleBuild,steps; opts...)
provider(::Type{Autotools},a::Autotools; opts...) = a
provides(provider::DependencyProvider,dep::LibraryDependency; opts...) = push!(dep.providers,(provider,Dict{Symbol,Any}(pairs(opts))))
provides(helper::DependencyHelper,dep::LibraryDependency; opts...) = push!(dep.helpers,(helper,Dict{Symbol,Any}(pairs(opts))))
provides(::Type{T},p,dep::LibraryDependency; opts...) where {T} = provides(provider(T,p; opts...),dep; opts...)
function provides(::Type{T},packages::AbstractArray,dep::LibraryDependency; opts...) where {T}
    for p in packages
        provides(T,p,dep; opts...)
    end
end
function provides(::Type{T},ps,deps::Vector{LibraryDependency}; opts...) where {T}
    p = provider(T,ps; opts...)
    for dep in deps
        provides(p,dep; opts...)
    end
end
function provides(::Type{T},providers::Dict; opts...) where {T}
    for (k,v) in providers
        provides(T,k,v;opts...)
    end
end
sudoname(c::Cmd) = c == `` ? "" : "sudo "
const have_sonames = Ref(false)
const sonames = Dict{String,String}()
function reread_sonames()
    if VERSION >= v"0.7.0-DEV.1287" # only use this where julia issue #22832 is fixed
        empty!(sonames)
        have_sonames[] = false
        nothing
    else
        ccall(:jl_read_sonames, Cvoid, ())
    end
end
if Compat.Sys.iswindows() || Compat.Sys.isapple()
    function read_sonames()
        have_sonames[] = true
    end
elseif Compat.Sys.islinux()
    let ldconfig_arch = Dict(:i386 => "x32",
                             :i387 => "x32",
                             :i486 => "x32",
                             :i586 => "x32",
                             :i686 => "x32",
                             :x86_64 => "x86-64",
                             :aarch64 => "AArch64"),
        arch = get(ldconfig_arch, Sys.ARCH, ""),
        arch_wrong = filter!(x -> (x != arch), ["x32", "x86-64", "AArch64", "soft-float"])
    global read_sonames
    function read_sonames()
        empty!(sonames)
        for line in eachline(`/sbin/ldconfig -p`)
            VERSION < v"0.6" && (line = chomp(line))
            m = match(r"^\s+([^ ]+)\.so[^ ]* \(([^)]*)\) => (.+)$", line)
            if m !== nothing
                desc = m[2]
                if Sys.WORD_SIZE != 32 && !isempty(arch)
                    contains(desc, arch) || continue
                end
                for wrong in arch_wrong
                    contains(desc, wrong) && continue
                end
                sonames[m[1]] = m[3]
            end
        end
        have_sonames[] = true
    end
    end
else
    function read_sonames()
        empty!(sonames)
        for line in eachline(`/sbin/ldconfig -r`)
            VERSION < v"0.6" && (line = chomp(line))
            m = match(r"^\s+\d+:-l([^ ]+)\.[^. ]+ => (.+)$", line)
            if m !== nothing
                sonames["lib" * m[1]] = m[2]
            end
        end
        have_sonames[] = true
    end
end
if VERSION >= v"0.7.0-DEV.1287" # only use this where julia issue #22832 is fixed
    lookup_soname(s) = lookup_soname(String(s))
    function lookup_soname(s::String)
        have_sonames[] || read_sonames()
        return get(sonames, s, "")
    end
else
    function lookup_soname(lib)
        if Compat.Sys.islinux() || (Compat.Sys.isbsd() && !Compat.Sys.isapple())
            soname = ccall(:jl_lookup_soname, Ptr{UInt8}, (Ptr{UInt8}, Csize_t), lib, sizeof(lib))
            soname != C_NULL && return unsafe_string(soname)
        end
        return ""
    end
end
generate_steps(h::DependencyProvider,dep::LibraryDependency) = error("Must also pass provider options")
generate_steps(h::BuildProcess,dep::LibraryDependency,opts) = h.steps
function generate_steps(dep::LibraryDependency,h::AptGet,opts)
    if get(opts,:force_rebuild,false)
        error("Will not force apt-get to rebuild dependency \"$(dep.name)\".\n"*
              "Please make any necessary adjustments manually (This might just be a version upgrade)")
    end
    sudo = get(opts, :sudo, has_sudo[]) ? `sudo` : ``
    @build_steps begin
        println("Installing dependency $(h.package) via `$(sudoname(sudo))apt-get install $(h.package)`:")
        `$sudo apt-get install $(h.package)`
        reread_sonames
    end
end
function generate_steps(dep::LibraryDependency,h::Yum,opts)
    if get(opts,:force_rebuild,false)
        error("Will not force yum to rebuild dependency \"$(dep.name)\".\n"*
              "Please make any necessary adjustments manually (This might just be a version upgrade)")
    end
    sudo = get(opts, :sudo, has_sudo[]) ? `sudo` : ``
    @build_steps begin
        println("Installing dependency $(h.package) via `$(sudoname(sudo))yum install $(h.package)`:")
        `$sudo yum install $(h.package)`
        reread_sonames
    end
end
function generate_steps(dep::LibraryDependency,h::Pacman,opts)
    if get(opts,:force_rebuild,false)
        error("Will not force pacman to rebuild dependency \"$(dep.name)\".\n"*
              "Please make any necessary adjustments manually (This might just be a version upgrade)")
    end
    sudo = get(opts, :sudo, has_sudo[]) ? `sudo` : ``
    @build_steps begin
        println("Installing dependency $(h.package) via `$(sudoname(sudo))pacman -S --needed $(h.package)`:")
        `$sudo pacman -S --needed $(h.package)`
        reread_sonames
    end
end
function generate_steps(dep::LibraryDependency,h::Zypper,opts)
    if get(opts,:force_rebuild,false)
        error("Will not force zypper to rebuild dependency \"$(dep.name)\".\n"*
              "Please make any necessary adjustments manually (This might just be a version upgrade)")
    end
    sudo = get(opts, :sudo, has_sudo[]) ? `sudo` : ``
    @build_steps begin
        println("Installing dependency $(h.package) via `$(sudoname(sudo))zypper install $(h.package)`:")
        `$sudo zypper install $(h.package)`
        reread_sonames
    end
end
function generate_steps(dep::LibraryDependency, p::BSDPkg, opts)
    if get(opts, :force_rebuild, false)
        error("Will not force pkg to rebuild dependency \"$(dep.name)\".\n" *
              "Please make any necessary adjustments manually. (This might just be a version upgrade.)")
    end
    sudo = get(opts, :sudo, has_sudo[]) ? `sudo` : ``
    @build_steps begin
        println("Installing dependency $(p.package) via `$(sudoname(sudo))pkg install -y $(p.package)`:`")
        `$sudo pkg install -y $(p.package)`
        reread_sonames
    end
end
function generate_steps(dep::LibraryDependency,h::NetworkSource,opts)
    localfile = joinpath(downloadsdir(dep),get(opts,:filename,basename(h.uri.path)))
    @build_steps begin
        FileDownloader(string(h.uri),localfile)
        ChecksumValidator(get(opts,:SHA,get(opts,:sha,"")),localfile)
        CreateDirectory(srcdir(dep))
        FileUnpacker(localfile,srcdir(dep),srcdir(dep,h,opts))
    end
end
function generate_steps(dep::LibraryDependency,h::RemoteBinaries,opts)
    get(opts,:force_rebuild,false) && error("Force rebuild not allowed for binaries. Use a different download location instead.")
    localfile = joinpath(downloadsdir(dep),get(opts,:filename,basename(h.uri.path)))
    # choose the destination to unpack into and the folder/file to validate
    (dest, target) = if haskey(opts, :unpacked_dir)
        if opts[:unpacked_dir] == "."
            # if the archive dumps right in the root dir, create a subdir
            (joinpath(depsdir(dep), dep.name), ".")
        else
            (depsdir(dep), opts[:unpacked_dir])
        end
    else
        (depsdir(dep), "usr")
    end
    steps = @build_steps begin
        FileDownloader(string(h.uri),localfile)
        ChecksumValidator(get(opts,:SHA,get(opts,:sha,"")),localfile)
        FileUnpacker(localfile,dest,target)
    end
end
generate_steps(dep::LibraryDependency,h::SimpleBuild,opts) = h.steps
function getoneprovider(dep::LibraryDependency,method)
    for (p,opts) = dep.providers
        if typeof(p) <: method && can_use(typeof(p))
            return (p,opts)
        end
    end
    return (nothing,nothing)
end
function getallproviders(dep::LibraryDependency,method)
    ret = Any[]
    for (p,opts) = dep.providers
        if typeof(p) <: method && can_use(typeof(p))
            push!(ret,(p,opts))
        end
    end
    ret
end
function gethelper(dep::LibraryDependency,method)
    for (p,opts) = dep.helpers
        if typeof(p) <: method
            return (p,opts)
        end
    end
    return (nothing,nothing)
end
stringarray(s::AbstractString) = [s]
stringarray(s) = s
function generate_steps(dep::LibraryDependency,method)
    (p,opts) = getoneprovider(dep,method)
    p !== nothing && return generate_steps(p,dep,opts)
    (p,hopts) = gethelper(dep,method)
    p !== nothing && return generate_steps(p,dep,hopts)
    error("No provider or helper for method $method found for dependency $(dep.name)")
end
function generate_steps(dep::LibraryDependency, h::Autotools,  provider_opts)
    if h.source === nothing
        h.source = gethelper(dep,Sources)
    end
    if isa(h.source,Sources)
        h.source = (h.source,Dict{Symbol,Any}())
    end
    h.source[1] === nothing && error("Could not obtain sources for dependency $(dep.name)")
    steps = lower(generate_steps(dep,h.source...))
    opts = Dict()
    opts[:srcdir]   = srcdir(dep,h.source...)
    opts[:prefix]   = usrdir(dep)
    opts[:builddir] = joinpath(builddir(dep),dep.name)
    merge!(opts,h.opts)
    if haskey(opts,:installed_libname)
        !haskey(opts,:installed_libpath) || error("Can't specify both installed_libpath and installed_libname")
        opts[:installed_libpath] = String[joinpath(libdir(dep),opts[:installed_libname])]
        delete!(opts, :installed_libname)
    elseif !haskey(opts,:installed_libpath)
        opts[:installed_libpath] = String[joinpath(libdir(dep),x)*"."*Libdl.dlext for x in stringarray(get(dep.properties,:aliases,String[]))]
    end
    if !haskey(opts,:libtarget) && haskey(dep.properties,:aliases)
        opts[:libtarget] = String[x*"."*Libdl.dlext for x in stringarray(dep.properties[:aliases])]
    end
    if !haskey(opts,:include_dirs)
        opts[:include_dirs] = AbstractString[]
    end
    if !haskey(opts,:lib_dirs)
        opts[:lib_dirs] = AbstractString[]
    end
    if !haskey(opts,:pkg_config_dirs)
        opts[:pkg_config_dirs] = AbstractString[]
    end
    if !haskey(opts,:rpath_dirs)
        opts[:rpath_dirs] = AbstractString[]
    end
    if haskey(opts,:configure_subdir)
        opts[:srcdir] = joinpath(opts[:srcdir],opts[:configure_subdir])
        delete!(opts, :configure_subdir)
    end
    pushfirst!(opts[:include_dirs],includedir(dep))
    pushfirst!(opts[:lib_dirs],libdir(dep))
    pushfirst!(opts[:rpath_dirs],libdir(dep))
    pushfirst!(opts[:pkg_config_dirs],joinpath(libdir(dep),"pkgconfig"))
    env = Dict{String,String}()
    env["PKG_CONFIG_PATH"] = join(opts[:pkg_config_dirs],":")
    delete!(opts,:pkg_config_dirs)
    if Compat.Sys.isunix()
        env["PATH"] = bindir(dep)*":"*ENV["PATH"]
    elseif Compat.Sys.iswindows()
        env["PATH"] = bindir(dep)*";"*ENV["PATH"]
    end
    haskey(opts,:env) && merge!(env,opts[:env])
    opts[:env] = env
    if get(provider_opts,:force_rebuild,false)
        opts[:force_rebuild] = true
    end
    steps |= AutotoolsDependency(;opts...)
    steps
end
const EXTENSIONS = ["", "." * Libdl.dlext]
function _find_library(dep::LibraryDependency; provider = Any)
    ret = Any[]
    # Same as find_library, but with extra check defined by dep
    libnames = [dep.name;get(dep.properties,:aliases,String[])]
    # Make sure we keep the defaults first, but also look in the other directories
    providers = unique([reduce(vcat,[getallproviders(dep,p) for p in defaults]);dep.providers])
    for (p,opts) in providers
        (p !== nothing && can_use(typeof(p)) && can_provide(p,opts,dep)) || continue
        paths = AbstractString[]
        # Allow user to override installation path
        if haskey(opts,:installed_libpath) && isdir(opts[:installed_libpath])
            pushfirst!(paths,opts[:installed_libpath])
        end
        ppaths = libdir(p,dep)
        append!(paths,isa(ppaths,Array) ? ppaths : [ppaths])
        if haskey(opts,:unpacked_dir)
            dir = opts[:unpacked_dir]
            if dir == "." && isdir(joinpath(depsdir(dep), dep.name))
                # the archive unpacks into the root, so we created a subdir with the dep name
                push!(paths, joinpath(depsdir(dep), dep.name))
            elseif isdir(joinpath(depsdir(dep),dir))
                push!(paths,joinpath(depsdir(dep),dir))
            end
        end
        # Windows, do you know what `lib` stands for???
        if Compat.Sys.iswindows()
            push!(paths,bindir(p,dep))
        end
        (isempty(paths) || all(map(isempty,paths))) && continue
        for lib in libnames, path in paths
            l = joinpath(path, lib)
            h = Libdl.dlopen_e(l, Libdl.RTLD_LAZY)
            if h != C_NULL
                works = dep.libvalidate(l,h)
                l = Libdl.dlpath(h)
                Libdl.dlclose(h)
                if works
                    push!(ret, ((p, opts), l))
                else
                    # We tried to load this providers' library, but it didn't satisfy
                    # the requirements, so tell it to force a rebuild since the requirements
                    # have most likely changed
                    opts[:force_rebuild] = true
                end
            end
        end
    end
    # Now check system libraries
    for lib in libnames
        # We don't want to use regular dlopen, because we want to get at
        # system libraries even if one of our providers is higher in the
        # DL_LOAD_PATH
        for path in Libdl.DL_LOAD_PATH
            for ext in EXTENSIONS
                opath = string(joinpath(path,lib),ext)
                check_path!(ret,dep,opath)
            end
        end
        for ext in EXTENSIONS
            opath = string(lib,ext)
            check_path!(ret,dep,opath)
        end
        soname = lookup_soname(lib)
        isempty(soname) || check_path!(ret, dep, soname)
    end
    return ret
end
function check_path!(ret, dep, opath)
    flags = Libdl.RTLD_LAZY
    handle = ccall(:jl_dlopen, Ptr{Cvoid}, (Cstring, Cuint), opath, flags)
    try
        check_system_handle!(ret, dep, handle)
    finally
        handle != C_NULL && Libdl.dlclose(handle)
    end
end
function check_system_handle!(ret,dep,handle)
    if handle != C_NULL
        libpath = Libdl.dlpath(handle)
        # Check that this is not a duplicate
        for p in ret
            try
                if realpath(p[2]) == realpath(libpath)
                    return
                end
            catch
                warn("""
                    Found a library that does not exist.
                    This may happen if the library has an active open handle.
                    Please quit julia and try again.
                    """)
                return
            end
        end
        works = dep.libvalidate(libpath,handle)
        if works
            push!(ret, ((SystemPaths(),Dict()), libpath))
        end
    end
end
defaults = if Compat.Sys.isapple()
    [Binaries, PackageManager, SystemPaths, BuildProcess]
elseif Compat.Sys.isbsd() || (Compat.Sys.islinux() && glibc_version() === nothing) # non-glibc
    [PackageManager, SystemPaths, BuildProcess]
elseif Compat.Sys.islinux() # glibc
    [PackageManager, SystemPaths, Binaries, BuildProcess]
elseif Compat.Sys.iswindows()
    [Binaries, PackageManager, SystemPaths]
else
    [SystemPaths, BuildProcess]
end
function applicable(dep::LibraryDependency)
    if haskey(dep.properties,:os)
        if (dep.properties[:os] != OSNAME && dep.properties[:os] != :Unix) || (dep.properties[:os] == :Unix && !Compat.Sys.isunix())
            return false
        end
    elseif haskey(dep.properties,:runtime) && dep.properties[:runtime] == false
        return false
    end
    return true
end
applicable(deps::LibraryGroup) = any([applicable(dep) for dep in deps.deps])
function can_provide(p,opts,dep)
    if p === nothing || (haskey(opts,:os) && opts[:os] != OSNAME && (opts[:os] != :Unix || !Compat.Sys.isunix()))
        return false
    end
    if !haskey(opts,:validate)
        return true
    elseif isa(opts[:validate],Bool)
        return opts[:validate]
    else
        return opts[:validate](p,dep)
    end
end
function can_provide(p::PackageManager,opts,dep)
    if p === nothing || (haskey(opts,:os) && opts[:os] != OSNAME && (opts[:os] != :Unix || !Compat.Sys.isunix()))
        return false
    end
    if !package_available(p)
        return false
    end
    if !haskey(opts,:validate)
        return true
    elseif isa(opts[:validate],Bool)
        return opts[:validate]
    else
        return opts[:validate](p,dep)
    end
end
issatisfied(dep::LibraryDependency) = !isempty(_find_library(dep))
allf(deps) = Dict([(dep, _find_library(dep)) for dep in deps.deps])
function satisfied_providers(deps::LibraryGroup, allfl = allf(deps))
    viable_providers = nothing
    for dep in deps.deps
        if !applicable(dep)
            continue
        end
        providers = map(x->typeof(x[1][1]),allfl[dep])
        if viable_providers == nothing
            viable_providers = providers
        else
            viable_providers = intersect(viable_providers,providers)
        end
    end
    viable_providers
end
function viable_providers(deps::LibraryGroup)
    vp = nothing
    for dep in deps.deps
        if !applicable(dep)
            continue
        end
        providers = map(x->typeof(x[1]),dep.providers)
        if vp === nothing
            vp = providers
        else
            vp = intersect(vp,providers)
        end
    end
    vp
end
issatisfied(deps::LibraryGroup) = !isempty(satisfied_providers(deps))
function _find_library(deps::LibraryGroup, allfl = allf(deps); provider = Any)
    providers = satisfied_providers(deps,allfl)
    p = nothing
    if isempty(providers)
        return Dict()
    else
        for p2 in providers
            if p2 <: provider
                p = p2
            end
        end
    end
    p === nothing && error("Given provider does not satisfy the library group")
    Dict([(dep, begin
        thisfl = allfl[dep]
        ret = nothing
        for fl in thisfl
            if isa(fl[1][1],p)
                ret = fl
                break
            end
        end
        @assert ret != nothing
        ret
    end) for dep in filter(applicable,deps.deps)])
end
function satisfy!(deps::LibraryGroup, methods = defaults)
    sp = satisfied_providers(deps)
    if !isempty(sp)
        for m in methods
            for s in sp
                if s <: m
                    return s
                end
            end
        end
    end
    if !applicable(deps)
        return Any
    end
    vp = viable_providers(deps)
    didsatisfy = false
    for method in methods
        for p in vp
            if !(p <: method) || !can_use(p)
                continue
            end
            skip = false
            for dep in deps.deps
                !applicable(dep) && continue
                hasany = false
                for (p2,opts) in getallproviders(dep,p)
                    can_provide(p2, opts, dep) && (hasany = true)
                end
                if !hasany
                    skip = true
                    break
                end
            end
            if skip
                continue
            end
            for dep in deps.deps
                satisfy!(dep,[p])
            end
            return p
        end
    end
    error("""
        None of the selected providers could satisfy library group $(deps.name)
        Use BinDeps.debug(package_name) to see available providers
        """)
end
function satisfy!(dep::LibraryDependency, methods = defaults)
    sp = map(x->typeof(x[1][1]),_find_library(dep))
    if !isempty(sp)
        for m in methods
            for s in sp
                if s <: m
                    return s
                end
            end
        end
    end
    if !applicable(dep)
        return
    end
    for method in methods
        for (p,opts) in getallproviders(dep,method)
            can_provide(p,opts,dep) || continue
            if haskey(opts,:force_depends)
                for (dmethod,ddep) in opts[:force_depends]
                    (dp,dopts) = getallproviders(ddep,dmethod)[1]
                    run(lower(generate_steps(ddep,dp,dopts)))
                end
            end
            run(lower(generate_steps(dep,p,opts)))
            !issatisfied(dep) && error("Provider $method failed to satisfy dependency $(dep.name)")
            return p
        end
    end
    error("""
        None of the selected providers can install dependency $(dep.name).
        Use BinDeps.debug(package_name) to see available providers
        """)
end
execute(dep::LibraryDependency,method) = run(lower(generate_steps(dep,method)))
macro install(_libmaps...)
    if length(_libmaps) == 0
        return esc(quote
            if bindeps_context.do_install
                for d in bindeps_context.deps
                    BinDeps.satisfy!(d)
                end
            end
        end)
    else
        libmaps = eval(_libmaps[1])
        load_cache = gensym()
        ret = Expr(:block)
        push!(ret.args,
            esc(quote
                    load_cache = Dict()
                    pre_hooks = Set{$AbstractString}()
                    load_hooks = Set{$AbstractString}()
                    if bindeps_context.do_install
                        for d in bindeps_context.deps
                            p = BinDeps.satisfy!(d)
                            libs = BinDeps._find_library(d; provider = p)
                            if isa(d, BinDeps.LibraryGroup)
                                if !isempty(libs)
                                    for dep in d.deps
                                        !BinDeps.applicable(dep) && continue
                                        if !haskey(load_cache, dep.name)
                                            load_cache[dep.name] = libs[dep][2]
                                            opts = libs[dep][1][2]
                                            haskey(opts, :preload) && push!(pre_hooks,opts[:preload])
                                            haskey(opts, :onload) && push!(load_hooks,opts[:onload])
                                        end
                                    end
                                end
                            else
                                for (k,v) in libs
                                    if !haskey(load_cache, d.name)
                                        load_cache[d.name] = v
                                        opts = k[2]
                                        haskey(opts, :preload) && push!(pre_hooks,opts[:preload])
                                        haskey(opts, :onload) && push!(load_hooks,opts[:onload])
                                    end
                                end
                            end
                        end
                        # Generate "deps.jl" file for runtime loading
                        depsfile_location = joinpath(splitdir(Base.source_path())[1],"deps.jl")
                        depsfile_buffer = IOBuffer()
                        println(depsfile_buffer,
                            """
                            # This is an auto-generated file; do not edit
                            """)
                        println(depsfile_buffer, "# Pre-hooks")
                        println(depsfile_buffer, join(pre_hooks, "\n"))
                        println(depsfile_buffer,
                            """
                            if VERSION >= v"0.7.0-DEV.3382"
                                using Libdl
                            end
                            # Macro to load a library
                            macro checked_lib(libname, path)
                                if Libdl.dlopen_e(path) == C_NULL
                                    error("Unable to load \\n\\n\$libname (\$path)\\n\\nPlease ",
                                          "re-run Pkg.build(package), and restart Julia.")
                                end
                                quote
                                    const \$(esc(libname)) = \$path
                                end
                            end
                            """)
                        println(depsfile_buffer, "# Load dependencies")
                        for libkey in keys($libmaps)
                            ((cached = get(load_cache,string(libkey),nothing)) === nothing) && continue
                            println(depsfile_buffer, "@checked_lib ", $libmaps[libkey], " \"", escape_string(cached), "\"")
                        end
                        println(depsfile_buffer)
                        println(depsfile_buffer, "# Load-hooks")
                        println(depsfile_buffer, join(load_hooks,"\n"))
                        depsfile_content = chomp(String(take!(depsfile_buffer)))
                        if !isfile(depsfile_location) || readchomp(depsfile_location) != depsfile_content
                            # only overwrite if deps.jl file does not yet exist or content has changed
                            open(depsfile_location, "w") do depsfile
                                println(depsfile, depsfile_content)
                            end
                        end
                    end
                end))
        if !(typeof(libmaps) <: Associative)
            warn("Incorrect mapping in BinDeps.@install call. No dependencies will be cached.")
        end
        ret
    end
end
macro load_dependencies(args...)
    dir = dirname(normpath(joinpath(dirname(Base.source_path()),"..")))
    arg1 = nothing
    file = "../deps/build.jl"
    if length(args) == 1
        if isa(args[1],Expr)
            arg1 = eval(args[1])
        elseif typeof(args[1]) <: AbstractString
            file = args[1]
            dir = dirname(normpath(joinpath(dirname(file),"..")))
        elseif typeof(args[1]) <: Associative || isa(args[1],Vector)
            arg1 = args[1]
        else
            error("Type $(typeof(args[1])) not recognized for argument 1. See usage instructions!")
        end
    elseif length(args) == 2
        file = args[1]
        arg1 = typeof(args[2]) <: Associative || isa(args[2],Vector) ? args[2] : eval(args[2])
    elseif length(args) != 0
        error("No version of @load_dependencies takes $(length(args)) arguments. See usage instructions!")
    end
    pkg = ""
    r = search(dir,Pkg.Dir.path())
    if r != 0:-1
        s = search(dir,"/",last(r)+2)
        if s != 0:-1
            pkg = dir[(last(r)+2):(first(s)-1)]
        else
            pkg = dir[(last(r)+2):end]
        end
    end
    context = BinDeps.PackageContext(false,dir,pkg,Any[])
    eval_anon_module(context, file)
    ret = Expr(:block)
    for dep in context.deps
        if !applicable(dep)
            continue
        end
        name = sym = dep.name
        if arg1 !== nothing
            if (typeof(arg1) <: Associative) && all(map(x->(x == Symbol || x <: AbstractString),eltype(arg1)))
                found = false
                for need in keys(arg1)
                    found = (dep.name == string(need))
                    if found
                        sym = arg1[need]
                        delete!(arg1,need)
                        break
                    end
                end
                if !found
                    continue
                end
            elseif isa(arg1,Vector) && ((eltype(arg1) == Symbol) || (eltype(arg1) <: AbstractString))
                found = false
                for i = 1:length(args)
                    found = (dep.name == string(arg1[i]))
                    if found
                        sym = arg1[i]
                        splice!(arg1,i)
                        break
                    end
                end
                if !found
                    continue
                end
            elseif isa(arg1,Function)
                if !arg1(name)
                    continue
                end
            else
                error("Can't deal with argument type $(typeof(arg1)). See usage instructions!")
            end
        end
        s = Symbol(sym)
        errorcase = Expr(:block)
        push!(errorcase.args,:(error("Could not load library "*$(dep.name)*". Try running Pkg.build() to install missing dependencies!")))
        push!(ret.args,quote
            const $(esc(s)) = BinDeps._find_library($dep)
            if isempty($(esc(s)))
                $errorcase
            end
        end)
    end
    if arg1 !== nothing && !isa(arg1,Function)
        if !isempty(arg1)
            errrormsg = "The following required libraries were not declared in build.jl:\n"
            for k in (isa(arg1,Vector) ? arg1 : keys(arg1))
                errrormsg *= " - $k\n"
            end
            error(errrormsg)
        end
    end
    ret
end
function build(pkg::AbstractString, method; dep::AbstractString="", force=false)
    dir = Pkg.dir(pkg)
    file = joinpath(dir,"deps/build.jl")
    context = BinDeps.PackageContext(false,dir,pkg,Any[])
    eval_anon_module(context, file)
    for d in context.deps
        BinDeps.satisfy!(d,[method])
    end
end
using SHA
function sha_check(path, sha)
    open(path) do f
        calc_sha = sha256(f)
        # Workaround for SHA.jl API change.  Safe to remove once SHA versions
        # < v0.2.0 are rare, e.g. when Julia v0.4 is deprecated.
        if !isa(calc_sha, AbstractString)
            calc_sha = bytes2hex(calc_sha)
        end
        if calc_sha != sha
            error("Checksum mismatch!  Expected:\n$sha\nCalculated:\n$calc_sha\nDelete $path and try again")
        end
    end
end
import Base: show
function _show_indented(io::IO, dep::LibraryDependency, indent, lib)
    print_indented(io,"- Library \"$(dep.name)\"",indent+1)
    if !applicable(dep)
        println(io," (not applicable to this system)")
    else
        println(io)
        if !isempty(lib)
            print_indented(io,"- Satisfied by:\n",indent+4)
            for (k,v) in lib
                print_indented(io,"- $(k[1]) at $v\n",indent+6)
            end
        end
        if length(dep.providers) > 0
            print_indented(io,"- Providers:\n",indent+4)
            for (p,opts) in dep.providers
                show_indented(io,p,indent+6)
                if !can_provide(p,opts,dep)
                    print(io," (can't provide)")
                end
                println(io)
            end
        end
    end
end
show_indented(io::IO, dep::LibraryDependency, indent) = _show_indented(io,dep,indent, applicable(dep) ? _find_library(dep) : nothing)
show(io::IO, dep::LibraryDependency) = show_indented(io, dep, 0)
function show(io::IO, deps::LibraryGroup)
    print(io," - Library Group \"$(deps.name)\"")
    all = allf(deps)
    providers = satisfied_providers(deps,all)
    if providers !== nothing && !isempty(providers)
        print(io," (satisfied by ",join(providers,", "),")")
    end
    if !applicable(deps)
        println(io," (not applicable to this system)")
    else
        println(io)
        for dep in deps.deps
            _show_indented(io,dep,4,haskey(all,dep) ? all[dep] : nothing)
        end
    end
end
function debug_context(pkg::AbstractString)
    info("Reading build script...")
    dir = Pkg.dir(pkg)
    file = joinpath(dir, "deps", "build.jl")
    context = BinDeps.PackageContext(false, dir, pkg, Any[])
    eval_anon_module(context, file)
    context
end
function debug(io,pkg::AbstractString)
    context = debug_context(pkg)
    println(io,"The package declares $(length(context.deps)) dependencies.")
    # We need to `eval()` the rest of this function because `debug_context()` will
    # `eval()` in things like `Homebrew.jl`, which contain new methods for things
    # like `can_provide()`, and we cannot deal with those new methods in our
    # current world age; we need to `eval()` to force ourselves up into a newer
    # world age.
    @eval for dep in $(context.deps)
        show($io,dep)
    end
end
debug(pkg::AbstractString) = debug(STDOUT,pkg)
print_indented(io::IO,x,indent) = print(io," "^indent,x)
function show_indented(io::IO,x,indent)
    print_indented(io,"- ",indent)
    show(io,x)
end
show(io::IO,x::PackageManager) = print(io,"$(typeof(x)) package $(pkg_name(x))")
show(io::IO,x::SimpleBuild) = print(io,"Simple Build Process")
show(io::IO,x::Sources) = print(io,"Sources")
show(io::IO,x::Binaries) = print(io,"Binaries")
show(io::IO,x::Autotools) = print(io,"Autotools Build")
@Base.deprecate_binding shlib_ext Libdl.dlext
const has_sudo = Ref{Bool}(false)
function __init__()
    has_sudo[] = try success(`sudo -V`) catch err false end
end
end
