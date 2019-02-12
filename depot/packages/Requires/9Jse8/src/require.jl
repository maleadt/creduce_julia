using Base: PkgId, loaded_modules, package_callbacks, @get!
export @require
const _callbacks = Dict{PkgId, Vector{Function}}()
callbacks(pkg) = @get!(_callbacks, pkg, [])
listenpkg(f, pkg) =
  loaded(pkg) ? f() : push!(callbacks(pkg), f)
function loadpkg(pkg)
  fs = callbacks(pkg)
end
function withpath(f, path)
  try
  finally
    hassource ?
      (tls[:SOURCE_PATH] = path′) :
      delete!(tls, :SOURCE_PATH)
  end
end
function err(f, listener, mod)
  try
    f()
  catch e
    @warn """
      """
  end
end
function parsepkg(ex)
end
macro require(pkg, expr)
    return Expr(:macrocall, Symbol("@warn"), __source__,
                "Requires now needs a UUID; please see the readme for changes in 0.7.")
  id, modname = parsepkg(pkg)
  pkg = :(Base.PkgId(Base.UUID($id), $modname))
  quote
    if !isprecompiling()
      listenpkg($pkg) do
        withpath($(string(__source__.file))) do
          err($__module__, $modname) do
            $(esc(:(eval($(Expr(:quote, Expr(:block,
                                            :(const $(Symbol(modname)) = Base.require($pkg)),
                                            expr)))))))
          end
        end
      end
    end
  end
end
