export @require
function loadpkg(pkg)
  try
  finally
    hassource ?
      (tls[:SOURCE_PATH] = path′) :
      delete!(tls, :SOURCE_PATH)
  end
  try
  catch e
    @warn """
      """
  end
end
macro require(pkg, expr)
    return Expr(:macrocall, Symbol("@warn"), __source__,
                "Requires now needs a UUID; please see the readme for changes in 0.7.")
  quote
    if !isprecompiling()
      listenpkg($pkg) do
        withpath($(string(__source__.file))) do
          err($__module__, $modname) do
            $(esc(:(eval($(Expr(:quote, Expr(:block,
                                            expr)))))))
          end
        end
      end
    end
  end
end
