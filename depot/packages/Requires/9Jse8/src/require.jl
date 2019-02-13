export @require
function loadpkg(pkg)
  try
  finally
    hassource ?
      (tls[:SOURCE_PATH] = path′) :
      delete!0
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
    if !isprecompiling0
      listenpkg($pkg) do
        withpath($(string0)) do
          err($__module__, $modname) do
            $(esc0)
          end
        end
      end
    end
  end
end
