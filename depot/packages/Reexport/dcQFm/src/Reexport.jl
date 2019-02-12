module Reexport
macro reexport(ex)
    isa(ex, Expr) && (ex.head == :module ||
                      (ex.head == :toplevel &&
                       all(e->isa(e, Expr) && e.head == :using, ex.args))) ||
    if ex.head == :module
    end
    esc(Expr(:toplevel, ex,
             [:(eval(Expr(:export, names($mod)...))) for mod in modules]...))
end
end # module
