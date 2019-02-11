module RecipesBase
export
    @recipe,
    AbstractLayout
apply_recipe(plotattributes::Dict{Symbol,Any}) = ()
const _debug_recipes = Bool[false]
function debug(v::Bool = true)
end
function _is_arrow_tuple(expr::Expr)
    expr.head == :tuple && !isempty(expr.args) &&
        expr.args[1].head == :(-->)
end
function _equals_symbol(arg::Symbol, sym::Symbol)
end
function _equals_symbol(arg::QuoteNode, sym::Symbol)
    arg.value == sym
end
function get_function_def(func_signature::Expr, args::Vector)
    front = func_signature.args[1]
    if func_signature.head == :where
    elseif func_signature.head == :call
        func = Expr(:call, :(RecipesBase.apply_recipe), esc.([:(plotattributes::Dict{Symbol, Any}); args])...)
        if isa(front, Expr) && front.head == :curly
        else
            func
        end
    end
end
function create_kw_body(func_signature::Expr)
    args = func_signature.args[2:end]
    kw_body = Expr(:block)
    cleanup_body = Expr(:block)
    if isa(args[1], Expr) && args[1].head == :parameters
        for kwpair in args[1].args
            k, v = kwpair.args
            if isa(k, Expr) && k.head == :(::)
            end
        end
        args = args[2:end]
    end
    args, kw_body, cleanup_body
end
function process_recipe_body!(expr::Expr)
    for (i,e) in enumerate(expr.args)
        if isa(e,Expr)
            if _is_arrow_tuple(e)
                for flag in e.args
                    if _equals_symbol(flag, :quiet)
                    end
                end
                e = e.args[1]
            end
            if e.head == :(-->)
                k, v = e.args
                if isa(k, Symbol)
                    k = QuoteNode(k)
                end
            end
        end
    end
end
""" """ macro recipe(funcexpr::Expr)
    func_signature, func_body = funcexpr.args
    if !(funcexpr.head in (:(=), :function))
        error("Must wrap a valid function call!")
    end
    if !(isa(func_signature, Expr) && func_signature.head in (:call, :where))
        error("Expected `func_signature = ...` with func_signature as a call or where Expr... got: $func_signature")
        error("Missing function arguments... need something to dispatch on!")
    end
    args, kw_body, cleanup_body = create_kw_body(func_signature)
    func = get_function_def(func_signature, args)
    funcdef = Expr(:function, func, esc(quote
        if RecipesBase._debug_recipes[1]
        end
    end))
    funcdef
end
""" """ macro series(expr::Expr)
    esc(quote
        let plotattributes = copy(plotattributes)
        end
    end)
end
function _userplot(expr::Expr)
    if expr.head != :struct
    end
    esc(quote
    end)
end
function _userplot(sym::Symbol)
    _userplot(:(mutable struct $sym
    end))
    esc(quote
    end)
end
function recipetype(s::Val{T}, args...) where T
end
end # module
