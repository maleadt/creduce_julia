module RecipesBase
export
    @recipe,
    @series,
    @userplot,
    @shorthands,
    RecipeData,
    AbstractBackend,
    AbstractPlot,
    AbstractLayout
abstract type AbstractBackend end
abstract type AbstractPlot{T<:AbstractBackend} end
abstract type AbstractLayout end
function plot end
function plot! end
function animate end
function is_key_supported end
group_as_matrix(t) = false
apply_recipe(plotattributes::Dict{Symbol,Any}) = ()
const _debug_recipes = Bool[false]
function debug(v::Bool = true)
    _debug_recipes[1] = v
end
struct RecipeData
    plotattributes::Dict{Symbol,Any}
    args::Tuple
end
@inline to_symbol(s::Symbol) = s
@inline to_symbol(qn::QuoteNode) = qn.value
@inline wrap_tuple(tup::Tuple) = tup
@inline wrap_tuple(v) = (v,)
function _is_arrow_tuple(expr::Expr)
    expr.head == :tuple && !isempty(expr.args) &&
        isa(expr.args[1], Expr) &&
        expr.args[1].head == :(-->)
end
function _equals_symbol(arg::Symbol, sym::Symbol)
    arg == sym
end
function _equals_symbol(arg::Expr, sym::Symbol) #not sure this method is necessary anymore on 0.7
    arg.head == :quote && arg.args[1] == sym
end
function _equals_symbol(arg::QuoteNode, sym::Symbol)
    arg.value == sym
end
_equals_symbol(x, sym::Symbol) = false
function get_function_def(func_signature::Expr, args::Vector)
    front = func_signature.args[1]
    if func_signature.head == :where
        Expr(:where, get_function_def(front, args), esc.(func_signature.args[2:end])...)
    elseif func_signature.head == :call
        func = Expr(:call, :(RecipesBase.apply_recipe), esc.([:(plotattributes::Dict{Symbol, Any}); args])...)
        if isa(front, Expr) && front.head == :curly
            Expr(:where, func, esc.(front.args[2:end])...)
        else
            func
        end
    else
        error("Expected `func_signature = ...` with func_signature as a call or where Expr... got: $func_signature")
    end
end
function create_kw_body(func_signature::Expr)
    func_signature.head == :where && return create_kw_body(func_signature.args[1])
    args = func_signature.args[2:end]
    kw_body = Expr(:block)
    cleanup_body = Expr(:block)
    if isa(args[1], Expr) && args[1].head == :parameters
        for kwpair in args[1].args
            k, v = kwpair.args
            if isa(k, Expr) && k.head == :(::)
                k = k.args[1]
                @warn("Type annotations on keyword arguments not currently supported in recipes. Type information has been discarded")
            end
            push!(kw_body.args, :($k = get!(plotattributes, $(QuoteNode(k)), $v)))
            push!(cleanup_body.args, :(RecipesBase.is_key_supported($(QuoteNode(k))) || delete!(plotattributes, $(QuoteNode(k)))))
        end
        args = args[2:end]
    end
    args, kw_body, cleanup_body
end
function process_recipe_body!(expr::Expr)
    for (i,e) in enumerate(expr.args)
        if isa(e,Expr)
            quiet, require, force = false, false, false
            if _is_arrow_tuple(e)
                for flag in e.args
                    if _equals_symbol(flag, :quiet)
                        quiet = true
                    elseif _equals_symbol(flag, :require)
                        require = true
                    elseif _equals_symbol(flag, :force)
                        force = true
                    end
                end
                e = e.args[1]
            end
            if e.head == :(:=)
                force = true
                e.head = :(-->)
            end
            if e.head == :(-->)
                k, v = e.args
                if isa(k, Symbol)
                    k = QuoteNode(k)
                end
                set_expr = if force
                    :(plotattributes[$k] = $v)
                else
                    :(get!(plotattributes, $k, $v))
                end
                expr.args[i] = if quiet
                    :(RecipesBase.is_key_supported($k) ? $set_expr : nothing)
                elseif require
                    :(RecipesBase.is_key_supported($k) ? $set_expr : error("In recipe: required keyword ", $k, " is not supported by backend $(backend_name())"))
                else
                    set_expr
                end
            elseif e.head != :call
                process_recipe_body!(e)
            end
        end
    end
end
"""
This handy macro will process a function definition, replace `-->` commands, and
then add a new version of `RecipesBase.apply_recipe` for dispatching on the arguments.
This functionality is primarily geared to turning user types and settings into the
data and attributes that describe a Plots.jl visualization.
Set attributes using the `-->` command, and return a comma separated list of arguments that
should replace the current arguments.
An example:
```
using RecipesBase
type T end
@recipe function plot{N<:Integer}(t::T, n::N = 1; customcolor = :green)
    markershape --> :auto, :require
    markercolor --> customcolor, :force
    xrotation --> 5
    zrotation --> 6, :quiet
    rand(10,n)
end
using Plots; gr()
plot(T(), 5; customcolor = :black, shape=:c)
```
In this example, we see lots of the machinery in action.  We create a new type `T` which
we will use for dispatch, and an optional argument `n`, which will be used to determine the
number of series to display.  User-defined keyword arguments are passed through, and the
`-->` command can be trailed by flags:
- quiet:   Suppress unsupported keyword warnings
- require: Error if keyword is unsupported
- force:   Don't allow user override for this keyword
"""
macro recipe(funcexpr::Expr)
    func_signature, func_body = funcexpr.args
    if !(funcexpr.head in (:(=), :function))
        error("Must wrap a valid function call!")
    end
    if !(isa(func_signature, Expr) && func_signature.head in (:call, :where))
        error("Expected `func_signature = ...` with func_signature as a call or where Expr... got: $func_signature")
    end
    if length(func_signature.args) < 2
        error("Missing function arguments... need something to dispatch on!")
    end
    args, kw_body, cleanup_body = create_kw_body(func_signature)
    func = get_function_def(func_signature, args)
    process_recipe_body!(func_body)
    funcdef = Expr(:function, func, esc(quote
        if RecipesBase._debug_recipes[1]
            println("apply_recipe args: ", $args)
        end
        $kw_body
        $cleanup_body
        series_list = RecipesBase.RecipeData[]
        func_return = $func_body
        if func_return != nothing
            push!(series_list, RecipesBase.RecipeData(plotattributes, RecipesBase.wrap_tuple(func_return)))
        end
        series_list
    end))
    funcdef
end
"""
Meant to be used inside a recipe to add additional RecipeData objects to the list:
```
@recipe function f(::T)
    linecolor --> :red
    @series begin
        fillcolor := :green
        rand(10)
    end
    rand(100)
end
```
"""
macro series(expr::Expr)
    esc(quote
        let plotattributes = copy(plotattributes)
            args = $expr
            push!(series_list, RecipesBase.RecipeData(plotattributes, RecipesBase.wrap_tuple(args)))
            nothing
        end
    end)
end
"""
You can easily define your own plotting recipes with convenience methods:
```
@userplot GroupHist
@recipe function f(gh::GroupHist)
end
grouphist(rand(1000,4))
```
"""
macro userplot(expr)
    _userplot(expr)
end
function _userplot(expr::Expr)
    if expr.head != :struct
        error("Must call userplot on a [mutable] struct expression.  Got: $expr")
    end
    typename = expr.args[2]
    funcname = Symbol(lowercase(string(typename)))
    funcname2 = Symbol(funcname, "!")
    esc(quote
        $expr
        export $funcname, $funcname2
        Core.@__doc__ $funcname(args...; kw...) = RecipesBase.plot($typename(args); kw...)
        Core.@__doc__ $funcname2(args...; kw...) = RecipesBase.plot!($typename(args); kw...)
    end)
end
function _userplot(sym::Symbol)
    _userplot(:(mutable struct $sym
            args
    end))
end
macro shorthands(funcname::Symbol)
    funcname2 = Symbol(funcname, "!")
    esc(quote
        export $funcname, $funcname2
        Core.@__doc__ $funcname(args...; kw...) = RecipesBase.plot(args...; kw..., seriestype = $(Meta.quot(funcname)))
        Core.@__doc__ $funcname2(args...; kw...) = RecipesBase.plot!(args...; kw..., seriestype = $(Meta.quot(funcname)))
    end)
end
"""
`recipetype(s, args...)`
Use this function to refer to type recipes by their symbol, without taking a dependency.
```julia
import RecipesBase: recipetype
recipetype(:groupedbar, 1:10, rand(10, 2))
```
instead of
```julia
import StatPlots: GroupedBar
GroupedBar((1:10, rand(10, 2)))
```
"""
recipetype(s, args...) = recipetype(Val(s), args...)
function recipetype(s::Val{T}, args...) where T
    error("No type recipe defined for type $T. You may need to load StatPlots")
end
end # module
