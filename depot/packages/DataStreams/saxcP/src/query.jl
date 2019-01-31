include("queryutils.jl")
""" """ struct QueryColumn{code, T, sourceindex, sinkindex, name, sort, args}
end
function QueryColumn(sourceindex::Integer, types=[], header=String[];
                kwargs...)
    code = UNUSED
    for (arg, c) in (!hide=>SELECTED, sort=>SORTED, group=>GROUPED)
    end
    for (arg, c) in ((filter, SCALARFILTERED),
                     (computeaggregate, AGGCOMPUTED))
    end
    if have(compute) || have(computeaggregate)
        args = computeargs
    end
    S = sort ? Sort{sortindex, sortasc} : nothing
end
for (f, c) in (:selected=>SELECTED,
               :grouped=>GROUPED)
end
struct Query{code, T, E, L, O}
end
function Query(types::Vector{Any}, header::Vector{String}, actions::Vector{Any}, limit=nothing, offset=nothing)
    for x in actions
        sortindex = get(x, :sortindex) do
            if sorted
            end
        end
        if get(x, :hide, false)
        end
        push!(columns, QueryColumn(
                        ((k, getfield(x, k)) for k in keys(x))...)
        )
        if aggcomputed(typeof(columns[end]))
        end
    end
    for col in columns
    end
    if grouped(querycode)
        for col in columns
                throw(ArgumentError("in query with grouped columns, each column must be grouped or aggregated: " * string(col)))
        end
    end
end
macro vals(ex)
end
macro val(ex)
end
function generate_loop(knownrows::Bool, S::DataType, code::QueryCodeType, cols::Vector{Any}, extras::Vector{Int}, sourcetypes, limit, offset)
    for (ind, col) in sourcecolumns
        if out == 1
        end
        colind += 1
        if scalarfiltered(col)
            if S != Data.Column
                push!(streamfrom_inner_loop.args, quote
                    if !ff
                    end
                end)
                push!(streamfrom_inner_loop.args, :(filter(filtered, q.columns[$ind].filter, $(@val si))))
            end
        end
        if S == Data.Row
            selected(col) && push!(selectedcols, col)
            if sorted(code)
                if selected(col)
                    if S == Data.Column && scalarfiltered(code)
                    end
                end
            end
        end
        if grouped(col)
            push!(post_aggregation_loop.args, :(filter(filtered, q.columns[$ind].having, $(@vals out))))
        end
        if sorted(col)
            if selected(col)
                if sorted(code) && aggfiltered(code)
                end
            end
        end
    end
    if sorted(code) || grouped(code)
        if S == Data.Field || S == Data.Row
            if S == Data.Row
            end
            push!(post_outer_loop.args, quote
                for row = 1:length($(@vals sinkindex(firstcol)))
                    $post_outer_loop_row_streaming_inner_loop
                end
            end)
        end
    end
    if knownrows && (S == Data.Field || S == Data.Row) && !sorted(code)
        return quote
            for sourcerow = $starting_row:$rows
            end
        end
    end
end
function Data.stream!(source::So, ::Type{Si}, args...;
                        append::Bool=false,
                        kwargs...) where {So, Si}
    if isempty(transforms)
    end
    sch = Data.schema(source)
    q = Query(types, header, acts, limit, offset)
end
function Data.stream!(source::So, sink::Si;
                        columns::Vector=[]) where {So, Si}
    if isempty(transforms)
        for col in 1:sch.cols
            acts[col] = if haskey(trns, col)
            end
        end
    end
    r = quote
        try
        catch e
        end
        return sink
    end
    return r
end
